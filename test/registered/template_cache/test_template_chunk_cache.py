"""Unit tests for TemplateAwareChunkCache.

Drives the cache directly with mock pools; no model or GPU required.

For CI integration (registration helpers):

    register_cuda_ci(est_time=15, suite="stage-b-test-1-gpu-small")
    register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd")

These are commented out so the file runs on CPU-only dev boxes; uncomment
when wiring into CI.
"""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from typing import List, Optional

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    EvictParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.template_chunk_cache import (
    TemplateAwareChunkCache,
)


# ---------- Mocks ----------


class _MockReqToTokenPool:
    """Minimal stand-in: a 2D int32 tensor + a write() that does fancy indexing."""

    def __init__(self, size: int, max_context_len: int, device: str = "cpu"):
        self.size = size
        self.max_context_len = max_context_len
        self.device = device
        self.req_to_token = torch.zeros(
            (size + 1, max_context_len), dtype=torch.int32, device=device
        )

    def write(self, indices, values):
        self.req_to_token[indices] = values


class _MockKVAllocator:
    """Minimal allocator: tracks free slots, supports page_size>=1.

    For page_size>1, free() reclaims the *whole page* containing each supplied
    slot index (mirroring PagedTokenToKVPoolAllocator). alloc() returns the
    requested number of slots (which we ignore for these tests — the tests
    pre-fill req_to_token themselves).
    """

    def __init__(self, total_slots: int, page_size: int = 1, device: str = "cpu"):
        self.total_slots = total_slots
        self.page_size = page_size
        self.device = device
        self.free_slots = set(range(total_slots))
        self.free_log: List[List[int]] = []  # one entry per free() call

    def free(self, indices: torch.Tensor):
        if indices.numel() == 0:
            self.free_log.append([])
            return
        flat = indices.flatten().tolist()
        if self.page_size == 1:
            unique = sorted(set(flat))
            for s in unique:
                self.free_slots.add(int(s))
            self.free_log.append(unique)
            return
        # paged: reclaim every slot whose page is touched
        pages = sorted({int(s) // self.page_size for s in flat})
        freed_slots: List[int] = []
        for p in pages:
            for s in range(p * self.page_size, (p + 1) * self.page_size):
                if s < self.total_slots:
                    self.free_slots.add(s)
                    freed_slots.append(s)
        self.free_log.append(freed_slots)

    def available_size(self) -> int:
        return len(self.free_slots)


@dataclass
class _MockReq:
    """Minimal Req-like object exposing the fields the cache reads/writes."""

    rid: str
    req_pool_idx: int
    origin_input_ids: List[int]
    fill_ids: List[int]
    template_id: Optional[str]
    segment_boundaries: Optional[List[int]]
    segment_kinds: Optional[List[str]]
    prefix_indices: torch.Tensor = None
    last_node: object = None
    last_host_node: object = None
    cache_protected_len: int = 0
    _committed_kv_len: int = 0

    def __post_init__(self):
        if self.prefix_indices is None:
            self.prefix_indices = torch.empty((0,), dtype=torch.int64)

    def pop_committed_kv_cache(self) -> int:
        v = self._committed_kv_len
        self._committed_kv_len = 0
        return v


def _make_cache(page_size: int = 1, total_slots: int = 64) -> TemplateAwareChunkCache:
    req_pool = _MockReqToTokenPool(size=8, max_context_len=256)
    allocator = _MockKVAllocator(total_slots=total_slots, page_size=page_size)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=req_pool,
        token_to_kv_pool_allocator=allocator,
        page_size=page_size,
    )
    return TemplateAwareChunkCache(params)


def _populate_req(
    cache: TemplateAwareChunkCache,
    req: _MockReq,
    slot_start: int,
):
    """Write contiguous slots [slot_start, slot_start+len(fill_ids)) for req."""
    n = len(req.fill_ids)
    slots = torch.arange(slot_start, slot_start + n, dtype=torch.int32)
    cache.req_to_token_pool.req_to_token[req.req_pool_idx, :n] = slots
    # Mark the allocator as having handed these slots out.
    for s in slots.tolist():
        cache.token_to_kv_pool_allocator.free_slots.discard(int(s))
    req._committed_kv_len = n


# ---------- Tests ----------


class TestTemplateChunkCache(unittest.TestCase):
    TEMPLATE_ID = "t1"
    FIXED_PROMPT = [1, 2, 3, 4, 5]  # 5 tokens
    VAR_A = [10, 11]  # 2 tokens
    VAR_B = [20, 21, 22]  # 3 tokens

    def _make_req(
        self,
        rid: str,
        req_pool_idx: int,
        var_tokens: List[int],
        with_template: bool = True,
    ) -> _MockReq:
        full = list(self.FIXED_PROMPT) + list(var_tokens)
        if with_template:
            boundaries = [0, len(self.FIXED_PROMPT), len(full)]
            kinds = ["fixed", "var"]
            template_id = self.TEMPLATE_ID
        else:
            boundaries = None
            kinds = None
            template_id = None
        return _MockReq(
            rid=rid,
            req_pool_idx=req_pool_idx,
            origin_input_ids=full,
            fill_ids=full,
            template_id=template_id,
            segment_boundaries=boundaries,
            segment_kinds=kinds,
        )

    # -- match/insert basics --

    def test_first_request_misses_then_inserts(self):
        cache = _make_cache()
        req = self._make_req("r1", req_pool_idx=1, var_tokens=self.VAR_A)
        _populate_req(cache, req, slot_start=10)

        # Initial match should miss (cache is empty).
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(req.fill_ids), req=req))
        self.assertEqual(m.device_indices.numel(), 0)
        self.assertIs(m.last_device_node, cache._root)

        # Finalize: this should insert two chunks (fixed + var).
        cache.cache_finished_req(req)
        self.assertEqual(len(cache.chunks), 1 + 2)  # root + 2

        # Evictable size = total inserted tokens (lock has been released).
        self.assertEqual(
            cache.evictable_size(), len(self.FIXED_PROMPT) + len(self.VAR_A)
        )
        self.assertEqual(cache.protected_size(), 0)

    def test_second_request_hits_full_prefix(self):
        cache = _make_cache()
        r1 = self._make_req("r1", req_pool_idx=1, var_tokens=self.VAR_A)
        _populate_req(cache, r1, slot_start=10)
        cache.cache_finished_req(r1)

        # Identical request — should hit both segments.
        r2 = self._make_req("r2", req_pool_idx=2, var_tokens=self.VAR_A)
        # We do NOT populate r2's slots in advance; match is read-only.
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(r2.fill_ids), req=r2))
        self.assertEqual(
            m.device_indices.numel(), len(self.FIXED_PROMPT) + len(self.VAR_A)
        )
        self.assertIsNotNone(m.last_device_node)
        self.assertIsNot(m.last_device_node, cache._root)

    def test_partial_hit_fixed_shared_var_differs(self):
        cache = _make_cache()
        r1 = self._make_req("r1", req_pool_idx=1, var_tokens=self.VAR_A)
        _populate_req(cache, r1, slot_start=10)
        cache.cache_finished_req(r1)

        r2 = self._make_req("r2", req_pool_idx=2, var_tokens=self.VAR_B)
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(r2.fill_ids), req=r2))
        # Only the fixed segment hits.
        self.assertEqual(m.device_indices.numel(), len(self.FIXED_PROMPT))

    def test_different_templates_do_not_share(self):
        cache = _make_cache()
        r1 = self._make_req("r1", req_pool_idx=1, var_tokens=self.VAR_A)
        _populate_req(cache, r1, slot_start=10)
        cache.cache_finished_req(r1)

        r2 = self._make_req("r2", req_pool_idx=2, var_tokens=self.VAR_A)
        r2.template_id = "t2"  # different template
        # Same tokens, different template id => no match.
        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(r2.fill_ids), req=r2))
        self.assertEqual(m.device_indices.numel(), 0)

    # -- fall-through (no template_id) --

    def test_fall_through_when_no_template_id(self):
        cache = _make_cache()
        req = self._make_req(
            "r1", req_pool_idx=1, var_tokens=self.VAR_A, with_template=False
        )
        _populate_req(cache, req, slot_start=10)

        m = cache.match_prefix(MatchPrefixParams(key=RadixKey(req.fill_ids), req=req))
        self.assertEqual(m.device_indices.numel(), 0)

        # cache_finished_req should free everything and not insert chunks.
        cache.cache_finished_req(req)
        self.assertEqual(len(cache.chunks), 1)  # only root
        self.assertEqual(cache.evictable_size(), 0)

    # -- LRU eviction order --

    def test_lru_evicts_oldest_leaf(self):
        cache = _make_cache()

        # Insert template_id=tA fully, then template_id=tB fully.
        rA = self._make_req("rA", req_pool_idx=1, var_tokens=self.VAR_A)
        rA.template_id = "tA"
        _populate_req(cache, rA, slot_start=10)
        cache.cache_finished_req(rA)

        rB = self._make_req("rB", req_pool_idx=2, var_tokens=self.VAR_B)
        rB.template_id = "tB"
        _populate_req(cache, rB, slot_start=30)
        cache.cache_finished_req(rB)

        # Touch rA so it becomes MRU.
        rA_again = self._make_req("rA2", req_pool_idx=3, var_tokens=self.VAR_A)
        rA_again.template_id = "tA"
        cache.match_prefix(
            MatchPrefixParams(key=RadixKey(rA_again.fill_ids), req=rA_again)
        )

        # Evict some tokens; rB's var leaf (LRU) should go first.
        before = len(cache.chunks)
        result = cache.evict(EvictParams(num_tokens=1))
        self.assertGreater(result.num_tokens_evicted, 0)
        self.assertLess(len(cache.chunks), before)

    def test_lock_ref_prevents_eviction(self):
        cache = _make_cache()
        r1 = self._make_req("r1", req_pool_idx=1, var_tokens=self.VAR_A)
        _populate_req(cache, r1, slot_start=10)
        cache.cache_finished_req(r1)

        # Lock the leaf chunk.
        leaf = None
        for ck, ch in cache.chunks.items():
            if ch is not cache._root and not ch.children:
                leaf = ch
                break
        self.assertIsNotNone(leaf)
        cache.inc_lock_ref(leaf)

        before = cache.evictable_size()
        result = cache.evict(EvictParams(num_tokens=1000))
        # Nothing evicted because the only leaf and its parent are locked.
        self.assertEqual(result.num_tokens_evicted, 0)

        cache.dec_lock_ref(leaf, DecLockRefParams())
        result = cache.evict(EvictParams(num_tokens=1000))
        self.assertGreater(result.num_tokens_evicted, 0)

    # -- page-refcount sharing (page_size > 1) --

    def test_page_refcount_holds_shared_boundary_page(self):
        """page_size=4: fixed (5 tokens) and var (2 tokens) share page 2.

        Slot layout (slot_start=8 to land on a page boundary): fixed=[8..13),
        var=[13..15). Page 2 = slots 8..12; page 3 = slots 12..15 — fixed and
        var share page 3. Evicting var alone must NOT free page 3, since the
        fixed chunk still owns slot 12.
        """
        cache = _make_cache(page_size=4)
        req = self._make_req("r1", req_pool_idx=1, var_tokens=self.VAR_A)

        n = len(req.fill_ids)
        slot_start = 8
        slots = torch.arange(slot_start, slot_start + n, dtype=torch.int32)
        cache.req_to_token_pool.req_to_token[req.req_pool_idx, :n] = slots
        for s in slots.tolist():
            cache.token_to_kv_pool_allocator.free_slots.discard(int(s))
        req._committed_kv_len = n

        cache.cache_finished_req(req)
        self.assertEqual(len(cache.chunks), 1 + 2)

        # Identify fixed (idx=0) and var (idx=1) chunks.
        fixed = next(
            c for c in cache.chunks.values()
            if c is not cache._root and c.segment_index == 0
        )
        var = next(
            c for c in cache.chunks.values()
            if c is not cache._root and c.segment_index == 1
        )

        # Both chunks should touch page 3 (slot 12 is in page 3).
        shared_page = 12 // 4
        self.assertIn(shared_page, fixed.touched_pages)
        self.assertIn(shared_page, var.touched_pages)
        self.assertEqual(cache.page_refcount[shared_page], 2)

        # Evict var only — shared page must survive, refcount drops to 1.
        cache._evict_chunk(var)
        self.assertEqual(cache.page_refcount[shared_page], 1)

        # Page 3 slots (12,13,14,15) must NOT all be back in the free pool;
        # slot 12 belongs to fixed.
        free_slots = cache.token_to_kv_pool_allocator.free_slots
        self.assertNotIn(12, free_slots)

        # Now evict fixed — page 3 should be fully reclaimed.
        cache._evict_chunk(fixed)
        self.assertNotIn(shared_page, cache.page_refcount)
        for s in range(8, 16):
            self.assertIn(s, cache.token_to_kv_pool_allocator.free_slots)


if __name__ == "__main__":
    unittest.main()
