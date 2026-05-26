"""Template-aware chunk prefix cache.

For requests that arrive with a registered prompt-template id and a list of
segment boundaries, this cache stores each segment's KV as a discrete `Chunk`.
Chunks are hash-chained by `parent_chunk_key` so two requests share KV iff they
share an identical prefix of (template_id, segment_kind, segment_content) up to
some point. Requests without a `template_id` fall through to chunk-cache-like
behavior (no prefix sharing, no insertions).

Eviction is LRU at the chunk level; only leaf chunks (no live children, no
lock refs) are evicted. KV slots are returned to the allocator with page-level
reference counting so the paged allocator (which frees in whole pages) stays
correct when neighbouring chunks share a tail page.
"""

from __future__ import annotations

import hashlib
import heapq
import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import torch

from sglang.srt.mem_cache.base_prefix_cache import (
    BasePrefixCache,
    DecLockRefParams,
    DecLockRefResult,
    EvictParams,
    EvictResult,
    IncLockRefResult,
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams


logger = logging.getLogger(__name__)


ROOT_CHUNK_KEY: int = 0


class Chunk:
    __slots__ = (
        "chunk_key",
        "template_id",
        "segment_index",
        "segment_kind",
        "parent_chunk_key",
        "kv_indices",
        "touched_pages",
        "children",
        "lock_ref",
        "last_access_time",
    )

    def __init__(
        self,
        chunk_key: int,
        template_id: Optional[str],
        segment_index: int,
        segment_kind: str,
        parent_chunk_key: Optional[int],
        kv_indices: torch.Tensor,
        touched_pages: Tuple[int, ...],
    ):
        self.chunk_key = chunk_key
        self.template_id = template_id
        self.segment_index = segment_index
        self.segment_kind = segment_kind
        self.parent_chunk_key = parent_chunk_key
        self.kv_indices = kv_indices
        self.touched_pages = touched_pages
        self.children: Dict[int, "Chunk"] = {}
        self.lock_ref: int = 0
        self.last_access_time: float = time.monotonic()


def _hash_tokens(tokens: List[int]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for t in tokens:
        h.update(int(t).to_bytes(8, "little", signed=True))
    return h.digest()


def _compose_chunk_key(
    parent_chunk_key: int,
    template_id: str,
    segment_index: int,
    segment_kind: str,
    content_digest: bytes,
) -> int:
    h = hashlib.blake2b(digest_size=8)
    h.update(int(parent_chunk_key).to_bytes(8, "little", signed=False))
    h.update(template_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(int(segment_index).to_bytes(4, "little", signed=False))
    h.update(segment_kind.encode("utf-8"))
    h.update(b"\x00")
    h.update(content_digest)
    # Reserve 0 as a sentinel for the synthetic root.
    key = int.from_bytes(h.digest(), "little", signed=False)
    return key if key != ROOT_CHUNK_KEY else 1


class TemplateAwareChunkCache(BasePrefixCache):
    """Prefix cache keyed by template segment, with page-refcount eviction."""

    def __init__(self, params: "CacheInitParams"):
        self.disable = params.disable
        self.req_to_token_pool = params.req_to_token_pool
        self.token_to_kv_pool_allocator = params.token_to_kv_pool_allocator
        self.page_size = params.page_size

        if self.token_to_kv_pool_allocator is not None:
            dev = self.token_to_kv_pool_allocator.device
            self.device = torch.device(dev) if isinstance(dev, (str, torch.device)) else torch.device("cpu")
        else:
            self.device = torch.device("cpu")

        if params.enable_metrics:
            self.init_metrics_collector()

        # Synthetic root: parent of all top-level chunks; always locked.
        self._root = Chunk(
            chunk_key=ROOT_CHUNK_KEY,
            template_id=None,
            segment_index=-1,
            segment_kind="root",
            parent_chunk_key=None,
            kv_indices=torch.empty((0,), dtype=torch.int64, device=self.device),
            touched_pages=tuple(),
        )
        self._root.lock_ref = 1

        self.chunks: Dict[int, Chunk] = {ROOT_CHUNK_KEY: self._root}
        # Reference count per physical page across all live chunks.
        self.page_refcount: Dict[int, int] = defaultdict(int)

        self.evictable_size_: int = 0
        self.protected_size_: int = 0

        self._empty_indices = torch.empty(
            (0,), dtype=torch.int64, device=self.device
        )

    # ---------------- BasePrefixCache API ----------------

    def reset(self):
        # Release any allocator-held slots first.
        for chunk in list(self.chunks.values()):
            if chunk is self._root:
                continue
            self.token_to_kv_pool_allocator.free(chunk.kv_indices)
        self._root.children.clear()
        self.chunks = {ROOT_CHUNK_KEY: self._root}
        self.page_refcount.clear()
        self.evictable_size_ = 0
        self.protected_size_ = 0

    def is_chunk_cache(self) -> bool:
        # We expose a tree-like API (chunks form a parent chain), so
        # downstream is_tree_cache() returns True. This matches RadixCache.
        return False

    @property
    def root_node(self):
        # zero_match_result() in base_prefix_cache references tree_cache.root_node.
        return self._root

    # ---------------- Match / insert ----------------

    def match_prefix(self, params: MatchPrefixParams) -> MatchResult:
        if self.disable:
            return self._miss_result()

        req = params.req
        if req is None:
            return self._miss_result()
        template_id = getattr(req, "template_id", None)
        if template_id is None:
            return self._miss_result()

        boundaries: Optional[List[int]] = getattr(req, "segment_boundaries", None)
        kinds: Optional[List[str]] = getattr(req, "segment_kinds", None)
        if not boundaries or not kinds:
            return self._miss_result()

        # The fill_ids is the authoritative token sequence; key.token_ids is
        # equivalent but may be truncated by max_prefix_len upstream.
        max_prefix_len = len(params.key.token_ids) if params.key is not None else 0
        token_source = req.fill_ids if req.fill_ids else req.origin_input_ids
        if not token_source:
            return self._miss_result()

        access_time = time.monotonic()
        device_indices: List[torch.Tensor] = []
        last_chunk = self._root
        parent_chunk_key = ROOT_CHUNK_KEY
        for seg_idx in range(len(kinds)):
            seg_start = boundaries[seg_idx]
            seg_end = boundaries[seg_idx + 1]
            # Only consider segments fully within the prefix-matching window.
            if seg_end > max_prefix_len or seg_end > len(token_source):
                break
            seg_tokens = token_source[seg_start:seg_end]
            content_digest = _hash_tokens(seg_tokens)
            chunk_key = _compose_chunk_key(
                parent_chunk_key,
                template_id,
                seg_idx,
                kinds[seg_idx],
                content_digest,
            )
            chunk = self.chunks.get(chunk_key)
            if chunk is None:
                break
            chunk.last_access_time = access_time
            device_indices.append(chunk.kv_indices)
            last_chunk = chunk
            parent_chunk_key = chunk_key

        if device_indices:
            indices_tensor = torch.cat(device_indices)
        else:
            indices_tensor = self._empty_indices
        return MatchResult(
            device_indices=indices_tensor,
            last_device_node=last_chunk,
            last_host_node=last_chunk,
        )

    def insert(self, params: InsertParams) -> InsertResult:
        # Template cache does not accept arbitrary key/value inserts; all
        # insertions are driven by cache_(un)finished_req.
        return InsertResult(prefix_len=0)

    def cache_unfinished_req(self, req: "Req", chunked: bool = False):
        if self.disable:
            return
        if getattr(req, "template_id", None) is None:
            # Fall-through to chunk-cache behavior: just point prefix_indices
            # at the request's own slots.
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, : len(req.fill_ids)
            ]
            req.prefix_indices = kv_indices.to(dtype=torch.int64, copy=True)
            return

        self._cache_req(req, end_offset=len(req.fill_ids), finished=False)

    def cache_finished_req(self, req: "Req", is_insert: bool = True, **kwargs):
        kv_committed_len = req.pop_committed_kv_cache()

        if self.disable or getattr(req, "template_id", None) is None:
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(kv_indices)
            return

        if not is_insert:
            # Free everything not yet protected and release lock.
            kv_indices = self.req_to_token_pool.req_to_token[
                req.req_pool_idx, :kv_committed_len
            ]
            self.token_to_kv_pool_allocator.free(
                kv_indices[req.cache_protected_len:]
            )
            if req.last_node is not None:
                self.dec_lock_ref(req.last_node)
            return

        self._cache_req(req, end_offset=kv_committed_len, finished=True)

        # Release the lock acquired during the last cache_unfinished_req /
        # newly acquired in this call.
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

    def evict(self, params: EvictParams) -> EvictResult:
        if self.disable:
            return EvictResult()
        num_tokens = params.num_tokens
        if num_tokens <= 0:
            return EvictResult()

        start_time = time.perf_counter()
        # Seed heap with currently evictable leaves.
        candidates: List[Tuple[float, int, Chunk]] = []
        for chunk in self.chunks.values():
            if (
                chunk is self._root
                or chunk.lock_ref > 0
                or chunk.children
            ):
                continue
            candidates.append(
                (chunk.last_access_time, chunk.chunk_key, chunk)
            )
        heapq.heapify(candidates)

        num_evicted = 0
        while num_evicted < num_tokens and candidates:
            _ts, _key, chunk = heapq.heappop(candidates)
            # Stale heap entry: chunk may already be gone or no longer a leaf.
            if (
                chunk.chunk_key not in self.chunks
                or chunk.lock_ref > 0
                or chunk.children
            ):
                continue
            evicted_tokens = len(chunk.kv_indices)
            self._evict_chunk(chunk)
            num_evicted += evicted_tokens
            # Parent may now be a leaf; reconsider it.
            parent = (
                self.chunks.get(chunk.parent_chunk_key)
                if chunk.parent_chunk_key is not None
                else None
            )
            if (
                parent is not None
                and parent is not self._root
                and not parent.children
                and parent.lock_ref == 0
            ):
                heapq.heappush(
                    candidates,
                    (parent.last_access_time, parent.chunk_key, parent),
                )

        self.update_eviction_metrics(num_evicted, start_time)
        return EvictResult(num_tokens_evicted=num_evicted)

    def inc_lock_ref(self, node: Any) -> IncLockRefResult:
        if self.disable or node is None or node is self._root:
            return IncLockRefResult(delta=0)
        delta = 0
        chunk = node
        while chunk is not None and chunk is not self._root:
            if chunk.lock_ref == 0:
                self.evictable_size_ -= len(chunk.kv_indices)
                self.protected_size_ += len(chunk.kv_indices)
                delta -= len(chunk.kv_indices)
            chunk.lock_ref += 1
            chunk = (
                self.chunks.get(chunk.parent_chunk_key)
                if chunk.parent_chunk_key is not None
                else None
            )
        return IncLockRefResult(delta=delta)

    def dec_lock_ref(
        self, node: Any, params: Optional[DecLockRefParams] = None
    ) -> DecLockRefResult:
        if self.disable or node is None or node is self._root:
            return DecLockRefResult(delta=0)
        delta = 0
        chunk = node
        while chunk is not None and chunk is not self._root:
            if chunk.lock_ref == 1:
                self.evictable_size_ += len(chunk.kv_indices)
                self.protected_size_ -= len(chunk.kv_indices)
                delta += len(chunk.kv_indices)
            chunk.lock_ref -= 1
            chunk = (
                self.chunks.get(chunk.parent_chunk_key)
                if chunk.parent_chunk_key is not None
                else None
            )
        return DecLockRefResult(delta=delta)

    def evictable_size(self) -> int:
        return self.evictable_size_

    def protected_size(self) -> int:
        return self.protected_size_

    def total_size(self) -> int:
        return self.evictable_size_ + self.protected_size_

    def pretty_print(self) -> str:
        lines: List[str] = [
            f"TemplateAwareChunkCache: {len(self.chunks) - 1} chunks, "
            f"evictable={self.evictable_size_}, protected={self.protected_size_}"
        ]

        def _walk(node: Chunk, depth: int):
            for child in node.children.values():
                lines.append(
                    "  " * depth
                    + f"[{child.template_id}/{child.segment_index}/{child.segment_kind}] "
                    + f"len={len(child.kv_indices)} lock={child.lock_ref}"
                )
                _walk(child, depth + 1)

        _walk(self._root, 1)
        text = "\n".join(lines)
        print(text)
        return text

    # ---------------- Helpers ----------------

    def _miss_result(self) -> MatchResult:
        return MatchResult(
            device_indices=self._empty_indices,
            last_device_node=self._root,
            last_host_node=self._root,
        )

    def _cache_req(self, req: "Req", end_offset: int, finished: bool) -> None:
        template_id: str = req.template_id
        boundaries: List[int] = req.segment_boundaries
        kinds: List[str] = req.segment_kinds
        token_source = req.fill_ids if req.fill_ids else req.origin_input_ids

        kv_indices_all = self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :end_offset
        ]

        new_prefix_parts: List[torch.Tensor] = []
        parent_chunk_key = ROOT_CHUNK_KEY
        last_chunk: Chunk = self._root
        cached_end = 0

        for seg_idx in range(len(kinds)):
            seg_start = boundaries[seg_idx]
            seg_end = boundaries[seg_idx + 1]
            if seg_end > end_offset:
                break
            seg_tokens = token_source[seg_start:seg_end]
            content_digest = _hash_tokens(seg_tokens)
            chunk_key = _compose_chunk_key(
                parent_chunk_key,
                template_id,
                seg_idx,
                kinds[seg_idx],
                content_digest,
            )

            existing = self.chunks.get(chunk_key)
            if seg_end <= req.cache_protected_len:
                # Segment was inserted in a previous cache_unfinished_req call
                # of this same request — chunk is held alive by the lock on
                # req.last_node, so we just collect its indices.
                if existing is None:
                    logger.warning(
                        "template-cache: protected chunk %s missing (rid=%s seg=%d)",
                        chunk_key,
                        req.rid,
                        seg_idx,
                    )
                    break
                existing.last_access_time = time.monotonic()
                new_prefix_parts.append(existing.kv_indices)
                parent_chunk_key = chunk_key
                last_chunk = existing
                cached_end = seg_end
                continue

            if existing is not None:
                # Cache hit on a chunk inserted by another request. Free our
                # duplicate slots and rewrite req_to_token_pool to point at the
                # cached slots.
                duplicate_slots = kv_indices_all[seg_start:seg_end]
                self.token_to_kv_pool_allocator.free(duplicate_slots)
                self.req_to_token_pool.write(
                    (req.req_pool_idx, slice(seg_start, seg_end)),
                    existing.kv_indices,
                )
                existing.last_access_time = time.monotonic()
                new_prefix_parts.append(existing.kv_indices)
                parent_chunk_key = chunk_key
                last_chunk = existing
                cached_end = seg_end
                continue

            # Miss: insert this segment as a new chunk, transferring slot
            # ownership from the request to the cache.
            seg_kv = kv_indices_all[seg_start:seg_end].to(
                dtype=torch.int64, copy=True
            )
            new_chunk = self._insert_chunk(
                chunk_key=chunk_key,
                template_id=template_id,
                segment_index=seg_idx,
                segment_kind=kinds[seg_idx],
                parent_chunk_key=parent_chunk_key,
                kv_indices=seg_kv,
            )
            new_prefix_parts.append(new_chunk.kv_indices)
            parent_chunk_key = chunk_key
            last_chunk = new_chunk
            cached_end = seg_end

        # Build prefix_indices: cached chunks' kv + any uncached tail.
        if new_prefix_parts:
            cached_tensor = torch.cat(new_prefix_parts)
        else:
            cached_tensor = self._empty_indices

        if cached_end < end_offset:
            tail = kv_indices_all[cached_end:end_offset]
            req.prefix_indices = torch.cat([cached_tensor, tail])
        else:
            req.prefix_indices = cached_tensor

        # Migrate lock from the previous last_node to the new one.
        prev_last = req.last_node
        if last_chunk is not self._root:
            self.inc_lock_ref(last_chunk)
        if prev_last is not None and prev_last is not last_chunk:
            self.dec_lock_ref(prev_last)
        req.last_node = last_chunk if last_chunk is not self._root else None
        req.last_host_node = req.last_node
        req.cache_protected_len = cached_end

        if finished:
            # Free the uncached tail (it was the last decode step's KV that
            # we don't want to cache as a partial chunk).
            if cached_end < end_offset:
                self.token_to_kv_pool_allocator.free(
                    kv_indices_all[cached_end:end_offset]
                )

    def _insert_chunk(
        self,
        chunk_key: int,
        template_id: str,
        segment_index: int,
        segment_kind: str,
        parent_chunk_key: int,
        kv_indices: torch.Tensor,
    ) -> Chunk:
        touched_pages = self._compute_touched_pages(kv_indices)
        chunk = Chunk(
            chunk_key=chunk_key,
            template_id=template_id,
            segment_index=segment_index,
            segment_kind=segment_kind,
            parent_chunk_key=parent_chunk_key,
            kv_indices=kv_indices,
            touched_pages=touched_pages,
        )
        self.chunks[chunk_key] = chunk
        parent = self.chunks.get(parent_chunk_key, self._root)
        parent.children[chunk_key] = chunk
        for p in touched_pages:
            self.page_refcount[p] += 1
        self.evictable_size_ += len(kv_indices)
        return chunk

    def _evict_chunk(self, chunk: Chunk) -> None:
        del self.chunks[chunk.chunk_key]
        parent = (
            self.chunks.get(chunk.parent_chunk_key)
            if chunk.parent_chunk_key is not None
            else None
        )
        if parent is not None:
            parent.children.pop(chunk.chunk_key, None)
        self.evictable_size_ -= len(chunk.kv_indices)

        # Page-refcount accounting: free whole pages only when no live chunk
        # references them. For the paged allocator, free() reclaims the WHOLE
        # page containing the supplied slot, so freeing one slot per page is
        # both sufficient and required.
        pages_to_free: List[int] = []
        for p in chunk.touched_pages:
            self.page_refcount[p] -= 1
            if self.page_refcount[p] <= 0:
                del self.page_refcount[p]
                pages_to_free.append(p)

        if pages_to_free:
            if self.page_size == 1:
                # All slot indices in the chunk's range map to pages we are
                # about to free; pass them all so the (page_size==1) allocator
                # reclaims exactly those slots.
                self.token_to_kv_pool_allocator.free(chunk.kv_indices)
            else:
                # For paged allocator, one slot per freed page is sufficient.
                free_idx = torch.tensor(
                    [p * self.page_size for p in pages_to_free],
                    dtype=torch.int64,
                    device=self.device,
                )
                self.token_to_kv_pool_allocator.free(free_idx)

    def _compute_touched_pages(self, kv_indices: torch.Tensor) -> Tuple[int, ...]:
        if kv_indices.numel() == 0:
            return tuple()
        if self.page_size == 1:
            # Each slot is its own page; ordered unique == sorted unique.
            pages = torch.unique(kv_indices).tolist()
        else:
            pages = torch.unique(kv_indices // self.page_size).tolist()
        return tuple(int(p) for p in pages)
