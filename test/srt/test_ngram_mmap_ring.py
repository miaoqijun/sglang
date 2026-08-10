import threading
import time

import numpy as np
import pytest

from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus
from sglang.srt.speculative.ngram_history import (
    NgramHistoryConfig,
    NgramHistoryReplicator,
)
from sglang.srt.speculative.ngram_mmap_ring import MmapNgramHistoryBackend


def _csr(rows):
    flat = np.asarray([token for row in rows for token in row], dtype=np.int32)
    offsets = np.asarray(
        [0] + list(np.cumsum([len(row) for row in rows])), dtype=np.int64
    )
    return flat, offsets


def _backend(tmp_path, writer_id, *, capacity=32):
    return MmapNgramHistoryBackend(
        root_path=str(tmp_path / "rings"),
        namespace="test",
        writer_id=writer_id,
        capacity=capacity,
        max_trie_depth=4,
        start_from_beginning=True,
        max_records=32,
        max_tokens=128,
    )


def _read_rows(backend, *, records=32, tokens=128):
    flat, offsets, _ = backend.read_windows_wait(
        records, tokens, timeout_ms=20, scan_interval_ms=2
    )
    return [
        flat[int(offsets[i]) : int(offsets[i + 1])].tolist()
        for i in range(offsets.size - 1)
    ]


def _wait_for(predicate, timeout=5.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_window_csr_roundtrip_and_stats(tmp_path):
    left = _backend(tmp_path, "left")
    right = _backend(tmp_path, "right")
    try:
        flat, offsets = _csr([[1, 2], [3, 4, 5]])
        assert left.append_windows_csr(flat, offsets) == 2
        assert _read_rows(right) == [[1, 2], [3, 4, 5]]
        assert left._extension.SLOT_SIZE == 128
        assert left.stats()["published_windows"] == 2
        assert right.stats()["polled_windows"] == 2
    finally:
        left.close()
        right.close()


def test_csr_validation_is_atomic(tmp_path):
    left = _backend(tmp_path, "left")
    right = _backend(tmp_path, "right")
    try:
        flat = np.arange(5, dtype=np.int32)
        with pytest.raises((ValueError, RuntimeError)):
            left.append_windows_csr(flat, np.asarray([0, 2, 1, 5], np.int64))
        assert _read_rows(right) == []
        with pytest.raises(ValueError):
            left.append_windows_csr(flat.astype(np.int64), np.asarray([0, 5]))
    finally:
        left.close()
        right.close()


def test_overrun_keeps_latest_complete_windows(tmp_path):
    left = _backend(tmp_path, "left", capacity=4)
    right = _backend(tmp_path, "right", capacity=4)
    try:
        # Attach the reader before advancing beyond its capacity.
        assert _read_rows(right) == []
        flat, offsets = _csr([[1], [2]])
        left.append_windows_csr(flat, offsets)
        assert _read_rows(right) == [[1], [2]]

        flat, offsets = _csr([[value] for value in range(3, 10)])
        left.append_windows_csr(flat, offsets)
        assert _read_rows(right) == [[6], [7], [8], [9]]
        assert right.stats()["overrun_events"] >= 1
        assert right.stats()["reset_events"] >= 1
    finally:
        left.close()
        right.close()


def test_native_wait_wakes_for_new_windows(tmp_path):
    left = _backend(tmp_path, "left")
    right = _backend(tmp_path, "right")
    try:
        assert _read_rows(right) == []

        def publish():
            time.sleep(0.03)
            left.append_windows_csr(*_csr([[7, 8, 9]]))

        thread = threading.Thread(target=publish)
        thread.start()
        flat, offsets, _ = right.read_windows_wait(
            32, 128, timeout_ms=200, scan_interval_ms=2
        )
        thread.join()
        assert flat.tolist() == [7, 8, 9]
        assert offsets.tolist() == [0, 3]
        assert right.stats()["wait_wakeups"] >= 1
    finally:
        left.close()
        right.close()


def test_replicator_stages_native_window_epoch(tmp_path):
    corpus = NgramCorpus(
        max_trie_depth=4,
        min_bfs_breadth=1,
        max_bfs_breadth=1,
        draft_token_num=4,
        match_type="BFS",
        capacity=100_000,
    )
    common = dict(
        path=str(tmp_path / "rings"),
        namespace="test",
        mmap_capacity=32,
        max_trie_depth=4,
        pull_interval_ms=2,
        import_max_records_per_step=32,
        import_max_tokens_per_step=128,
        start_from_beginning=True,
    )
    left = NgramHistoryReplicator(
        NgramHistoryConfig(writer_id="left", **common),
        remote_epoch_sink=lambda _flat, _offsets: 0,
    )
    right = NgramHistoryReplicator(
        NgramHistoryConfig(writer_id="right", **common),
        remote_epoch_sink=corpus.stage_remote_windows,
    )
    try:
        left.enqueue_push(*_csr([[10, 11, 12], [11, 12, 13]]))
        assert _wait_for(
            lambda: corpus.insert_stats()["remote_staged_ticket"] >= 1
        )
        ticket = corpus.release_remote_epochs()
        corpus.wait_remote(ticket)
        tokens, _ = corpus.batch_get(["query"], [[10, 11]], [2])
        assert 12 in tokens.tolist()
        stats = corpus.insert_stats()
        assert stats["remote_epoch_tasks"] == 2
        assert stats["window_records"] == 2
        assert stats["squeeze_calls"] == 0
        assert left.dropped_push_records == 0
        assert right.dropped_import_records == 0
    finally:
        left.close()
        right.close()
