from __future__ import annotations

import atexit
import logging
import os
import queue
import socket
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

from sglang.srt.speculative.ngram_mmap_ring import MmapNgramHistoryBackend

logger = logging.getLogger(__name__)


@dataclass
class NgramHistoryConfig:
    path: str
    namespace: str
    writer_id: str
    push_max_records: int = 256
    pull_interval_ms: int = 5
    import_max_records_per_step: int = 512
    import_max_tokens_per_step: int = 8192
    queue_size: int = 8192
    start_from_beginning: bool = False
    max_trie_depth: int = 18
    mmap_capacity: int = 65536


class NgramHistoryReplicator:
    """Replicate fixed NGRAM windows through a native mmap data plane.

    One queue item is one scheduler CSR batch. No per-window Python record is
    created on either side: the publisher writes CSR rows directly to ring
    slots, and the receiver stages a CSR view directly into the C++ corpus.
    """

    def __init__(
        self,
        config: NgramHistoryConfig,
        backend: Optional[Any] = None,
        remote_epoch_sink: Optional[
            Callable[[np.ndarray, np.ndarray], int]
        ] = None,
    ) -> None:
        self.config = config
        pull_records = config.import_max_records_per_step
        self.backend = backend or MmapNgramHistoryBackend(
            config.path,
            config.namespace,
            config.writer_id,
            capacity=config.mmap_capacity,
            max_trie_depth=config.max_trie_depth,
            start_from_beginning=config.start_from_beginning,
            max_records=pull_records,
            max_tokens=config.import_max_tokens_per_step,
        )
        self._remote_epoch_sink = remote_epoch_sink
        if self._remote_epoch_sink is None:
            raise ValueError("NGRAM history requires a native remote epoch sink")

        self._pull_records = pull_records
        self._wait_timeout_ms = 200
        self._wait_scan_interval_ms = max(
            1, min(int(config.pull_interval_ms), self._wait_timeout_ms)
        )
        self._push_queue: queue.Queue[tuple[np.ndarray, np.ndarray]] = queue.Queue(
            config.queue_size
        )
        self._stop_event = threading.Event()
        self._push_event = threading.Event()
        self._sink_lock = threading.RLock()
        self._cursor = self.backend.end_offset()

        self.pushed_records = 0
        self.pulled_records = 0
        self.imported_records = 0
        self.dropped_push_records = 0
        self.dropped_import_records = 0
        self.published_batches = 0
        self.published_tokens = 0
        self.push_loop_wakeups = 0
        self.push_loop_flush_calls = 0
        self.push_loop_empty_wakeups = 0
        self._last_import_log_time = 0.0

        self._push_thread = threading.Thread(
            target=self._push_loop, name="ngram-history-push", daemon=True
        )
        self._pull_thread = threading.Thread(
            target=self._pull_loop, name="ngram-history-pull", daemon=True
        )
        self._push_thread.start()
        self._pull_thread.start()
        atexit.register(self.close)
        logger.info(
            "Enabled native mmap NGRAM history: path=%s namespace=%s writer_id=%s",
            self.config.path,
            self.config.namespace,
            self.config.writer_id,
        )

    @classmethod
    def from_server_args(
        cls,
        server_args,
        *,
        tp_rank: int,
        dp_rank: Optional[int],
        remote_epoch_sink: Callable[[np.ndarray, np.ndarray], int],
    ) -> Optional["NgramHistoryReplicator"]:
        path = server_args.speculative_ngram_l2_history_path
        if path is None:
            return None
        namespace = server_args.speculative_ngram_l2_namespace
        if namespace is None:
            model = getattr(server_args, "served_model_name", None) or getattr(
                server_args, "model_path", "unknown-model"
            )
            namespace = (
                f"{model}|draft={server_args.speculative_num_draft_tokens}"
                f"|depth={server_args.speculative_ngram_max_trie_depth}"
                f"|match={server_args.speculative_ngram_match_type}"
            )

        writer_id = server_args.speculative_ngram_l2_instance_id
        if writer_id is None:
            dp = "none" if dp_rank is None else str(dp_rank)
            writer_id = f"{socket.gethostname()}:{os.getpid()}:tp{tp_rank}:dp{dp}"

        return cls(
            NgramHistoryConfig(
                path=path,
                namespace=namespace,
                writer_id=writer_id,
                push_max_records=server_args.speculative_ngram_l2_push_max_records,
                pull_interval_ms=server_args.speculative_ngram_l2_pull_interval_ms,
                import_max_records_per_step=server_args.speculative_ngram_l2_import_max_records_per_step,
                import_max_tokens_per_step=server_args.speculative_ngram_l2_import_max_tokens_per_step,
                start_from_beginning=server_args.speculative_ngram_l2_start_from_beginning,
                max_trie_depth=server_args.speculative_ngram_max_trie_depth,
                mmap_capacity=server_args.speculative_ngram_l2_mmap_capacity,
            ),
            remote_epoch_sink=remote_epoch_sink,
        )

    def enqueue_push(self, flat_tokens: np.ndarray, offsets: np.ndarray) -> None:
        """Queue the exact CSR storage already used by the local Trie insert."""
        MmapNgramHistoryBackend._validate_csr(flat_tokens, offsets)
        window_count = int(offsets.size) - 1
        if window_count == 0:
            return
        try:
            self._push_queue.put_nowait((flat_tokens, offsets))
            self._push_event.set()
        except queue.Full:
            self.dropped_push_records += window_count

    def pause_remote_sink(self) -> None:
        self._sink_lock.acquire()

    def resume_remote_sink(self) -> None:
        self._sink_lock.release()

    def close(self) -> None:
        if self._stop_event.is_set():
            return
        self._stop_event.set()
        self._push_event.set()
        for thread in (self._push_thread, self._pull_thread):
            if thread.is_alive():
                thread.join(timeout=2.0)
        if not self._push_thread.is_alive():
            while not self._push_queue.empty():
                self._flush_push_queue()
        self.backend.close()

    def _push_loop(self) -> None:
        while True:
            self._push_event.wait()
            self.push_loop_wakeups += 1
            self._push_event.clear()
            if self._push_queue.empty():
                self.push_loop_empty_wakeups += 1
            while not self._push_queue.empty():
                self._flush_push_queue()
            if self._stop_event.is_set():
                return

    def _flush_push_queue(self) -> None:
        self.push_loop_flush_calls += 1
        batches: list[tuple[np.ndarray, np.ndarray]] = []
        queued_windows = 0
        while queued_windows < self.config.push_max_records or not batches:
            try:
                batch = self._push_queue.get_nowait()
            except queue.Empty:
                break
            batches.append(batch)
            queued_windows += int(batch[1].size) - 1
        for index, (flat_tokens, offsets) in enumerate(batches):
            window_count = int(offsets.size) - 1
            try:
                published = int(
                    self.backend.append_windows_csr(flat_tokens, offsets)
                )
                if published != window_count:
                    raise RuntimeError(
                        "NGRAM ring published an incomplete CSR batch: "
                        f"{published} != {window_count}"
                    )
            except Exception:
                self.dropped_push_records += sum(
                    int(remaining_offsets.size) - 1
                    for _, remaining_offsets in batches[index:]
                )
                logger.warning("Failed to publish NGRAM CSR batch.", exc_info=True)
                return
            self.pushed_records += published
            self.published_batches += 1
            self.published_tokens += int(flat_tokens.size)

    def _pull_loop(self) -> None:
        while not self._stop_event.is_set():
            self._pull_once()

    def _pull_once(self) -> None:
        try:
            flat, offsets, cursor = self.backend.read_windows_wait(
                self._pull_records,
                self.config.import_max_tokens_per_step,
                timeout_ms=self._wait_timeout_ms,
                scan_interval_ms=self._wait_scan_interval_ms,
            )
            self._cursor = cursor
        except Exception:
            logger.warning("Failed to receive NGRAM mmap history.", exc_info=True)
            self._stop_event.wait(self._wait_scan_interval_ms / 1000.0)
            return

        record_count = int(offsets.size) - 1
        if record_count == 0:
            return
        self.pulled_records += record_count
        with self._sink_lock:
            try:
                assert self._remote_epoch_sink is not None
                self._remote_epoch_sink(flat, offsets)
                self.imported_records += record_count
            except Exception:
                self.dropped_import_records += record_count
                logger.warning("Failed to stage remote NGRAM epoch.", exc_info=True)

        now = time.time()
        if now - self._last_import_log_time >= 5.0:
            self._last_import_log_time = now
            logger.info(
                "Imported %d remote NGRAM windows "
                "(total=%d, pulled=%d, dropped=%d, backend=%s)",
                record_count,
                self.imported_records,
                self.pulled_records,
                self.dropped_import_records,
                self.backend.stats(),
            )


__all__ = ["NgramHistoryConfig", "NgramHistoryReplicator"]
