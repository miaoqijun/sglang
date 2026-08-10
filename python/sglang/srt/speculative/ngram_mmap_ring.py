from __future__ import annotations

import hashlib
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_DISCOVERY_INTERVAL_S = 0.2
_extension = None
_extension_lock = threading.Lock()


def _load_extension():
    global _extension
    if _extension is not None:
        return _extension
    with _extension_lock:
        if _extension is not None:
            return _extension
        from torch.utils.cpp_extension import load

        source = Path(__file__).with_name("cpp_ngram") / "ngram_mmap_ring.cpp"
        _extension = load(
            name="sglang_ngram_mmap_ring",
            sources=[str(source)],
            extra_cflags=["-O3", "-std=c++17"],
            with_cuda=False,
            verbose=False,
        )
        return _extension


def _stable_u64(value: str) -> int:
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "little")


class MmapNgramHistoryBackend:
    """One fixed-window mmap ring per writer.

    Python only manages peer discovery and lifecycle. Window payloads remain in
    CSR buffers and are copied to/from ring slots by native code without the GIL.
    """

    def __init__(
        self,
        root_path: str,
        namespace: str,
        writer_id: str,
        *,
        capacity: int,
        max_trie_depth: int,
        start_from_beginning: bool,
        max_records: int,
        max_tokens: int,
    ) -> None:
        if not 1 < max_trie_depth <= 18:
            raise ValueError("mmap NGRAM history requires max depth in [2, 18]")
        if capacity < 4:
            raise ValueError("mmap NGRAM history capacity must be at least 4")
        if max_records <= 0:
            raise ValueError("mmap NGRAM history requires a positive record budget")
        if max_tokens < max_trie_depth:
            raise ValueError("mmap token budget must cover max_trie_depth")

        self.namespace = namespace
        self.writer_id = writer_id
        self.max_trie_depth = int(max_trie_depth)
        self.start_from_beginning = bool(start_from_beginning)
        self.capacity = int(capacity)
        self._namespace_hash = _stable_u64(namespace)

        namespace_key = hashlib.sha256(namespace.encode()).hexdigest()[:16]
        writer_key = hashlib.sha256(writer_id.encode()).hexdigest()[:16]
        self.directory = Path(root_path).expanduser().resolve() / namespace_key
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = self.directory / f"{writer_key}.ring"

        self._extension = _load_extension()
        self._writer = self._extension.RingWriter(
            str(self.path),
            self.capacity,
            self._namespace_hash,
            secrets.randbits(63) + 1,
        )
        self._readers: dict[Path, Any] = {}
        self._reader_rotation = 0
        self._next_discovery_time = 0.0
        self._cursor = 0

        self._flat = np.empty(max_tokens, dtype=np.int32)
        self._offsets = np.empty(max_records + 1, dtype=np.int64)
        self._offsets[0] = 0

        self.published_batches = 0
        self.published_windows = 0
        self.published_tokens = 0
        self.polled_windows = 0
        self.polled_tokens = 0
        self.poll_calls = 0
        self.nonempty_polls = 0
        self.wait_calls = 0
        self.wait_scans = 0
        self.wait_wakeups = 0
        self.wait_timeouts = 0
        self.reset_events = 0
        self.overrun_events = 0

    def end_offset(self) -> int:
        return self._cursor

    def append_windows_csr(
        self, flat_tokens: np.ndarray, offsets: np.ndarray
    ) -> int:
        self._validate_csr(flat_tokens, offsets)
        window_count = int(offsets.size) - 1
        if window_count == 0:
            return 0
        count = int(self._writer.publish_windows_csr(flat_tokens, offsets))
        self.published_batches += 1
        self.published_windows += count
        self.published_tokens += int(flat_tokens.size)
        return count

    def read_windows_wait(
        self,
        max_records: int,
        max_tokens: int,
        *,
        timeout_ms: int,
        scan_interval_ms: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """Return one backend-owned CSR view, waiting natively when empty."""
        if timeout_ms <= 0 or scan_interval_ms <= 0:
            raise ValueError("wait timeout and scan interval must be positive")

        self.poll_calls += 1
        rotation_before_probe = self._reader_rotation
        flat, offsets, cursor = self._read_once(max_records, max_tokens)
        if offsets.size > 1:
            return flat, offsets, cursor

        # An empty probe must not consume the round-robin fairness turn.
        self._reader_rotation = rotation_before_probe
        self._discover_readers()
        readers = list(self._readers.values())
        effective_timeout_ms = (
            int(timeout_ms)
            if readers
            else min(int(timeout_ms), int(scan_interval_ms))
        )
        self.wait_calls += 1
        ready, scans = self._extension.wait_readers(
            readers, effective_timeout_ms, int(scan_interval_ms)
        )
        self.wait_scans += int(scans)
        if ready:
            self.wait_wakeups += 1
        else:
            self.wait_timeouts += 1
        if not readers:
            self._next_discovery_time = 0.0
        return self._read_once(max_records, max_tokens)

    def _read_once(
        self, max_records: int, max_tokens: int
    ) -> tuple[np.ndarray, np.ndarray, int]:
        if max_records <= 0:
            self._offsets[0] = 0
            return self._flat[:0], self._offsets[:1], self._cursor
        if max_tokens < self.max_trie_depth:
            raise ValueError("token budget must cover max_trie_depth")
        if max_records > self._offsets.size - 1:
            raise ValueError("record budget exceeds the preallocated buffer")
        if max_tokens > self._flat.size:
            raise ValueError("token budget exceeds the preallocated buffer")

        self._discover_readers()
        self._offsets[0] = 0
        produced = 0
        token_count = 0
        reader_items = list(self._readers.items())
        if reader_items:
            start = self._reader_rotation % len(reader_items)
            reader_items = reader_items[start:] + reader_items[:start]
            self._reader_rotation = (start + 1) % len(reader_items)

        for path, reader in reader_items:
            if produced >= max_records or token_count >= max_tokens:
                break
            try:
                (
                    peer_records,
                    peer_tokens,
                    _epoch,
                    reset,
                    gaps,
                    _peer_cursor,
                    _published,
                ) = reader.poll_into(
                    self._flat,
                    self._offsets,
                    produced,
                    token_count,
                    max_records,
                    max_tokens,
                )
            except Exception:
                logger.warning(
                    "Failed to poll NGRAM mmap ring %s", path, exc_info=True
                )
                self._drop_reader(path, reader)
                continue

            if reset:
                self.reset_events += 1
                self.overrun_events += int(gaps)
            produced += int(peer_records)
            token_count += int(peer_tokens)

        self._cursor += produced
        self.polled_windows += produced
        self.polled_tokens += token_count
        if produced:
            self.nonempty_polls += 1
        return (
            self._flat[:token_count],
            self._offsets[: produced + 1],
            self._cursor,
        )

    def stats(self) -> dict[str, int]:
        return {
            "published_batches": self.published_batches,
            "published_windows": self.published_windows,
            "published_tokens": self.published_tokens,
            "polled_windows": self.polled_windows,
            "polled_tokens": self.polled_tokens,
            "reset_events": self.reset_events,
            "overrun_events": self.overrun_events,
            "peer_readers": len(self._readers),
            "poll_calls": self.poll_calls,
            "nonempty_polls": self.nonempty_polls,
            "wait_calls": self.wait_calls,
            "wait_scans": self.wait_scans,
            "wait_wakeups": self.wait_wakeups,
            "wait_timeouts": self.wait_timeouts,
        }

    def close(self) -> None:
        for reader in self._readers.values():
            reader.close()
        self._readers.clear()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        try:
            self.path.unlink(missing_ok=True)
            self.directory.rmdir()
        except OSError:
            pass

    @staticmethod
    def _validate_csr(flat_tokens: np.ndarray, offsets: np.ndarray) -> None:
        if (
            flat_tokens.dtype != np.int32
            or flat_tokens.ndim != 1
            or not flat_tokens.flags.c_contiguous
        ):
            raise ValueError("window tokens must be contiguous int32")
        if (
            offsets.dtype != np.int64
            or offsets.ndim != 1
            or not offsets.flags.c_contiguous
            or offsets.size == 0
        ):
            raise ValueError("window offsets must be contiguous int64")
        if int(offsets[0]) != 0 or int(offsets[-1]) != flat_tokens.size:
            raise ValueError("window offsets do not cover the token buffer")

    def _drop_reader(self, path: Path, reader: Any) -> None:
        try:
            reader.close()
        finally:
            self._readers.pop(path, None)

    def _discover_readers(self) -> None:
        now = time.monotonic()
        if now < self._next_discovery_time:
            return
        self._next_discovery_time = now + _DISCOVERY_INTERVAL_S

        present = set(self.directory.glob("*.ring"))
        for path, reader in list(self._readers.items()):
            if path not in present or not reader.writer_alive():
                self._drop_reader(path, reader)
        for path in present:
            if path == self.path or path in self._readers:
                continue
            try:
                self._readers[path] = self._extension.RingReader(
                    str(path), self._namespace_hash, self.start_from_beginning
                )
            except Exception:
                # A peer may be between ftruncate and publishing its ready flag.
                logger.debug("NGRAM mmap peer is not ready: %s", path, exc_info=True)


__all__ = ["MmapNgramHistoryBackend"]
