"""Corpus-compatible client for the standalone NGRAM service."""

from __future__ import annotations

import os
import socket
from urllib.parse import urlsplit

import numpy as np

from sglang.srt.speculative.ngram_service_protocol import (
    BATCH_HEADER,
    BATCH_PUT_HEADER,
    COUNT_RESPONSE,
    ERASE_HEADER,
    MATCH_RESPONSE_HEADER,
    OP_BATCH_GET,
    OP_BATCH_PUT,
    OP_ERASE_MATCH_STATE,
    OP_RESPONSE_BIT,
    ProtocolError,
    raise_if_error,
    recv_frame,
    send_frame,
)


def _pack_csr(batch_tokens: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    if not batch_tokens or any(not tokens for tokens in batch_tokens):
        raise ValueError("batch_tokens must contain non-empty token sequences")
    lengths = np.fromiter(
        (len(tokens) for tokens in batch_tokens),
        dtype=np.int64,
        count=len(batch_tokens),
    )
    offsets = np.empty(len(batch_tokens) + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(lengths, out=offsets[1:])
    tokens = np.empty(int(offsets[-1]), dtype=np.int32)
    for index, row in enumerate(batch_tokens):
        tokens[offsets[index] : offsets[index + 1]] = row
    return tokens, offsets


def _parse_address(address: str) -> tuple[str, int]:
    parsed = urlsplit(address)
    if parsed.scheme != "tcp" or not parsed.hostname or parsed.port is None:
        raise ValueError(
            "--speculative-ngram-service-address must be a tcp://host:port URL"
        )
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("NGRAM service address cannot contain credentials or metadata")
    if parsed.path not in ("", "/"):
        raise ValueError("NGRAM service address cannot contain a path")
    return parsed.hostname, parsed.port


class NgramServiceClient:
    """Expose the local ``NgramCorpus`` hot-path API over a TCP connection.

    By default ``batch_put`` waits for visibility. Analysis runs can set
    ``SGLANG_NGRAM_SERVICE_WAIT_FOR_VISIBILITY=0`` to use asynchronous one-way
    enqueue semantics, where the next ``batch_get`` may observe stale corpus
    state and put failures are reported by a later acknowledged request.
    """

    def __init__(
        self,
        address: str,
        draft_token_num: int,
        timeout_s: float = 5.0,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("NGRAM service timeout must be positive")
        self._host, self._port = _parse_address(address)
        self._draft_token_num = draft_token_num
        self._timeout_s = timeout_s
        self._socket: socket.socket | None = None
        self._next_state_id = 0
        self._req_to_state: dict[str, int] = {}
        wait_value = os.environ.get(
            "SGLANG_NGRAM_SERVICE_WAIT_FOR_VISIBILITY", "1"
        ).strip().lower()
        if wait_value not in ("0", "1", "false", "true"):
            raise ValueError(
                "SGLANG_NGRAM_SERVICE_WAIT_FOR_VISIBILITY must be 0/1/false/true"
            )
        self._default_wait_for_visibility = wait_value in ("1", "true")
        self._connect()

    def _connect(self) -> None:
        sock = socket.create_connection(
            (self._host, self._port), timeout=self._timeout_s
        )
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._socket = sock

    def _request(self, opcode: int, buffers=()) -> bytearray:
        if self._socket is None:
            raise RuntimeError("NGRAM service client is closed")
        send_frame(self._socket, opcode, buffers)
        frame = recv_frame(self._socket)
        if frame is None:
            raise EOFError("NGRAM service closed the connection")
        response_opcode, payload = frame
        raise_if_error(response_opcode, payload)
        expected = opcode | OP_RESPONSE_BIT
        if response_opcode != expected:
            raise ProtocolError(
                f"unexpected response opcode {response_opcode}; expected {expected}"
            )
        return payload

    def _send_oneway(self, opcode: int, buffers=()) -> None:
        if self._socket is None:
            raise RuntimeError("NGRAM service client is closed")
        send_frame(self._socket, opcode, buffers)

    def _state_ids(self, req_ids: list[str]) -> np.ndarray:
        state_ids = np.empty(len(req_ids), dtype=np.int64)
        for index, req_id in enumerate(req_ids):
            state_id = self._req_to_state.get(req_id)
            if state_id is None:
                if self._next_state_id >= 1 << 32:
                    raise RuntimeError("NGRAM service local state id space is exhausted")
                state_id = self._next_state_id
                self._next_state_id += 1
                self._req_to_state[req_id] = state_id
            state_ids[index] = state_id
        return state_ids

    def batch_put(
        self,
        batch_tokens: list[list[int]],
        *,
        wait_for_visibility: bool | None = None,
    ) -> None:
        if wait_for_visibility is None:
            wait_for_visibility = self._default_wait_for_visibility
        tokens, offsets = _pack_csr(batch_tokens)
        buffers = (
            BATCH_PUT_HEADER.pack(
                len(batch_tokens), tokens.size, int(wait_for_visibility)
            ),
            offsets,
            tokens,
        )
        if wait_for_visibility:
            payload = self._request(OP_BATCH_PUT, buffers)
            if len(payload) != COUNT_RESPONSE.size:
                raise ProtocolError("invalid batch_put response")
            (accepted,) = COUNT_RESPONSE.unpack(payload)
            if accepted != len(batch_tokens):
                raise ProtocolError("batch_put response has an invalid accepted count")
        else:
            self._send_oneway(OP_BATCH_PUT, buffers)

    def synchronize(self) -> None:
        # Visibility is controlled by batch_put; no fourth RPC is used.
        return

    def batch_get(
        self,
        req_ids: list[str],
        batch_tokens: list[list[int]],
        total_lens: list[int],
    ) -> tuple[np.ndarray, np.ndarray]:
        if not (len(req_ids) == len(batch_tokens) == len(total_lens)):
            raise ValueError("batch_get inputs must have equal lengths")
        state_ids = self._state_ids(req_ids)
        total_lens_array = np.asarray(total_lens, dtype=np.int64)
        tokens, offsets = _pack_csr(batch_tokens)
        payload = self._request(
            OP_BATCH_GET,
            (
                BATCH_HEADER.pack(len(batch_tokens), tokens.size),
                state_ids,
                total_lens_array,
                offsets,
                tokens,
            ),
        )
        if len(payload) < MATCH_RESPONSE_HEADER.size:
            raise ProtocolError("truncated batch_get response")
        batch_size, draft_token_num = MATCH_RESPONSE_HEADER.unpack_from(payload)
        if batch_size != len(batch_tokens):
            raise ProtocolError("batch_get response has an invalid batch size")
        if draft_token_num != self._draft_token_num:
            raise ProtocolError(
                "NGRAM service draft-token configuration mismatch: "
                f"service={draft_token_num}, client={self._draft_token_num}"
            )
        token_count = batch_size * draft_token_num
        mask_count = token_count * draft_token_num
        token_offset = MATCH_RESPONSE_HEADER.size
        mask_offset = token_offset + token_count * np.dtype(np.int32).itemsize
        expected = mask_offset + mask_count
        if len(payload) != expected:
            raise ProtocolError(
                f"invalid batch_get response size {len(payload)}/{expected}"
            )
        draft_tokens = np.frombuffer(
            payload, dtype=np.int32, count=token_count, offset=token_offset
        ).astype(np.int64)
        tree_masks = np.frombuffer(
            payload, dtype=np.uint8, count=mask_count, offset=mask_offset
        ).astype(np.int64)
        return draft_tokens, tree_masks

    def erase_match_state(self, req_ids: list[str]) -> None:
        state_ids = []
        for req_id in req_ids:
            state_id = self._req_to_state.pop(req_id, None)
            if state_id is not None:
                state_ids.append(state_id)
        state_ids_array = np.asarray(state_ids, dtype=np.int64)
        payload = self._request(
            OP_ERASE_MATCH_STATE,
            (ERASE_HEADER.pack(len(state_ids)), state_ids_array),
        )
        if len(payload) != COUNT_RESPONSE.size:
            raise ProtocolError("invalid erase_match_state response")
        (erased,) = COUNT_RESPONSE.unpack(payload)
        if erased != len(state_ids):
            raise ProtocolError("erase_match_state response has an invalid count")

    def reset(self) -> None:
        # Reconnecting gives the client a fresh service-side state namespace;
        # closing the old connection erases its cursors but preserves the Trie.
        self.close()
        self._req_to_state.clear()
        self._next_state_id = 0
        self._connect()

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def load_external_corpus_named(self, *_args, **_kwargs):
        raise RuntimeError("External corpora are not supported by NGRAM_SERVICE")

    def commit_external_corpus_load(self, *_args, **_kwargs) -> None:
        raise RuntimeError("External corpora are not supported by NGRAM_SERVICE")

    def remove_external_corpus(self, *_args, **_kwargs) -> None:
        raise RuntimeError("External corpora are not supported by NGRAM_SERVICE")

    def list_external_corpora(self) -> dict[str, int]:
        raise RuntimeError("External corpora are not supported by NGRAM_SERVICE")

    def __enter__(self) -> NgramServiceClient:
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()
