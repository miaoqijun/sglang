"""Standalone binary service around SGLang's existing C++ NGRAM corpus."""

from __future__ import annotations

import argparse
import logging
import signal
import socket
import socketserver
import threading

import torch

from ngram_service.protocol import (
    BATCH_HEADER,
    BATCH_PUT_HEADER,
    COUNT_RESPONSE,
    ERASE_HEADER,
    MATCH_RESPONSE_HEADER,
    OP_BATCH_GET,
    OP_BATCH_PUT,
    OP_ERASE_MATCH_STATE,
    OP_ERROR,
    OP_RESPONSE_BIT,
    ProtocolError,
    recv_frame,
    send_frame,
)
from sglang.srt.speculative.cpp_ngram.ngram_corpus import NgramCorpus


logger = logging.getLogger(__name__)
_MAX_LOCAL_STATE_ID = (1 << 32) - 1
_MAX_SESSION_ID = (1 << 31) - 1


def _tensor_from_buffer(
    payload: bytearray,
    dtype: torch.dtype,
    count: int,
    offset: int,
) -> torch.Tensor:
    return torch.frombuffer(payload, dtype=dtype, count=count, offset=offset)


def _validate_csr(offsets: torch.Tensor, token_count: int) -> None:
    if offsets.numel() < 2:
        raise ProtocolError("CSR offsets must contain at least two entries")
    if int(offsets[0]) != 0 or int(offsets[-1]) != token_count:
        raise ProtocolError("CSR offsets do not span the token payload")
    if bool(torch.any(offsets[1:] < offsets[:-1])):
        raise ProtocolError("CSR offsets must be nondecreasing")
    if bool(torch.any(offsets < 0)) or bool(torch.any(offsets > token_count)):
        raise ProtocolError("CSR offset is outside the token payload")


class NgramServiceServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 1024

    def __init__(
        self,
        server_address: tuple[str, int],
        corpus: NgramCorpus,
        draft_token_num: int,
    ) -> None:
        super().__init__(server_address, NgramServiceHandler)
        self.corpus = corpus
        self.draft_token_num = draft_token_num
        self._session_lock = threading.Lock()
        self._next_session_id = 1

    def allocate_session_id(self) -> int:
        with self._session_lock:
            if self._next_session_id > _MAX_SESSION_ID:
                raise RuntimeError("NGRAM service session id space is exhausted")
            session_id = self._next_session_id
            self._next_session_id += 1
            return session_id


class NgramServiceHandler(socketserver.BaseRequestHandler):
    server: NgramServiceServer

    def setup(self) -> None:
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._session_id = self.server.allocate_session_id()
        self._known_state_ids: set[int] = set()

    def finish(self) -> None:
        if not self._known_state_ids:
            return
        try:
            state_ids = torch.tensor(
                sorted(self._known_state_ids), dtype=torch.int64
            )
            self.server.corpus.erase_match_state_ids(state_ids)
        except Exception:
            logger.exception(
                "failed to erase match state for disconnected session %d",
                self._session_id,
            )

    def handle(self) -> None:
        while True:
            try:
                frame = recv_frame(self.request)
                if frame is None:
                    return
                opcode, payload = frame
                self._dispatch(opcode, payload)
            except (EOFError, ConnectionError, OSError):
                return
            except (ProtocolError, ValueError, RuntimeError) as exc:
                send_frame(self.request, OP_ERROR, (str(exc).encode(),))
            except Exception as exc:
                logger.exception("NGRAM service request failed")
                send_frame(self.request, OP_ERROR, (str(exc).encode(),))

    def _dispatch(self, opcode: int, payload: bytearray) -> None:
        if opcode == OP_BATCH_GET:
            self._batch_get(payload)
        elif opcode == OP_BATCH_PUT:
            self._batch_put(payload)
        elif opcode == OP_ERASE_MATCH_STATE:
            self._erase_match_state(payload)
        else:
            raise ProtocolError(f"unknown opcode {opcode}")

    @staticmethod
    def _parse_csr(
        payload: bytearray,
        *,
        header_size: int,
        batch_size: int,
        token_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        offsets_offset = header_size
        tokens_offset = offsets_offset + (batch_size + 1) * 8
        expected = tokens_offset + token_count * 4
        if len(payload) != expected:
            raise ProtocolError(f"invalid CSR payload size {len(payload)}/{expected}")
        offsets = _tensor_from_buffer(
            payload, torch.int64, batch_size + 1, offsets_offset
        )
        tokens = _tensor_from_buffer(
            payload, torch.int32, token_count, tokens_offset
        )
        _validate_csr(offsets, token_count)
        return tokens, offsets

    def _batch_put(self, payload: bytearray) -> None:
        if len(payload) < BATCH_PUT_HEADER.size:
            raise ProtocolError("truncated batch_put header")
        batch_size, token_count, wait_for_visibility = BATCH_PUT_HEADER.unpack_from(
            payload
        )
        if batch_size == 0:
            raise ProtocolError("batch_put batch size must be positive")
        if wait_for_visibility not in (0, 1):
            raise ProtocolError("wait_for_visibility must be 0 or 1")
        tokens, offsets = self._parse_csr(
            payload,
            header_size=BATCH_PUT_HEADER.size,
            batch_size=batch_size,
            token_count=token_count,
        )
        self.server.corpus.batch_put_csr(
            tokens,
            offsets,
            wait_for_visibility=bool(wait_for_visibility),
        )
        send_frame(
            self.request,
            OP_BATCH_PUT | OP_RESPONSE_BIT,
            (COUNT_RESPONSE.pack(batch_size),),
        )

    def _batch_get(self, payload: bytearray) -> None:
        if len(payload) < BATCH_HEADER.size:
            raise ProtocolError("truncated batch_get header")
        batch_size, token_count = BATCH_HEADER.unpack_from(payload)
        if batch_size == 0:
            raise ProtocolError("batch_get batch size must be positive")

        state_offset = BATCH_HEADER.size
        lens_offset = state_offset + batch_size * 8
        offsets_offset = lens_offset + batch_size * 8
        tokens_offset = offsets_offset + (batch_size + 1) * 8
        expected = tokens_offset + token_count * 4
        if len(payload) != expected:
            raise ProtocolError(
                f"invalid batch_get payload size {len(payload)}/{expected}"
            )

        local_state_ids = _tensor_from_buffer(
            payload, torch.int64, batch_size, state_offset
        )
        if bool(torch.any(local_state_ids < 0)) or bool(
            torch.any(local_state_ids > _MAX_LOCAL_STATE_ID)
        ):
            raise ProtocolError("local state id must fit in an unsigned 32-bit value")
        if len(set(local_state_ids.tolist())) != batch_size:
            raise ProtocolError("state ids in one batch_get call must be unique")
        total_lens = _tensor_from_buffer(
            payload, torch.int64, batch_size, lens_offset
        )
        if bool(torch.any(total_lens <= 0)):
            raise ProtocolError("total_lens must be positive")
        offsets = _tensor_from_buffer(
            payload, torch.int64, batch_size + 1, offsets_offset
        )
        tokens = _tensor_from_buffer(
            payload, torch.int32, token_count, tokens_offset
        )
        _validate_csr(offsets, token_count)

        state_ids = local_state_ids.clone()
        state_ids += self._session_id << 32
        self._known_state_ids.update(state_ids.tolist())

        d = self.server.draft_token_num
        output_tokens = torch.zeros(batch_size * d, dtype=torch.int32)
        output_mask = torch.zeros(batch_size * d * d, dtype=torch.uint8)
        self.server.corpus.batch_get_csr_into(
            state_ids,
            tokens,
            offsets,
            total_lens,
            output_tokens,
            output_mask,
        )
        send_frame(
            self.request,
            OP_BATCH_GET | OP_RESPONSE_BIT,
            (
                MATCH_RESPONSE_HEADER.pack(batch_size, d),
                output_tokens.numpy(),
                output_mask.numpy(),
            ),
        )

    def _erase_match_state(self, payload: bytearray) -> None:
        if len(payload) < ERASE_HEADER.size:
            raise ProtocolError("truncated erase_match_state header")
        (count,) = ERASE_HEADER.unpack_from(payload)
        expected = ERASE_HEADER.size + count * 8
        if len(payload) != expected:
            raise ProtocolError(
                f"invalid erase_match_state payload size {len(payload)}/{expected}"
            )
        if count == 0:
            send_frame(
                self.request,
                OP_ERASE_MATCH_STATE | OP_RESPONSE_BIT,
                (COUNT_RESPONSE.pack(0),),
            )
            return
        local_state_ids = _tensor_from_buffer(
            payload, torch.int64, count, ERASE_HEADER.size
        )
        if bool(torch.any(local_state_ids < 0)) or bool(
            torch.any(local_state_ids > _MAX_LOCAL_STATE_ID)
        ):
            raise ProtocolError("local state id must fit in an unsigned 32-bit value")
        state_ids = local_state_ids.clone()
        state_ids += self._session_id << 32
        self.server.corpus.erase_match_state_ids(state_ids)
        self._known_state_ids.difference_update(state_ids.tolist())
        send_frame(
            self.request,
            OP_ERASE_MATCH_STATE | OP_RESPONSE_BIT,
            (COUNT_RESPONSE.pack(count),),
        )


def create_server(
    host: str,
    port: int,
    *,
    capacity: int,
    max_trie_depth: int,
    min_bfs_breadth: int,
    max_bfs_breadth: int,
    draft_token_num: int,
    match_type: str,
) -> NgramServiceServer:
    corpus = NgramCorpus(
        capacity=capacity,
        max_trie_depth=max_trie_depth,
        min_bfs_breadth=min_bfs_breadth,
        max_bfs_breadth=max_bfs_breadth,
        draft_token_num=draft_token_num,
        match_type=match_type,
    )
    return NgramServiceServer((host, port), corpus, draft_token_num)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31291)
    parser.add_argument("--capacity", type=int, default=10_000_000)
    parser.add_argument("--max-trie-depth", type=int, default=18)
    parser.add_argument("--min-bfs-breadth", type=int, default=1)
    parser.add_argument("--max-bfs-breadth", type=int, default=10)
    parser.add_argument("--draft-token-num", type=int, default=12)
    parser.add_argument("--match-type", choices=["BFS", "PROB"], default="BFS")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    server = create_server(
        args.host,
        args.port,
        capacity=args.capacity,
        max_trie_depth=args.max_trie_depth,
        min_bfs_breadth=args.min_bfs_breadth,
        max_bfs_breadth=args.max_bfs_breadth,
        draft_token_num=args.draft_token_num,
        match_type=args.match_type,
    )

    def stop(_signum, _frame) -> None:
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    logger.info(
        "NGRAM service listening on %s:%d with one shared Trie",
        args.host,
        args.port,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
