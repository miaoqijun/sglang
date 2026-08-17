"""Binary protocol shared by NGRAM_SERVICE clients and the standalone server."""

from __future__ import annotations

import socket
import struct
from collections.abc import Iterable


MAGIC = b"NGRS"
PROTOCOL_VERSION = 1
MAX_FRAME_BYTES = 64 * 1024 * 1024

FRAME_HEADER = struct.Struct("<4sBBI")
BATCH_HEADER = struct.Struct("<II")
BATCH_PUT_HEADER = struct.Struct("<IIB7x")
MATCH_RESPONSE_HEADER = struct.Struct("<II")
ERASE_HEADER = struct.Struct("<I4x")
COUNT_RESPONSE = struct.Struct("<Q")

OP_BATCH_GET = 1
OP_BATCH_PUT = 2
OP_ERASE_MATCH_STATE = 3
OP_RESPONSE_BIT = 0x80
OP_ERROR = 0xFF


class ProtocolError(RuntimeError):
    pass


def recv_exact(sock: socket.socket, size: int) -> bytearray | None:
    data = bytearray(size)
    view = memoryview(data)
    received = 0
    while received < size:
        count = sock.recv_into(view[received:])
        if count == 0:
            if received == 0:
                return None
            raise EOFError("connection closed in the middle of a frame")
        received += count
    return data


def recv_frame(sock: socket.socket) -> tuple[int, bytearray] | None:
    raw_header = recv_exact(sock, FRAME_HEADER.size)
    if raw_header is None:
        return None
    magic, version, opcode, size = FRAME_HEADER.unpack(raw_header)
    if magic != MAGIC:
        raise ProtocolError("invalid NGRAM service frame magic")
    if version != PROTOCOL_VERSION:
        raise ProtocolError(
            f"unsupported NGRAM service protocol version {version}; "
            f"expected {PROTOCOL_VERSION}"
        )
    if size > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame is too large: {size}")
    payload = recv_exact(sock, size)
    if payload is None:
        raise EOFError("connection closed before frame payload")
    return opcode, payload


def send_frame(
    sock: socket.socket,
    opcode: int,
    buffers: Iterable[bytes | bytearray | memoryview] = (),
) -> None:
    views = [memoryview(buffer).cast("B") for buffer in buffers]
    size = sum(len(view) for view in views)
    if size > MAX_FRAME_BYTES:
        raise ProtocolError(f"frame is too large: {size}")
    sock.sendall(FRAME_HEADER.pack(MAGIC, PROTOCOL_VERSION, opcode, size))
    for view in views:
        if view:
            sock.sendall(view)


def raise_if_error(opcode: int, payload: bytearray) -> None:
    if opcode == OP_ERROR:
        raise ProtocolError(bytes(payload).decode("utf-8", errors="replace"))
