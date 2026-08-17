"""Standalone service frontend for SGLang's existing NGRAM corpus."""

from ngram_service.server import NgramServiceServer, create_server

__all__ = ["NgramServiceServer", "create_server"]
