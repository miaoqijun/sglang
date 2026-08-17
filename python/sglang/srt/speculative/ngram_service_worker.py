"""NGRAM worker whose corpus lives in a standalone service."""

from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_service_client import NgramServiceClient
from sglang.srt.speculative.ngram_worker import NGRAMWorker


class NGRAMServiceWorker(NGRAMWorker):
    def _create_ngram_corpus(self, server_args: ServerArgs) -> NgramServiceClient:
        return NgramServiceClient(
            address=server_args.speculative_ngram_service_address,
            draft_token_num=server_args.speculative_num_draft_tokens,
            timeout_s=server_args.speculative_ngram_service_timeout_s,
        )
