"""SGLang gateway backend for the official SpecDec-Bench runner.

The official SGLang backend creates an in-process ``sgl.Engine``.  This
backend retains the runner's token-level interface but sends its already
tokenized ``input_ids`` to a running SGLang gateway's native ``/generate``
endpoint.  It is intended for multi-instance serving experiments.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .base import Model


class SGLANGRemoteModel(Model):
    """Run official SpecDec-Bench prompts through a remote SGLang gateway."""

    def __init__(
        self,
        model_dir: str,
        max_concurrent_requests: int,
        sampling_kwargs: dict[str, Any],
        **kwargs: Any,
    ) -> None:
        del model_dir
        server_url = kwargs.get("server_url")
        if not server_url:
            raise ValueError("SGLANG_REMOTE requires --server_url, e.g. http://127.0.0.1:1919")

        self.server_url = str(server_url).rstrip("/")
        # The official runner has no API-key argument because its built-in
        # SGLang backend is in-process.  The remote gateway can require one,
        # so obtain it explicitly from the batch launch environment.
        self.api_key = os.environ.get("SGLANG_API_KEY", "dummy")
        self.timeout_s = float(kwargs.get("server_timeout", 600))
        self.sampling_config = dict(sampling_kwargs)
        self.context_length = kwargs.get("context_length")
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(max_concurrent_requests)),
            thread_name_prefix="specbench-gateway",
        )

    async def run(
        self,
        prompt_ids: list[int],
        max_length: int,
        end_id: int,
        request_id: int,
        turn_id: int,
    ) -> dict[str, Any]:
        sampling_params = dict(self.sampling_config)
        sampling_params["max_new_tokens"] = max_length
        sampling_params["stop_token_ids"] = [end_id]
        payload = {
            "input_ids": prompt_ids,
            "sampling_params": sampling_params,
            "stream": True,
        }
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self._stream_generate, payload, end_id)

    def _stream_generate(self, payload: dict[str, Any], end_id: int) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            f"{self.server_url}/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )

        timings = [time.perf_counter()]
        cumulative_outputs: list[int] = []
        emitted_lengths: list[int] = []
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(f"SGLang gateway error: {chunk['error']}")
                    output_ids = chunk.get("output_ids")
                    if output_ids is None:
                        continue
                    cumulative_outputs = [int(token) for token in output_ids]
                    completion_tokens = chunk.get("meta_info", {}).get("completion_tokens")
                    if completion_tokens is None:
                        completion_tokens = len(cumulative_outputs)
                    emitted_lengths.append(int(completion_tokens))
                    timings.append(time.perf_counter())
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"SGLang gateway HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Could not reach SGLang gateway at {self.server_url}: {exc}") from exc

        if not emitted_lengths:
            raise RuntimeError("SGLang gateway returned no streaming output chunks")

        # Byte-for-byte mirror the official in-process SGLANGModel's segment
        # construction. In particular, do not merge equal cumulative lengths:
        # the official SpecBench acceptance histogram observes those segments.
        if cumulative_outputs and cumulative_outputs[-1] == end_id:
            cumulative_outputs.pop()
            emitted_lengths.pop()

        segments: list[list[int]] = []
        if emitted_lengths:
            if emitted_lengths[0] != 0:
                segments.append(cumulative_outputs[: emitted_lengths[0]])
            for start, end in zip(emitted_lengths, emitted_lengths[1:]):
                segments.append(cumulative_outputs[start:end])
            if len(cumulative_outputs) > emitted_lengths[-1]:
                segments.append(cumulative_outputs[emitted_lengths[-1] :])

        if len(timings) == 1:
            timings.append(time.perf_counter())

        return {
            "output_ids": [segments],
            "output_logits": None,
            "token_times": timings,
        }

    def get_serving_config(self) -> dict[str, Any]:
        return {
            "backend": "SGLANG_REMOTE",
            "server_url": self.server_url,
            "native_endpoint": "/generate",
            "context_length_hint": self.context_length,
        }

    def stop(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=True)
