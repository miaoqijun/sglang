"""Benchmark variant for the template-aware chunk cache.

Compared to bench_sglang.py, this variant:
  1. Registers the 5 agent prompt templates via /register_prompt_template once.
  2. Sends each request as {template_id, template_vars} via /generate, so the
     server's TemplateAwareChunkCache can carve KV per template segment.

Usage:
    python -m sglang.launch_server \
        --model-path <model> --port 30000 --enable-template-chunk-cache
    python benchmark/generative_agents/bench_sglang_template.py \
        --base-url http://127.0.0.1:30000 --num-events 50
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, Dict

import requests
from templates import (
    ACTION_LOCATION_OBJECT_TEMPLATE,
    ACTION_LOCATION_SECTOR_TEMPLATE,
    ALL_TEMPLATES,
    GENERATE_EVENT_TRIPLE_TEMPLATE,
    GENERATE_PRONUNCIATIO_TEMPLATE,
    POIGNANCY_EVENT_TEMPLATE,
)

from sglang.utils import read_jsonl

FUNC_TO_TEMPLATE: Dict[str, Dict[str, Any]] = {
    "poignancy_event": POIGNANCY_EVENT_TEMPLATE,
    "generate_event_triple": GENERATE_EVENT_TRIPLE_TEMPLATE,
    "generate_pronunciatio": GENERATE_PRONUNCIATIO_TEMPLATE,
    "action_location_sector": ACTION_LOCATION_SECTOR_TEMPLATE,
    "action_location_object": ACTION_LOCATION_OBJECT_TEMPLATE,
}


def register_templates(base_url: str) -> None:
    for tpl in ALL_TEMPLATES:
        payload = {"template_id": tpl["template_id"], "segments": tpl["segments"]}
        r = requests.post(f"{base_url}/register_prompt_template", json=payload)
        r.raise_for_status()
        body = r.json()
        if not body.get("success"):
            raise RuntimeError(
                f"Failed to register template {tpl['template_id']!r}: {body}"
            )
        print(f"registered template: {tpl['template_id']}")


def submit_one(base_url: str, func_name: str, template_vars: Dict[str, Any]) -> str:
    tpl = FUNC_TO_TEMPLATE[func_name]
    payload = {
        "template_id": tpl["template_id"],
        "template_vars": template_vars,
        "sampling_params": {
            "max_new_tokens": tpl["max_tokens"],
            "stop": tpl["stop"],
            "temperature": 0.0,
        },
    }
    r = requests.post(f"{base_url}/generate", json=payload)
    r.raise_for_status()
    body = r.json()
    return body.get("text", "")


def main(args: argparse.Namespace) -> None:
    register_templates(args.base_url)

    lines = list(read_jsonl(args.data_path))[: args.num_events]

    # Warm-up: one full pass per template id, so subsequent calls hit fixed segments.
    print("warming up cache by running each agent function once ...")
    seen_funcs = set()
    for line in lines:
        for func_name, vars_dict in line.items():
            if func_name in seen_funcs:
                continue
            submit_one(args.base_url, func_name, vars_dict)
            seen_funcs.add(func_name)
        if len(seen_funcs) == len(FUNC_TO_TEMPLATE):
            break

    states = []
    tic = time.perf_counter()
    for line in lines:
        # one key per line, matching bench_sglang.py
        for func_name, vars_dict in line.items():
            states.append(submit_one(args.base_url, func_name, vars_dict))
    latency = time.perf_counter() - tic

    print(f"Latency: {latency:.3f}")

    with open(args.result_file, "a") as fout:
        value = {
            "task": "Generative Agents (template cache)",
            "backend": "sglang-template",
            "num_gpus": 1,
            "latency": round(latency, 3),
            "num_requests": len(lines),
            "other": {"num_events": args.num_events},
        }
        fout.write(json.dumps(value) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, default="agent_calls.jsonl")
    parser.add_argument("--num-events", type=int, default=10)
    parser.add_argument("--base-url", type=str, default="http://127.0.0.1:30000")
    parser.add_argument("--result-file", type=str, default="result_template.jsonl")
    args = parser.parse_args()
    main(args)
