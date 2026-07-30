"""Replay local benchmark prompts against an OpenAI-compatible chat server.

This script is intentionally workload-oriented: it records the exact messages,
model output, token usage, latency, and benchmark metadata for each LLM turn.
It does not grade answer quality.

Examples:
    python SD_benchmark/run_benchmark.py \
        --benchmark mt_bench \
        --server-url http://127.0.0.1:1919/v1 \
        --model Qwen2.5-7B-Instruct-AWQ \
        --temperature 0 \
        --max-tokens 1024 \
        --concurrency 8

    python SD_benchmark/run_benchmark.py \
        --benchmark HumanEval \
        --limit 20
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_BENCHMARK_ROOT = SCRIPT_DIR
DEFAULT_OUTPUT_ROOT = SCRIPT_DIR / "outputs" / "benchmark_replay"


@dataclass(frozen=True)
class BenchmarkItem:
    benchmark: str
    question_id: str
    category: str
    turns: tuple[str, ...]
    source: dict[str, Any]


def normalize_benchmark_name(value: str) -> str:
    key = value.strip().lower().replace("-", "_")
    aliases = {
        "human_eval": "humaneval",
        "humaneval": "humaneval",
        "humaneval_v2": "humaneval",
        "mtbench": "mt_bench",
        "mt_bench": "mt_bench",
        "specbench": "spec_bench",
        "spec_bench": "spec_bench",
    }
    if key not in aliases:
        raise argparse.ArgumentTypeError(
            "benchmark must be one of: HumanEval, mt_bench, spec_bench"
        )
    return aliases[key]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def humaneval_messages(prompt: str, style: str) -> list[dict[str, str]]:
    if style == "raw_user":
        content = prompt
    elif style == "completion_instruction":
        content = (
            "Complete the following Python function. Return only the code needed "
            "to complete the function body.\n\n"
            f"```python\n{prompt}\n```"
        )
    else:
        raise ValueError(f"Unknown HumanEval style: {style}")
    return [{"role": "user", "content": content}]


def load_benchmark(
    benchmark: str,
    benchmark_root: Path,
    *,
    humaneval_style: str,
) -> list[BenchmarkItem]:
    if benchmark == "humaneval":
        path = benchmark_root / "HumanEval" / "human-eval-v2-20210705.jsonl"
        rows = read_jsonl(path)
        return [
            BenchmarkItem(
                benchmark="HumanEval",
                question_id=str(row.get("task_id", index)),
                category="coding",
                turns=(humaneval_messages(str(row.get("prompt", "")), humaneval_style)[0]["content"],),
                source=row,
            )
            for index, row in enumerate(rows)
        ]

    if benchmark == "mt_bench":
        path = benchmark_root / "mt_bench" / "question.jsonl"
        rows = read_jsonl(path)
        return [
            BenchmarkItem(
                benchmark="mt_bench",
                question_id=str(row.get("question_id", index)),
                category=str(row.get("category", "")),
                turns=tuple(str(turn) for turn in row.get("turns", [])),
                source=row,
            )
            for index, row in enumerate(rows)
        ]

    if benchmark == "spec_bench":
        path = benchmark_root / "spec_bench" / "question.jsonl"
        rows = read_jsonl(path)
        return [
            BenchmarkItem(
                benchmark="spec_bench",
                question_id=str(row.get("question_id", index)),
                category=str(row.get("category", "")),
                turns=tuple(str(turn) for turn in row.get("turns", [])),
                source=row,
            )
            for index, row in enumerate(rows)
        ]

    raise ValueError(f"Unsupported benchmark: {benchmark}")


def filter_items(
    items: list[BenchmarkItem],
    *,
    categories: set[str],
    limit: int | None,
) -> list[BenchmarkItem]:
    filtered = [
        item for item in items if not categories or item.category in categories
    ]
    if limit is not None:
        filtered = filtered[:limit]
    return filtered


def endpoint(server_url: str) -> str:
    base = server_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def call_chat_completion(
    *,
    server_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    seed: int | None,
    max_tokens: int,
    timeout: float,
    extra_body: dict[str, Any],
) -> tuple[dict[str, Any] | None, float, str | None]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    payload.update(extra_body)
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    request = urllib.request.Request(
        endpoint(server_url),
        data=data,
        headers=headers,
        method="POST",
    )
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        latency = time.perf_counter() - start
        return json.loads(raw), latency, None
    except urllib.error.HTTPError as exc:
        latency = time.perf_counter() - start
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return None, latency, f"HTTP {exc.code}: {body[:1000]}"
    except Exception as exc:
        latency = time.perf_counter() - start
        return None, latency, repr(exc)


def extract_output(response: dict[str, Any] | None) -> str:
    if not response:
        return ""
    choices = response.get("choices") or []
    if not choices:
        return ""
    first = choices[0]
    message = first.get("message") or {}
    if isinstance(message, dict) and message.get("content") is not None:
        return str(message.get("content") or "")
    if first.get("text") is not None:
        return str(first.get("text") or "")
    return ""


def usage_value(response: dict[str, Any] | None, key: str) -> int | None:
    if not response:
        return None
    usage = response.get("usage") or {}
    value = usage.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_item(
    item: BenchmarkItem,
    *,
    server_url: str,
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    seed: int | None,
    max_tokens: int,
    timeout: float,
    extra_body: dict[str, Any],
    stop_on_error: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    conversation: list[dict[str, str]] = []
    for turn_id, user_text in enumerate(item.turns):
        conversation.append({"role": "user", "content": user_text})
        request_messages = [dict(message) for message in conversation]
        response, latency_s, error = call_chat_completion(
            server_url=server_url,
            api_key=api_key,
            model=model,
            messages=request_messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=max_tokens,
            timeout=timeout,
            extra_body=extra_body,
        )
        output = extract_output(response)
        record = {
            "request_id": f"{item.benchmark}:{item.question_id}:turn{turn_id}",
            "benchmark": item.benchmark,
            "question_id": item.question_id,
            "turn_id": turn_id,
            "category": item.category,
            "messages": request_messages,
            "output": output,
            "prompt_tokens": usage_value(response, "prompt_tokens"),
            "completion_tokens": usage_value(response, "completion_tokens"),
            "total_tokens": usage_value(response, "total_tokens"),
            "latency_s": round(latency_s, 6),
            "error": error,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
        }
        records.append(record)
        if error and stop_on_error:
            break
        conversation.append({"role": "assistant", "content": output})
    return records


def write_jsonl_line(path: Path, lock: threading.Lock, row: dict[str, Any]) -> None:
    text = json.dumps(row, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [
        float(row["latency_s"]) for row in records if row.get("latency_s") is not None
    ]
    prompt_tokens = sum(int(row.get("prompt_tokens") or 0) for row in records)
    completion_tokens = sum(int(row.get("completion_tokens") or 0) for row in records)
    errors = sum(1 for row in records if row.get("error"))
    return {
        "turns": len(records),
        "errors": errors,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_avg_s": round(sum(latencies) / len(latencies), 6)
        if latencies
        else None,
        "latency_max_s": round(max(latencies), 6) if latencies else None,
    }


def parse_extra_body(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"Invalid --extra-body JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError("--extra-body must decode to a JSON object")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly one local benchmark against an OpenAI-compatible chat "
            "server and save per-turn JSONL traces."
        )
    )
    parser.add_argument(
        "--benchmark",
        required=True,
        type=normalize_benchmark_name,
        help="One benchmark per run: HumanEval, mt_bench, or spec_bench.",
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=DEFAULT_BENCHMARK_ROOT,
        help="Directory containing HumanEval/, mt_bench/, and spec_bench/.",
    )
    parser.add_argument(
        "--server-url",
        "--server_url",
        default="http://127.0.0.1:1919/v1",
    )
    parser.add_argument("--model", "--model-name", "--model_name", required=True)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0)
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Optional request seed. Use --seed -1 to omit the seed field.",
    )
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=1024)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--categories",
        default="",
        help="Optional comma-separated category filter.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--humaneval-style",
        choices=["completion_instruction", "raw_user"],
        default="completion_instruction",
    )
    parser.add_argument(
        "--extra-body",
        type=parse_extra_body,
        default={},
        help="Optional JSON object merged into every chat completion request.",
    )
    parser.add_argument(
        "--keep-going-after-error",
        action="store_true",
        help="Continue later turns of the same question after an error.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--print-every",
        type=int,
        default=10,
        help="Print progress every N completed questions.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")

    benchmark_root = args.benchmark_root.expanduser().resolve()
    items = load_benchmark(
        args.benchmark,
        benchmark_root,
        humaneval_style=args.humaneval_style,
    )
    categories = {
        item.strip() for item in args.categories.split(",") if item.strip()
    }
    items = filter_items(items, categories=categories, limit=args.limit)
    if not items:
        raise SystemExit("No benchmark items selected.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / args.benchmark / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "turn_traces.jsonl"
    summary_path = output_dir / "summary.json"
    if trace_path.exists():
        trace_path.unlink()

    print(f"benchmark: {args.benchmark}")
    print(f"items: {len(items)}")
    print(f"server_url: {args.server_url}")
    print(f"model: {args.model}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"seed: {None if args.seed < 0 else args.seed}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"concurrency: {args.concurrency}")
    print(f"trace_output: {trace_path}")

    lock = threading.Lock()
    all_records: list[dict[str, Any]] = []
    start = time.perf_counter()
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_to_item = {
            pool.submit(
                run_item,
                item,
                server_url=args.server_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=None if args.seed < 0 else args.seed,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body=args.extra_body,
                stop_on_error=not args.keep_going_after_error,
            ): item
            for item in items
        }
        for future in concurrent.futures.as_completed(future_to_item):
            item = future_to_item[future]
            try:
                records = future.result()
            except Exception as exc:
                records = [
                    {
                        "request_id": f"{item.benchmark}:{item.question_id}:error",
                        "benchmark": item.benchmark,
                        "question_id": item.question_id,
                        "turn_id": None,
                        "category": item.category,
                        "messages": [],
                        "output": "",
                        "prompt_tokens": None,
                        "completion_tokens": None,
                        "total_tokens": None,
                        "latency_s": None,
                        "error": repr(exc),
                        "model": args.model,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "seed": None if args.seed < 0 else args.seed,
                        "max_tokens": args.max_tokens,
                    }
                ]
            for record in records:
                write_jsonl_line(trace_path, lock, record)
            all_records.extend(records)
            completed += 1
            if args.print_every > 0 and (
                completed == len(items) or completed % args.print_every == 0
            ):
                elapsed = time.perf_counter() - start
                print(
                    f"[{completed}/{len(items)}] elapsed={elapsed:.1f}s "
                    f"turns={len(all_records)}"
                )
                sys.stdout.flush()

    wall_time_s = time.perf_counter() - start
    summary = {
        "benchmark": args.benchmark,
        "benchmark_root": str(benchmark_root),
        "server_url": args.server_url,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": None if args.seed < 0 else args.seed,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "items": len(items),
        "wall_time_s": round(wall_time_s, 6),
        "trace_output": str(trace_path),
        **summarize(all_records),
    }
    if summary["completion_tokens"]:
        summary["completion_tokens_s"] = round(
            summary["completion_tokens"] / wall_time_s, 6
        )
    if summary["total_tokens"]:
        summary["total_tokens_s"] = round(summary["total_tokens"] / wall_time_s, 6)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"summary_output: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
