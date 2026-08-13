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
import copy
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


@dataclass(frozen=True)
class TeacherForcingTurn:
    """A recorded target continuation supplied to the NGRAM teacher-forcing path."""

    output: str
    token_ids: tuple[int, ...]
    completion_tokens: int


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
    collect_sglang_spec_metrics: bool,
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
    if collect_sglang_spec_metrics:
        # SGLang extension: returns request-level speculative counters in meta_info.
        payload["return_meta_info"] = True
    payload.update(extra_body)
    data = json.dumps(payload).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    if collect_sglang_spec_metrics:
        # Lets the multi-instance gateway restore the SGLang-only payload field
        # after its typed OpenAI request round trip.
        headers["X-SGLang-Return-Meta-Info"] = "true"
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


def sglang_spec_metrics(response: dict[str, Any] | None) -> dict[str, int | float | None]:
    """Extract raw per-request counters when SGLang return_meta_info is enabled."""
    # SGLang's OpenAI chat response stores meta_info on each choice. Keep the
    # top-level fallback for compatible non-chat/proxy response formats.
    choices = (response or {}).get("choices") or []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    meta_info = first_choice.get("meta_info") or (response or {}).get("meta_info") or {}
    if not isinstance(meta_info, dict):
        meta_info = {}

    def as_int(key: str) -> int | None:
        try:
            value = meta_info.get(key)
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def as_float(key: str) -> float | None:
        try:
            value = meta_info.get(key)
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "spec_verify_ct": as_int("spec_verify_ct"),
        "spec_num_correct_drafts": as_int("spec_num_correct_drafts"),
        "spec_num_proposed_drafts": as_int("spec_num_proposed_drafts"),
        "spec_accept_length": as_float("spec_accept_length"),
        "spec_accept_rate": as_float("spec_accept_rate"),
    }


def response_meta_info(response: dict[str, Any] | None) -> dict[str, Any]:
    """Return SGLang metadata without requiring it for ordinary servers."""
    choices = (response or {}).get("choices") or []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    value = first_choice.get("meta_info") or (response or {}).get("meta_info")
    return value if isinstance(value, dict) else {}


def load_teacher_forcing_turns(
    trace_path: Path,
    *,
    tokenizer_path: str,
) -> dict[tuple[str, int], TeacherForcingTurn]:
    """Load prior benchmark outputs and encode them for SGLang teacher forcing."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    turns: dict[tuple[str, int], TeacherForcingTurn] = {}
    for row in read_jsonl(trace_path):
        if row.get("error"):
            raise ValueError(
                f"Teacher-forcing reference contains an error: {row.get('request_id')}"
            )
        key = (str(row["question_id"]), int(row["turn_id"]))
        if key in turns:
            raise ValueError(f"Duplicate teacher-forcing reference turn: {key}")
        output = str(row.get("output") or "")
        token_ids = tokenizer.encode(output, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Reference {key} has no output tokens")
        turns[key] = TeacherForcingTurn(
            output=output,
            token_ids=tuple(int(token) for token in token_ids),
            completion_tokens=int(row.get("completion_tokens") or len(token_ids)),
        )
    return turns


def run_item(
    item: BenchmarkItem,
    *,
    item_index: int,
    server_urls: tuple[str, ...],
    api_key: str,
    model: str,
    temperature: float,
    top_p: float,
    seed: int | None,
    max_tokens: int,
    timeout: float,
    extra_body: dict[str, Any],
    collect_sglang_spec_metrics: bool,
    teacher_forcing_turns: dict[tuple[str, int], TeacherForcingTurn],
    stop_on_error: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    conversation: list[dict[str, str]] = []
    for turn_id, user_text in enumerate(item.turns):
        conversation.append({"role": "user", "content": user_text})
        request_messages = [dict(message) for message in conversation]
        teacher_turn = teacher_forcing_turns.get((item.question_id, turn_id))
        request_extra_body = copy.deepcopy(extra_body)
        request_max_tokens = max_tokens
        if teacher_turn is not None:
            request_max_tokens = len(teacher_turn.token_ids)
            request_extra_body["ignore_eos"] = True
            request_extra_body["min_tokens"] = request_max_tokens
            custom_params = request_extra_body.setdefault("custom_params", {})
            if not isinstance(custom_params, dict):
                raise ValueError("extra_body.custom_params must be a JSON object")
            custom_params["ngram_teacher_forcing_token_ids"] = list(
                teacher_turn.token_ids
            )
            custom_params["ngram_teacher_forcing_record_id"] = (
                f"{item.benchmark}:{item.question_id}:turn{turn_id}"
            )

        request_server_url = server_urls[(item_index + turn_id) % len(server_urls)]
        response, latency_s, error = call_chat_completion(
            server_url=request_server_url,
            api_key=api_key,
            model=model,
            messages=request_messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            max_tokens=request_max_tokens,
            timeout=timeout,
            extra_body=request_extra_body,
            collect_sglang_spec_metrics=collect_sglang_spec_metrics,
        )
        output = extract_output(response)
        meta_info = response_meta_info(response)
        record = {
            "request_id": f"{item.benchmark}:{item.question_id}:turn{turn_id}",
            "server_url": request_server_url,
            "benchmark": item.benchmark,
            "question_id": item.question_id,
            "turn_id": turn_id,
            "category": item.category,
            "messages": request_messages,
            "output": output,
            "prompt_tokens": usage_value(response, "prompt_tokens"),
            "completion_tokens": usage_value(response, "completion_tokens"),
            "total_tokens": usage_value(response, "total_tokens"),
            **sglang_spec_metrics(response),
            "latency_s": round(latency_s, 6),
            "error": error,
            "model": model,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
            "max_tokens": max_tokens,
            "request_max_tokens": request_max_tokens,
            "teacher_forcing": teacher_turn is not None,
            "teacher_forcing_match": (
                output == teacher_turn.output if teacher_turn is not None else None
            ),
            "teacher_forcing_reference_completion_tokens": (
                teacher_turn.completion_tokens if teacher_turn is not None else None
            ),
            "meta_info": meta_info,
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
    forced = [row for row in records if row.get("teacher_forcing")]
    teacher_mismatches = sum(
        1 for row in forced if row.get("teacher_forcing_match") is not True
    )
    spec_records = [
        row
        for row in records
        if int(row.get("spec_verify_ct") or 0) > 0
        and row.get("spec_num_correct_drafts") is not None
        and row.get("spec_num_proposed_drafts") is not None
    ]
    spec_verify_ct = sum(int(row["spec_verify_ct"]) for row in spec_records)
    spec_correct_drafts = sum(
        int(row["spec_num_correct_drafts"]) for row in spec_records
    )
    spec_proposed_drafts = sum(
        int(row["spec_num_proposed_drafts"]) for row in spec_records
    )
    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        if len(ordered) == 1:
            return round(ordered[0], 6)
        pos = (len(ordered) - 1) * q
        lower = int(pos)
        upper = min(lower + 1, len(ordered) - 1)
        weight = pos - lower
        value = ordered[lower] * (1 - weight) + ordered[upper] * weight
        return round(value, 6)

    summary = {
        "turns": len(records),
        "errors": errors,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_avg_s": round(sum(latencies) / len(latencies), 6)
        if latencies
        else None,
        "latency_max_s": round(max(latencies), 6) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_p99_s": percentile(latencies, 0.99),
        "teacher_forced_turns": len(forced),
        "teacher_forcing_mismatches": teacher_mismatches,
    }
    if spec_verify_ct > 0:
        # These use sums of raw request counters, never a mean of request/worker means.
        summary.update(
            {
                "spec_metric_turns": len(spec_records),
                "spec_verify_ct": spec_verify_ct,
                "spec_num_correct_drafts": spec_correct_drafts,
                "spec_num_proposed_drafts": spec_proposed_drafts,
                "spec_accept_length": round(
                    (spec_correct_drafts + spec_verify_ct) / spec_verify_ct, 6
                ),
                "spec_draft_accept_length": round(
                    spec_correct_drafts / spec_verify_ct, 6
                ),
                "spec_accept_rate": round(
                    spec_correct_drafts / spec_proposed_drafts, 6)
                if spec_proposed_drafts > 0
                else None,
            }
        )
    return summary


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
    parser.add_argument(
        "--server-urls",
        nargs="+",
        default=None,
        help="Optional worker URLs used with client-side round-robin routing.",
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
        "--collect-sglang-spec-metrics",
        action="store_true",
        help=(
            "Request SGLang return_meta_info and aggregate raw per-request "
            "speculative counters in the JSONL and summary."
        ),
    )
    parser.add_argument(
        "--keep-going-after-error",
        action="store_true",
        help="Continue later turns of the same question after an error.",
    )
    parser.add_argument(
        "--teacher-forcing-trace",
        type=Path,
        default=None,
        help="Reference turn_traces.jsonl whose output tokens are forced by SGLang.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Local tokenizer path used to encode --teacher-forcing-trace.",
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

    teacher_forcing_turns: dict[tuple[str, int], TeacherForcingTurn] = {}
    if args.teacher_forcing_trace:
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required with --teacher-forcing-trace")
        teacher_forcing_turns = load_teacher_forcing_turns(
            args.teacher_forcing_trace.expanduser().resolve(),
            tokenizer_path=args.tokenizer,
        )
        expected_keys = {
            (item.question_id, turn_id)
            for item in items
            for turn_id in range(len(item.turns))
        }
        missing = sorted(expected_keys - teacher_forcing_turns.keys())
        if missing:
            raise SystemExit(
                f"Teacher-forcing trace is missing {len(missing)} selected turns; "
                f"first missing={missing[0]}"
            )

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
    if args.server_urls:
        print(f"server_urls: {args.server_urls}")
    print(f"model: {args.model}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"seed: {None if args.seed < 0 else args.seed}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"concurrency: {args.concurrency}")
    print(f"trace_output: {trace_path}")
    print(f"teacher_forced_turns: {len(teacher_forcing_turns)}")

    lock = threading.Lock()
    server_urls = tuple(args.server_urls or [args.server_url])
    all_records: list[dict[str, Any]] = []
    start = time.perf_counter()
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_to_item = {
            pool.submit(
                run_item,
                item,
                item_index=item_index,
                server_urls=server_urls,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=None if args.seed < 0 else args.seed,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body=args.extra_body,
                collect_sglang_spec_metrics=args.collect_sglang_spec_metrics,
                teacher_forcing_turns=teacher_forcing_turns,
                stop_on_error=not args.keep_going_after_error,
            ): item
            for item_index, item in enumerate(items)
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
        "server_urls": args.server_urls,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": None if args.seed < 0 else args.seed,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "items": len(items),
        "wall_time_s": round(wall_time_s, 6),
        "requests_s": round(len(all_records) / wall_time_s, 6)
        if wall_time_s > 0
        else None,
        "trace_output": str(trace_path),
        "teacher_forcing_trace": (
            str(args.teacher_forcing_trace.expanduser().resolve())
            if args.teacher_forcing_trace
            else None
        ),
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
