"""Replay a portable AgentSociety record against an OpenAI-compatible server.

This runner intentionally does not import AgentSociety.  Its input is the raw
record JSONL bundled with this directory, which already contains the complete
messages and request parameters for each LLM call.

Faithful replay preserves the recorded scheduler topology:
step -> pre_dispatch -> main -> post_intercept; agent-local calls are serial,
while independent agent chains run concurrently.  It does not rerun simulation
state transitions or feed newly generated text back into later prompts: later
recorded prompts remain the original simulation prompts.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from openai import AsyncOpenAI
except ImportError as exc:  # pragma: no cover - exercised in the target env.
    raise SystemExit("Install the runner dependency first: python -m pip install openai") from exc


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RECORD = SCRIPT_DIR / "records" / "agentsociety_record.jsonl"
PHASE_ORDER = ("pre_dispatch", "main", "post_intercept")


@dataclass
class Step:
    pre_dispatch: dict[int, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    main: dict[int, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))
    post_intercept: dict[int, list[dict[str, Any]]] = field(default_factory=lambda: defaultdict(list))


def resolve_record(path: Path) -> Path:
    path = path.expanduser().resolve()
    if path.is_file():
        return path
    if path.is_dir():
        candidates = sorted(path.glob("*.jsonl"), key=lambda item: item.stat().st_mtime)
        if candidates:
            return candidates[-1]
    raise FileNotFoundError(f"No record JSONL found at: {path}")


def read_records(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Record at {path}:{line_no} must be a JSON object")
            rows.append((line_no, value))
    return rows


def call_type(record: dict[str, Any]) -> str:
    for message in record.get("messages") or []:
        template_id = message.get("template_id")
        if template_id:
            return str(template_id)
    block_name = record.get("block_name")
    if block_name:
        return str(block_name)
    agent_class = record.get("agent_class")
    return f"agent:{agent_class}" if agent_class else "unknown"


def call_id(record: dict[str, Any], line_no: int) -> str:
    return "s{step}:{phase}:a{agent}:q{seq}:l{line}".format(
        step=record.get("step", 0),
        phase=record.get("phase", "main"),
        agent=record.get("agent_id", -1),
        seq=record.get("seq", 0),
        line=line_no,
    )


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_steps(
    rows: list[tuple[int, dict[str, Any]]], selected_types: set[str]
) -> list[Step]:
    grouped: dict[int, Step] = {}
    for line_no, record in rows:
        if selected_types and call_type(record) not in selected_types:
            continue
        step_id = as_int(record.get("step"))
        step = grouped.setdefault(step_id, Step())
        phase = str(record.get("phase", "main"))
        target = getattr(step, phase, step.main)
        record = dict(record)
        record["_record_line"] = line_no
        target[as_int(record.get("agent_id"), -1)].append(record)

    for step in grouped.values():
        for phase in PHASE_ORDER:
            for records in getattr(step, phase).values():
                records.sort(key=lambda record: as_int(record.get("seq")))
    return [grouped[step_id] for step_id in sorted(grouped)]


def build_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for message in record.get("messages") or []:
        content = "".join(
            str(segment.get("text", ""))
            for segment in message.get("segments") or []
            if isinstance(segment, dict)
        )
        messages.append({"role": str(message.get("role", "user")), "content": content})
    return messages


def extract_output(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    return str(content or "")


def usage(response: Any, key: str) -> int | None:
    value = getattr(getattr(response, "usage", None), key, None)
    return as_int(value) if value is not None else None


def spec_metrics(response: Any) -> dict[str, int | float | None]:
    # OpenAI SDK response models may not expose SGLang-only extra fields as
    # attributes, whereas model_dump() retains them in compatible releases.
    raw = response.model_dump() if response is not None and hasattr(response, "model_dump") else {}
    choices = raw.get("choices") or [] if isinstance(raw, dict) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    meta = choice.get("meta_info") or raw.get("meta_info") or {}
    if not isinstance(meta, dict):
        meta = {}

    def integer(name: str) -> int | None:
        value = meta.get(name)
        return as_int(value) if value is not None else None

    def number(name: str) -> float | None:
        try:
            value = meta.get(name)
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "spec_verify_ct": integer("spec_verify_ct"),
        "spec_num_correct_drafts": integer("spec_num_correct_drafts"),
        "spec_num_proposed_drafts": integer("spec_num_proposed_drafts"),
        "spec_accept_length": number("spec_accept_length"),
        "spec_accept_rate": number("spec_accept_rate"),
    }


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] * (upper - position) + ordered[upper] * (position - lower), 6)


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(row["latency_s"]) for row in records if row.get("latency_s") is not None]
    prompt_tokens = sum(as_int(row.get("prompt_tokens")) for row in records)
    completion_tokens = sum(as_int(row.get("completion_tokens")) for row in records)
    errors = sum(bool(row.get("error")) for row in records)
    spec_rows = [
        row
        for row in records
        if as_int(row.get("spec_verify_ct")) > 0
        and row.get("spec_num_correct_drafts") is not None
        and row.get("spec_num_proposed_drafts") is not None
    ]
    verify = sum(as_int(row["spec_verify_ct"]) for row in spec_rows)
    correct = sum(as_int(row["spec_num_correct_drafts"]) for row in spec_rows)
    proposed = sum(as_int(row["spec_num_proposed_drafts"]) for row in spec_rows)
    summary: dict[str, Any] = {
        "turns": len(records),
        "errors": errors,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "latency_avg_s": round(statistics.mean(latencies), 6) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_p99_s": percentile(latencies, 0.99),
        "latency_max_s": round(max(latencies), 6) if latencies else None,
    }
    if verify:
        summary.update(
            {
                "spec_metric_turns": len(spec_rows),
                "spec_verify_ct": verify,
                "spec_num_correct_drafts": correct,
                "spec_num_proposed_drafts": proposed,
                # Includes SGLang's one target/bonus token per verification.
                "spec_accept_length": round((correct + verify) / verify, 6),
                "spec_draft_accept_length": round(correct / verify, 6),
                "spec_accept_rate": round(correct / proposed, 6) if proposed else None,
            }
        )
    return summary


async def main_async(args: argparse.Namespace) -> int:
    record_path = resolve_record(args.record)
    selected_types = {item.strip() for item in args.call_types.split(",") if item.strip()}
    steps = build_steps(read_records(record_path), selected_types)
    request_count = sum(
        len(requests)
        for step in steps
        for phase in PHASE_ORDER
        for requests in getattr(step, phase).values()
    )
    if not request_count:
        raise SystemExit("No replayable records selected.")

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "replay_trace.jsonl"
    summary_path = output_dir / "summary.json"
    trace_path.unlink(missing_ok=True)

    print(f"record_path: {record_path}")
    print(f"steps: {len(steps)} requests: {request_count}")
    print(f"mode: {args.mode} max_concurrency: {args.max_concurrency}")
    print(f"trace_output: {trace_path}")

    client = AsyncOpenAI(
        base_url=args.server_url.rstrip("/"),
        api_key=args.api_key,
        timeout=args.timeout,
        # Patched multi-instance gateways turn this header into the SGLang
        # return_meta_info extension before forwarding to a worker.
        default_headers=(
            {"X-SGLang-Return-Meta-Info": "true"}
            if args.collect_sglang_spec_metrics
            else None
        ),
    )
    semaphore = asyncio.Semaphore(args.max_concurrency)
    write_lock = asyncio.Lock()
    replay_rows: list[dict[str, Any]] = []
    completed = 0
    start = time.perf_counter()

    async def fire(record: dict[str, Any]) -> dict[str, Any]:
        nonlocal completed
        request = record.get("request") or {}
        payload: dict[str, Any] = {
            "model": args.model or str(request.get("model_hint") or "default"),
            "messages": build_messages(record),
            "temperature": args.temperature if args.temperature is not None else request.get("temperature", 1.0),
            "max_tokens": args.max_tokens if args.max_tokens is not None else request.get("max_tokens", 512),
        }
        for name in ("response_format", "tools", "tool_choice"):
            if request.get(name) is not None:
                payload[name] = request[name]
        response: Any = None
        error = ""
        async with semaphore:
            # Match the original AgentSociety replay: request latency starts
            # only once the global in-flight slot is acquired, not while the
            # caller waits for the local replay concurrency limit.
            started = time.perf_counter()
            try:
                # The OpenAI SDK validates named arguments, so SGLang-only
                # fields must be carried in extra_body rather than payload.
                # The companion header lets the patched gateway preserve this
                # field across its typed OpenAI request model.
                request_options: dict[str, Any] = {}
                if args.collect_sglang_spec_metrics:
                    request_options["extra_body"] = {"return_meta_info": True}
                response = await client.chat.completions.create(
                    **payload, **request_options
                )
            except Exception as exc:  # Preserve replay progress after a bad request.
                error = repr(exc)
        latency = time.perf_counter() - started
        line_no = as_int(record.get("_record_line"))
        row = {
            "call_id": call_id(record, line_no),
            "record_line": line_no,
            "agent_id": record.get("agent_id"),
            "agent_class": record.get("agent_class"),
            "simulation_step": record.get("step"),
            "phase": record.get("phase", "main"),
            "seq": record.get("seq", 0),
            "call_type": call_type(record),
            "messages": payload["messages"],
            "output": extract_output(response),
            "recorded_output": str((record.get("response") or {}).get("text", "") or ""),
            "prompt_tokens": usage(response, "prompt_tokens"),
            "completion_tokens": usage(response, "completion_tokens"),
            "total_tokens": usage(response, "total_tokens"),
            **spec_metrics(response),
            "latency_s": round(latency, 6),
            "error": error or None,
            "model": payload["model"],
            "temperature": payload["temperature"],
            "max_tokens": payload["max_tokens"],
        }
        async with write_lock:
            with trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            replay_rows.append(row)
            completed += 1
            if args.print_every > 0 and (completed == request_count or completed % args.print_every == 0):
                print(f"[{completed}/{request_count}] elapsed={time.perf_counter() - start:.1f}s")
                sys.stdout.flush()
        return row

    async def fire_chain(records: list[dict[str, Any]]) -> None:
        for record in records:
            await fire(record)

    for step in steps:
        pre = getattr(step, "pre_dispatch")
        if args.mode == "aggressive":
            await asyncio.gather(*(fire(record) for records in pre.values() for record in records))
        else:
            await asyncio.gather(*(fire_chain(records) for records in pre.values()))
        await asyncio.gather(*(fire_chain(records) for records in step.main.values()))
        for records in step.post_intercept.values():
            await fire_chain(records)

    await client.close()
    wall_time_s = time.perf_counter() - start
    summary = {
        "benchmark": "agentsociety_replay",
        "record_path": str(record_path),
        "server_url": args.server_url,
        "model": args.model,
        "mode": args.mode,
        # Keep the generic benchmark name while retaining the replay-specific
        # spelling for compatibility with prior AgentSociety outputs.
        "concurrency": args.max_concurrency,
        "max_concurrency": args.max_concurrency,
        "call_types": sorted(selected_types),
        "wall_time_s": round(wall_time_s, 6),
        "requests_s": round(len(replay_rows) / wall_time_s, 6) if wall_time_s else None,
        "trace_output": str(trace_path),
        **summarize(replay_rows),
    }
    if summary["completion_tokens"]:
        summary["completion_tokens_s"] = round(summary["completion_tokens"] / wall_time_s, 6)
    if summary["total_tokens"]:
        summary["total_tokens_s"] = round(summary["total_tokens"] / wall_time_s, 6)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary_output: {summary_path}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay a portable AgentSociety record.")
    parser.add_argument("--record", type=Path, default=DEFAULT_RECORD)
    parser.add_argument("--server-url", default="http://127.0.0.1:1919/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--api-key", default="dummy")
    parser.add_argument("--mode", choices=("faithful", "aggressive"), default="faithful")
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=None, help="Override recorded temperature.")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override recorded max_tokens.")
    parser.add_argument("--call-types", default="", help="Optional comma-separated call-type filter.")
    parser.add_argument("--collect-sglang-spec-metrics", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.max_concurrency < 1:
        raise SystemExit("--max-concurrency must be >= 1")
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
