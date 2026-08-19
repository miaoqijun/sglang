"""Replay converted SWE-bench/OpenHands LLM-call traces.

This runner is workload-oriented. It sends each converted LLM call to an
OpenAI-compatible chat server, records per-request latency and token usage, and
writes a summary for serving-performance analysis. It does not grade
SWE-bench correctness.
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
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DEFAULT_TRACE = BENCHMARK_DIR / "outputs" / "swe_bench" / "openhands_llm_calls.jsonl"
DEFAULT_OUTPUT_ROOT = BENCHMARK_DIR / "outputs" / "swe_bench_replay"
DEFAULT_TOOL_PROMPT = (
    SCRIPT_DIR / "tool_definitions" / "openhands_tools_plain_prompt.md"
)
DEFAULT_TOOL_SCHEMA = SCRIPT_DIR / "tool_definitions" / "tools_schema.json"


@dataclass(frozen=True)
class TeacherForcingTurn:
    output: str
    token_ids: tuple[int, ...]


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


def group_workflows(
    rows: list[dict[str, Any]],
    *,
    limit_workflows: int | None,
    max_steps_per_workflow: int | None,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        workflow_id = str(row.get("workflow_id", ""))
        grouped.setdefault(workflow_id, []).append(row)

    workflows: list[list[dict[str, Any]]] = []
    # Stable workflow and step ordering makes LIMIT reproducible and defines
    # the serial dependency order later enforced by run_workflow().
    for workflow_id in sorted(grouped, key=lambda value: int(value) if value.isdigit() else value):
        workflow_rows = sorted(
            grouped[workflow_id],
            key=lambda row: (
                int(row.get("step_id") or 0),
                int(row.get("call_id") or 0),
            ),
        )
        if max_steps_per_workflow is not None:
            workflow_rows = workflow_rows[:max_steps_per_workflow]
        if not workflow_rows:
            continue
        workflows.append(workflow_rows)
        if limit_workflows is not None and len(workflows) >= limit_workflows:
            break
    return workflows


def request_key(row: dict[str, Any]) -> tuple[str, str, str]:
    """Return the stable identity used by exported SWE replay failure lists."""
    return (
        str(row.get("workflow_id", "")),
        str(row.get("step_id", "")),
        str(row.get("call_id", "")),
    )


def load_failure_keys(path: Path) -> set[tuple[str, str, str]]:
    """Load request identities written by --failure-list-output."""
    return {request_key(item) for item in read_jsonl(path)}


def load_teacher_forcing_turns(
    path: Path, *, tokenizer_path: str
) -> dict[tuple[str, str, str], TeacherForcingTurn]:
    """Load source outputs and encode them with the serving model tokenizer."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    turns: dict[tuple[str, str, str], TeacherForcingTurn] = {}
    for row in read_jsonl(path):
        if row.get("error"):
            raise ValueError(
                f"Teacher-forcing reference contains an error: {request_key(row)}"
            )
        key = request_key(row)
        if key in turns:
            raise ValueError(f"Duplicate teacher-forcing reference: {key}")
        output = str(row.get("output") or "")
        token_ids = tokenizer.encode(output, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Teacher-forcing reference has empty output: {key}")
        turns[key] = TeacherForcingTurn(
            output=output,
            token_ids=tuple(int(token) for token in token_ids),
        )
    return turns


def failure_reasons(record: dict[str, Any]) -> list[str]:
    """Identify requests unsuitable for a later fixed-workload replay."""
    reasons: list[str] = []
    if record.get("error"):
        reasons.append("request_error")
    if record.get("finish_reason") == "length":
        reasons.append("max_tokens_reached")
    if record.get("source_has_tool_call") and not record.get(
        "generated_has_tool_call"
    ):
        reasons.append("expected_tool_call_not_completed")
    return reasons


def write_failure_list(
    path: Path, *, records: list[dict[str, Any]], trace_path: Path
) -> int:
    """Write failed request identities and enough context to audit them."""
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            reasons = failure_reasons(record)
            if not reasons:
                continue
            payload = {
                "workflow_id": record.get("workflow_id"),
                "step_id": record.get("step_id"),
                "call_id": record.get("call_id"),
                "request_id": record.get("request_id"),
                "trace_jsonl": str(trace_path),
                "reasons": reasons,
                "finish_reason": record.get("finish_reason"),
                "error": record.get("error"),
                "source_tool_names": record.get("source_tool_names", []),
                "model": record.get("model"),
                "tool_mode": record.get("tool_mode"),
                "tool_choice": record.get("tool_choice"),
                "max_tokens": record.get("max_tokens"),
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def endpoint(server_url: str) -> str:
    base = server_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def server_info_urls(server_url: str) -> list[str]:
    base = server_url.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    return [f"{base}/server_info", f"{base}/model_info"]


def fetch_json(url: str, *, api_key: str, timeout: float = 10.0) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def fetch_server_info(server_url: str, *, api_key: str, timeout: float = 10.0) -> dict[str, Any]:
    for url in server_info_urls(server_url):
        payload = fetch_json(url, api_key=api_key, timeout=timeout)
        if payload:
            return payload
    return {}


def first_internal_state(server_info: dict[str, Any]) -> dict[str, Any]:
    states = server_info.get("internal_states") or []
    for state in states:
        if isinstance(state, dict):
            return state
    return server_info


def extract_accept_len(server_info: dict[str, Any]) -> Any:
    state = first_internal_state(server_info)
    return state.get("avg_spec_accept_length", server_info.get("avg_spec_accept_length"))


def extract_spec_metrics(server_info: dict[str, Any]) -> dict[str, Any]:
    state = first_internal_state(server_info)
    summary = server_info.get("spec_metrics_summary") or state.get(
        "spec_metrics_summary"
    ) or {}
    spec = summary.get("SpecMetrics") or {}
    lifetime_accept_len = extract_accept_len(server_info)
    draft_tokens = state.get(
        "speculative_num_draft_tokens",
        server_info.get("speculative_num_draft_tokens"),
    )
    lifetime_accept_rate = None
    try:
        if lifetime_accept_len is not None and float(draft_tokens) > 0:
            lifetime_accept_rate = float(lifetime_accept_len) / float(draft_tokens)
    except (TypeError, ValueError):
        pass
    return {
        "spec_draft_attempts_total": spec.get("draft_attempts_total"),
        "spec_draft_tokens_total": spec.get("draft_tokens_total"),
        "spec_verified_tokens_total": spec.get("verified_tokens_total"),
        "spec_accepted_tokens_total": spec.get("accepted_tokens_total"),
        "spec_rejected_tokens_total": spec.get("rejected_tokens_total"),
        "spec_true_accept_rate": spec.get("acceptance_rate"),
        "spec_true_mean_accept_len": spec.get("mean_accepted_tokens"),
        "spec_zero_accept_ratio": spec.get("zero_accept_ratio"),
        "spec_lifetime_accept_len": lifetime_accept_len,
        "spec_lifetime_accept_rate": lifetime_accept_rate,
    }


def call_chat_completion(
    *,
    server_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float,
    top_p: float,
    seed: int | None,
    max_tokens: int,
    timeout: float,
    extra_body: dict[str, Any],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | None,
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
    if tools:
        payload["tools"] = tools
        if tool_choice:
            payload["tool_choice"] = tool_choice
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
        return json.loads(raw), time.perf_counter() - start, None
    except urllib.error.HTTPError as exc:
        try:
            body = exc.read().decode("utf-8", errors="replace")
        except Exception:
            body = ""
        return None, time.perf_counter() - start, f"HTTP {exc.code}: {body[:1000]}"
    except Exception as exc:
        return None, time.perf_counter() - start, repr(exc)


def extract_response_message(
    response: dict[str, Any] | None,
) -> tuple[str, list[dict[str, Any]], str | None]:
    if not response:
        return "", [], None
    choices = response.get("choices") or []
    if not choices:
        return "", [], None
    first = choices[0]
    message = first.get("message") or {}
    if not isinstance(message, dict):
        return str(first.get("text") or ""), [], first.get("finish_reason")
    content = "" if message.get("content") is None else str(message["content"])
    tool_calls = message.get("tool_calls")
    return (
        content,
        tool_calls if isinstance(tool_calls, list) else [],
        first.get("finish_reason"),
    )


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


def chat_message_from_record(message: dict[str, Any]) -> dict[str, Any]:
    role = str(message.get("role") or "user")
    text = str(message.get("text") or "")
    chat_message: dict[str, Any] = {"role": role, "content": text}
    if role == "assistant" and message.get("tool_calls"):
        chat_message["tool_calls"] = message["tool_calls"]
    if role == "tool":
        if message.get("tool_call_id") is not None:
            chat_message["tool_call_id"] = str(message["tool_call_id"])
        if message.get("name") is not None:
            chat_message["name"] = str(message["name"])
    return chat_message


def with_tool_prompt(
    messages: list[dict[str, Any]], tool_prompt: str | None
) -> tuple[list[dict[str, Any]], bool]:
    if not tool_prompt:
        return messages, False

    for index, message in enumerate(messages):
        if message.get("role") != "system":
            continue
        updated_messages = list(messages)
        updated_message = dict(message)
        content = str(updated_message.get("content") or "").rstrip()
        updated_message["content"] = f"{content}\n\n{tool_prompt}" if content else tool_prompt
        updated_messages[index] = updated_message
        return updated_messages, True

    return [{"role": "system", "content": tool_prompt}, *messages], True


def messages_for_row(
    row: dict[str, Any], tool_prompt: str | None
) -> tuple[list[dict[str, Any]], bool, bool]:
    recorded_messages = row.get("messages")
    if isinstance(recorded_messages, list) and recorded_messages:
        messages = [
            chat_message_from_record(message)
            for message in recorded_messages
            if isinstance(message, dict)
        ]
        if messages:
            messages, tool_prompt_applied = with_tool_prompt(messages, tool_prompt)
            return messages, True, tool_prompt_applied
    prompt = str(row.get("prompt") or "")
    messages, tool_prompt_applied = with_tool_prompt(
        [{"role": "user", "content": prompt}], tool_prompt
    )
    return messages, False, tool_prompt_applied


def run_row(
    row: dict[str, Any],
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
    tool_prompt: str | None,
    tool_prompt_path: str | None,
    tools: list[dict[str, Any]] | None,
    tool_schema_path: str | None,
    tool_mode: str,
    tool_choice: str | None,
    collect_sglang_spec_metrics: bool,
    teacher_forcing_turns: dict[tuple[str, str, str], TeacherForcingTurn],
) -> dict[str, Any]:
    messages, used_recorded_messages, tool_prompt_applied = messages_for_row(
        row, tool_prompt
    )
    teacher_turn = teacher_forcing_turns.get(request_key(row))
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
            f"swe_bench:{row.get('workflow_id')}:"
            f"step{row.get('step_id')}:call{row.get('call_id')}"
        )

    response, latency_s, error = call_chat_completion(
        server_url=server_url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        max_tokens=request_max_tokens,
        timeout=timeout,
        extra_body=request_extra_body,
        tools=tools,
        tool_choice=tool_choice,
        collect_sglang_spec_metrics=collect_sglang_spec_metrics,
    )
    output, generated_tool_calls, finish_reason = extract_response_message(response)
    return {
        "request_id": f"swe_bench:{row.get('workflow_id')}:step{row.get('step_id')}",
        "benchmark": "swe_bench",
        "call_id": row.get("call_id"),
        "workflow_id": row.get("workflow_id"),
        "step_id": row.get("step_id"),
        "message_index": row.get("message_index"),
        "messages": messages,
        "used_recorded_messages": used_recorded_messages,
        "recorded_output": row.get("output", ""),
        "output": output,
        "generated_tool_calls": generated_tool_calls,
        "generated_has_tool_call": bool(generated_tool_calls),
        "finish_reason": finish_reason,
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
        "source_prompt_char_length": row.get("prompt_char_length"),
        "source_output_char_length": row.get("output_char_length"),
        "source_has_tool_call": row.get("has_tool_call"),
        "source_tool_names": row.get("tool_names", []),
        "tool_prompt_applied": tool_prompt_applied,
        "tool_prompt_path": tool_prompt_path,
        "tool_mode": tool_mode,
        "tool_schema_path": tool_schema_path,
        "tool_choice": tool_choice,
    }


def run_workflow(
    workflow_rows: list[dict[str, Any]],
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
    tool_prompt: str | None,
    tool_prompt_path: str | None,
    tools: list[dict[str, Any]] | None,
    tool_schema_path: str | None,
    tool_mode: str,
    tool_choice: str | None,
    collect_sglang_spec_metrics: bool,
    teacher_forcing_turns: dict[tuple[str, str, str], TeacherForcingTurn],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    # Workflow parallelism is created by the caller. Keep its steps serial so
    # a later source prompt is never submitted before its predecessor returns.
    for row in workflow_rows:
        records.append(
            run_row(
                row,
                server_url=server_url,
                api_key=api_key,
                model=model,
                temperature=temperature,
                top_p=top_p,
                seed=seed,
                max_tokens=max_tokens,
                timeout=timeout,
                extra_body=extra_body,
                tool_prompt=tool_prompt,
                tool_prompt_path=tool_prompt_path,
                tools=tools,
                tool_schema_path=tool_schema_path,
                tool_mode=tool_mode,
                tool_choice=tool_choice,
                collect_sglang_spec_metrics=collect_sglang_spec_metrics,
                teacher_forcing_turns=teacher_forcing_turns,
            )
        )
    return records


def write_jsonl_line(path: Path, lock: threading.Lock, row: dict[str, Any]) -> None:
    text = json.dumps(row, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(text + "\n")


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
    generated_tool_call_turns = 0
    generated_tool_calls_total = 0
    generated_tool_name_match_turns = 0
    expected_tool_call_turns = 0
    for row in records:
        expected_names = {
            str(name) for name in (row.get("source_tool_names") or []) if name
        }
        if expected_names:
            expected_tool_call_turns += 1
        generated_calls = row.get("generated_tool_calls") or []
        if not generated_calls:
            continue
        generated_tool_call_turns += 1
        generated_tool_calls_total += len(generated_calls)
        generated_names = {
            str((call.get("function") or {}).get("name"))
            for call in generated_calls
            if isinstance(call, dict) and isinstance(call.get("function"), dict)
        }
        if expected_names and expected_names.intersection(generated_names):
            generated_tool_name_match_turns += 1
    summary = {
        "turns": len(records),
        "errors": errors,
        "teacher_forced_turns": len(forced),
        "teacher_forcing_mismatches": teacher_mismatches,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "expected_tool_call_turns": expected_tool_call_turns,
        "generated_tool_call_turns": generated_tool_call_turns,
        "generated_tool_calls_total": generated_tool_calls_total,
        "generated_tool_name_match_turns": generated_tool_name_match_turns,
        "latency_avg_s": round(sum(latencies) / len(latencies), 6)
        if latencies
        else None,
        "latency_max_s": round(max(latencies), 6) if latencies else None,
        "latency_p50_s": percentile(latencies, 0.50),
        "latency_p90_s": percentile(latencies, 0.90),
        "latency_p99_s": percentile(latencies, 0.99),
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


def load_tool_schema(path: Path) -> list[dict[str, Any]]:
    try:
        # The schema may be edited on Windows, which commonly writes a UTF-8 BOM.
        value = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid tool schema JSON at {path}: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise SystemExit(f"Tool schema must be a JSON array of objects: {path}")
    if not value:
        raise SystemExit(f"Tool schema is empty: {path}")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay converted SWE-bench/OpenHands LLM-call traces as a causal "
            "serving workload. Steps within one workflow are serial; separate "
            "workflows may run concurrently."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python SD_benchmark/swe_bench/run_swe_trace.py "
            "--trace-jsonl SD_benchmark/swe_bench/mini_swe_qwen25_coder_32b_50workflows.jsonl "
            "--server-url http://127.0.0.1:1919/v1 --model Qwen2.5-14B-Instruct "
            "--tool-mode none --limit 20 --max-steps-per-workflow 20 "
            "--output-dir SD_benchmark/outputs/swe_bench/smoke\n\n"
            "Generated commands are recorded but not executed."
        ),
    )
    parser.add_argument("--trace-jsonl", type=Path, default=DEFAULT_TRACE, help="Converted workflow JSONL to replay.")
    parser.add_argument("--server-url", default="http://127.0.0.1:1919/v1", help="OpenAI-compatible chat-completions base URL.")
    parser.add_argument("--model", "--model-name", "--model_name", required=True, help="Served model name.")
    parser.add_argument("--api-key", default="dummy", help="Bearer token for the server.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature.")
    parser.add_argument("--top-p", "--top_p", type=float, default=1.0, help="Top-p sampling value.")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed.")
    parser.add_argument("--max-tokens", "--max_tokens", type=int, default=1024, help="Maximum completion tokens per turn.")
    parser.add_argument("--concurrency", type=int, default=1, help="Maximum workflows in flight; steps remain serial within each workflow.")
    parser.add_argument("--timeout", type=float, default=600.0, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of workflows/tasks to replay. It does not limit raw LLM-call rows.",
    )
    parser.add_argument(
        "--max-steps-per-workflow",
        type=int,
        default=None,
        help=(
            "Maximum number of LLM calls to replay from each workflow. This "
            "keeps intra-workflow causal order while avoiding very long later "
            "trajectory states."
        ),
    )
    parser.add_argument(
        "--tool-mode",
        choices=("api", "plain", "none"),
        default="api",
        help="Use OpenAI tools, the legacy plain tool prompt, or neither.",
    )
    parser.add_argument(
        "--tool-schema",
        type=Path,
        default=DEFAULT_TOOL_SCHEMA,
        help="OpenAI-compatible tool schema used when --tool-mode=api.",
    )
    parser.add_argument(
        "--tool-choice",
        choices=("auto", "required", "none"),
        default="required",
        help="OpenAI tool_choice sent when --tool-mode=api.",
    )
    parser.add_argument(
        "--tool-prompt",
        type=Path,
        default=DEFAULT_TOOL_PROMPT,
        help="Plain-text tool instructions used only when --tool-mode=plain.",
    )
    parser.add_argument("--extra-body", type=parse_extra_body, default={}, help="JSON object merged into every chat-completion request.")
    parser.add_argument(
        "--collect-sglang-spec-metrics",
        action="store_true",
        help=(
            "Request SGLang return_meta_info and aggregate raw per-request "
            "speculative counters in the JSONL and summary."
        ),
    )
    parser.add_argument(
        "--teacher-forcing-trace",
        type=Path,
        default=None,
        help="Reference JSONL generated by build_teacher_forcing_trace.py.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Local serving-model tokenizer path required for teacher forcing.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for replay_trace.jsonl and summary.json.")
    parser.add_argument("--print-every", type=int, default=20, help="Print progress every N completed requests.")
    parser.add_argument(
        "--server-info-output",
        type=Path,
        default=None,
        help="Optional path to save /server_info after the replay.",
    )
    parser.add_argument(
        "--failure-list-output",
        type=Path,
        default=None,
        help=(
            "Write JSONL identities for requests that error, reach max_tokens, "
            "or fail to complete an expected tool call."
        ),
    )
    parser.add_argument(
        "--skip-failure-list",
        type=Path,
        default=None,
        help=(
            "JSONL created by --failure-list-output. Matching requests are "
            "removed before replay."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be >= 1")
    if args.max_steps_per_workflow is not None and args.max_steps_per_workflow < 1:
        raise SystemExit("--max-steps-per-workflow must be >= 1")

    trace_path = args.trace_jsonl.expanduser().resolve()
    tool_prompt_path: Path | None = None
    tool_prompt: str | None = None
    tool_schema_path: Path | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | None = None
    if args.tool_mode == "plain":
        tool_prompt_path = args.tool_prompt.expanduser().resolve()
        if not tool_prompt_path.is_file():
            raise SystemExit(f"Tool prompt file does not exist: {tool_prompt_path}")
        tool_prompt = tool_prompt_path.read_text(encoding="utf-8").strip()
        if not tool_prompt:
            raise SystemExit(f"Tool prompt file is empty: {tool_prompt_path}")
    elif args.tool_mode == "api":
        tool_schema_path = args.tool_schema.expanduser().resolve()
        if not tool_schema_path.is_file():
            raise SystemExit(f"Tool schema file does not exist: {tool_schema_path}")
        tools = load_tool_schema(tool_schema_path)
        tool_choice = args.tool_choice

    rows = read_jsonl(trace_path)
    workflows = group_workflows(
        rows,
        limit_workflows=args.limit,
        max_steps_per_workflow=args.max_steps_per_workflow,
    )
    skip_failure_list: Path | None = None
    skipped_failure_calls = 0
    if args.skip_failure_list:
        skip_failure_list = args.skip_failure_list.expanduser().resolve()
        if not skip_failure_list.is_file():
            raise SystemExit(f"Failure list does not exist: {skip_failure_list}")
        failure_keys = load_failure_keys(skip_failure_list)
        filtered_workflows: list[list[dict[str, Any]]] = []
        for workflow in workflows:
            kept = [row for row in workflow if request_key(row) not in failure_keys]
            skipped_failure_calls += len(workflow) - len(kept)
            if kept:
                filtered_workflows.append(kept)
        workflows = filtered_workflows
    if not workflows:
        raise SystemExit(f"No rows selected from {trace_path}")
    selected_rows = [row for workflow in workflows for row in workflow]

    teacher_forcing_turns: dict[tuple[str, str, str], TeacherForcingTurn] = {}
    teacher_forcing_trace: Path | None = None
    if args.teacher_forcing_trace:
        if not args.tokenizer:
            raise SystemExit("--tokenizer is required with --teacher-forcing-trace")
        teacher_forcing_trace = args.teacher_forcing_trace.expanduser().resolve()
        teacher_forcing_turns = load_teacher_forcing_turns(
            teacher_forcing_trace, tokenizer_path=args.tokenizer
        )
        missing = [
            request_key(row)
            for row in selected_rows
            if request_key(row) not in teacher_forcing_turns
        ]
        if missing:
            raise SystemExit(
                f"Teacher-forcing reference is missing {len(missing)} selected calls; "
                f"first missing={missing[0]}"
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else DEFAULT_OUTPUT_ROOT / timestamp
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_trace = output_dir / "turn_traces.jsonl"
    summary_path = output_dir / "summary.json"
    server_info_output = args.server_info_output or (output_dir / "server_info.json")
    if out_trace.exists():
        out_trace.unlink()

    print("benchmark: swe_bench")
    print(f"trace_jsonl: {trace_path}")
    print(f"workflows: {len(workflows)}")
    print(f"calls: {len(selected_rows)}")
    print(f"server_url: {args.server_url}")
    print(f"model: {args.model}")
    print(f"temperature: {args.temperature}")
    print(f"top_p: {args.top_p}")
    print(f"seed: {None if args.seed < 0 else args.seed}")
    print(f"max_tokens: {args.max_tokens}")
    print(f"concurrency: {args.concurrency}")
    print(f"max_steps_per_workflow: {args.max_steps_per_workflow}")
    print(f"tool_mode: {args.tool_mode}")
    print(f"tool_schema: {tool_schema_path or 'disabled'}")
    print(f"tool_prompt: {tool_prompt_path or 'disabled'}")
    print(f"tool_choice: {tool_choice or 'disabled'}")
    print(f"skip_failure_list: {skip_failure_list or 'disabled'}")
    print(f"skipped_failure_calls: {skipped_failure_calls}")
    print(f"teacher_forced_turns: {len(teacher_forcing_turns)}")
    print(f"trace_output: {out_trace}")

    lock = threading.Lock()
    all_records: list[dict[str, Any]] = []
    start = time.perf_counter()
    completed = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        # Submit one serial chain per workflow. The executor limits concurrent
        # workflows, rather than allowing independent calls within a workflow.
        future_to_workflow = {
            pool.submit(
                run_workflow,
                workflow_rows,
                server_url=args.server_url,
                api_key=args.api_key,
                model=args.model,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=None if args.seed < 0 else args.seed,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                extra_body=args.extra_body,
                tool_prompt=tool_prompt,
                tool_prompt_path=str(tool_prompt_path) if tool_prompt_path else None,
                tools=tools,
                tool_schema_path=str(tool_schema_path) if tool_schema_path else None,
                tool_mode=args.tool_mode,
                tool_choice=tool_choice,
                collect_sglang_spec_metrics=args.collect_sglang_spec_metrics,
                teacher_forcing_turns=teacher_forcing_turns,
            ): workflow_rows
            for workflow_rows in workflows
        }
        for future in concurrent.futures.as_completed(future_to_workflow):
            workflow_rows = future_to_workflow[future]
            try:
                records = future.result()
            except Exception as exc:
                source_row = workflow_rows[0]
                records = [
                    {
                        "request_id": f"swe_bench:{source_row.get('workflow_id')}:workflow_error",
                        "benchmark": "swe_bench",
                        "call_id": source_row.get("call_id"),
                        "workflow_id": source_row.get("workflow_id"),
                        "step_id": None,
                        "message_index": None,
                        "messages": [],
                        "used_recorded_messages": False,
                        "recorded_output": "",
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
                write_jsonl_line(out_trace, lock, record)
            all_records.extend(records)
            completed += 1
            if args.print_every > 0 and (
                completed == len(workflows) or completed % args.print_every == 0
            ):
                elapsed = time.perf_counter() - start
                print(
                    f"[{completed}/{len(workflows)} workflows] "
                    f"elapsed={elapsed:.1f}s turns={len(all_records)}"
                )
                sys.stdout.flush()

    wall_time_s = time.perf_counter() - start
    server_info = fetch_server_info(args.server_url, api_key=args.api_key)
    server_info_output.write_text(
        json.dumps(server_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary = {
        "benchmark": "swe_bench",
        "trace_jsonl": str(trace_path),
        "server_url": args.server_url,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": None if args.seed < 0 else args.seed,
        "max_tokens": args.max_tokens,
        "concurrency": args.concurrency,
        "max_steps_per_workflow": args.max_steps_per_workflow,
        "skip_failure_list": str(skip_failure_list) if skip_failure_list else None,
        "teacher_forcing_trace": (
            str(teacher_forcing_trace) if teacher_forcing_trace else None
        ),
        "skipped_failure_calls": skipped_failure_calls,
        "tool_mode": args.tool_mode,
        "tool_schema_path": str(tool_schema_path) if tool_schema_path else None,
        "tool_choice": tool_choice,
        "tool_count": len(tools) if tools else 0,
        "tool_prompt_path": str(tool_prompt_path) if tool_prompt_path else None,
        "tool_prompt_chars": len(tool_prompt) if tool_prompt else 0,
        "items": len(workflows),
        "source_calls": len(selected_rows),
        "wall_time_s": round(wall_time_s, 6),
        "requests_s": round(len(all_records) / wall_time_s, 6)
        if wall_time_s > 0
        else None,
        "trace_output": str(out_trace),
        "server_info_path": str(server_info_output),
        "accept_len_mean": extract_accept_len(server_info),
        **extract_spec_metrics(server_info),
        **summarize(all_records),
    }
    if args.failure_list_output:
        failure_list_output = args.failure_list_output.expanduser().resolve()
        summary["failure_list_output"] = str(failure_list_output)
        summary["failure_list_count"] = write_failure_list(
            failure_list_output,
            records=all_records,
            trace_path=trace_path,
        )
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
