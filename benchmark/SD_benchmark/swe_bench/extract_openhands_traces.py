"""Convert OpenHands SWE-bench trajectories into LLM-call JSONL traces.

The input trajectories are expected to be JSON files whose top-level value is a
conversation list. Each assistant message is treated as one recorded LLM call:
all preceding messages form the prompt, and the assistant message is the
recorded output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = BENCHMARK_DIR / "outputs" / "swe_bench" / "openhands_llm_calls.jsonl"


def text_from_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("text") is not None:
                    parts.append(str(item.get("text") or ""))
                elif item.get("content") is not None:
                    parts.append(text_from_content(item.get("content")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if content.get("text") is not None:
            return str(content.get("text") or "")
        return json.dumps(content, ensure_ascii=False, sort_keys=True)
    return str(content)


def normalized_message(message: dict[str, Any], message_index: int) -> dict[str, Any]:
    row: dict[str, Any] = {
        "message_index": message_index,
        "role": str(message.get("role", "")),
        "text": text_from_content(message.get("content")),
    }
    for key in ("name", "tool_call_id"):
        if message.get(key) is not None:
            row[key] = message[key]
    if message.get("tool_calls") is not None:
        row["tool_calls"] = message["tool_calls"]
    return row


def format_prompt(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for index, message in enumerate(messages):
        original_index = int(message.get("message_index", index))
        role = message.get("role", "")
        text = message.get("text", "")
        if text:
            parts.append(f"<{role} #{original_index}>\n{text}")
        if message.get("tool_calls"):
            tool_calls = json.dumps(message["tool_calls"], ensure_ascii=False)
            parts.append(f"<{role} tool_calls #{original_index}>\n{tool_calls}")
    return "\n\n".join(parts)


def format_assistant_output(text: str, tool_calls: Any) -> str:
    parts: list[str] = []
    if text:
        parts.append(text)
    if tool_calls:
        parts.append("<tool_calls>\n" + json.dumps(tool_calls, ensure_ascii=False))
    return "\n\n".join(parts)


def load_trajectory(path: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc

    if isinstance(data, list):
        messages = data
    elif isinstance(data, dict):
        for key in ("messages", "history", "trajectory", "events"):
            value = data.get(key)
            if isinstance(value, list):
                messages = value
                break
        else:
            raise ValueError(f"No message list found in {path}")
    else:
        raise ValueError(f"Unsupported trajectory root in {path}: {type(data).__name__}")

    normalized = []
    for item in messages:
        if isinstance(item, dict):
            normalized.append(item)
    return normalized


def tool_names(tool_calls: Any) -> list[str]:
    names: list[str] = []
    if not isinstance(tool_calls, list):
        return names
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if isinstance(function, dict) and function.get("name") is not None:
            names.append(str(function["name"]))
        elif call.get("name") is not None:
            names.append(str(call["name"]))
    return names


def iter_llm_calls(
    path: Path,
    *,
    workflow_id: int,
    include_empty_assistant: bool,
    include_messages: bool,
) -> list[dict[str, Any]]:
    raw_messages = load_trajectory(path)
    prompt_so_far: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    assistant_step = 0

    for message_index, raw_message in enumerate(raw_messages):
        message = normalized_message(raw_message, message_index)
        if message["role"] == "assistant":
            output_text = message.get("text", "")
            calls = message.get("tool_calls") or []
            if output_text or calls or include_empty_assistant:
                prompt_text = format_prompt(prompt_so_far)
                output = format_assistant_output(output_text, calls)
                row = {
                    "workflow_id": workflow_id,
                    "step_id": assistant_step,
                    "message_index": message_index,
                    "prompt": prompt_text,
                    "output": output,
                    "tool_calls": calls,
                    "tool_names": tool_names(calls),
                    "has_tool_call": bool(calls),
                    "prompt_message_count": len(prompt_so_far),
                    "prompt_char_length": len(prompt_text),
                    "output_char_length": len(output),
                }
                if include_messages:
                    row["messages"] = list(prompt_so_far)
                rows.append(row)
            assistant_step += 1
        prompt_so_far.append(message)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Extract assistant-turn LLM calls from OpenHands SWE-bench "
            "trajectory JSON files."
        )
    )
    parser.add_argument(
        "trajectory_dir",
        type=Path,
        help="Directory containing OpenHands trajectory .json files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output JSONL path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of trajectory files to process.",
    )
    parser.add_argument(
        "--include-empty-assistant",
        action="store_true",
        help="Keep assistant messages with neither text nor tool calls.",
    )
    parser.add_argument(
        "--no-messages",
        action="store_true",
        help="Do not save normalized prompt messages. By default they are kept for role-faithful replay.",
    )
    args = parser.parse_args()

    trajectory_dir = args.trajectory_dir.expanduser().resolve()
    if not trajectory_dir.is_dir():
        raise SystemExit(f"trajectory_dir does not exist: {trajectory_dir}")

    files = sorted(trajectory_dir.glob("*.json"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"No .json trajectory files found in {trajectory_dir}")

    rows: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    for workflow_id, path in enumerate(files):
        try:
            workflow_rows = iter_llm_calls(
                path,
                workflow_id=workflow_id,
                include_empty_assistant=args.include_empty_assistant,
                include_messages=not args.no_messages,
            )
            for row in workflow_rows:
                rows.append({"call_id": len(rows), **row})
        except Exception as exc:
            failed.append({"path": str(path), "error": repr(exc)})

    output = args.output.expanduser().resolve()
    write_jsonl(output, rows)

    print(f"trajectory_dir: {trajectory_dir}")
    print(f"files: {len(files)}")
    print(f"llm_calls: {len(rows)}")
    print(f"failed_files: {len(failed)}")
    print(f"output: {output}")
    if failed:
        failed_path = output.with_suffix(output.suffix + ".failed.json")
        failed_path.write_text(
            json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"failed_output: {failed_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
