"""Build a teacher-forcing reference from a converted mini-swe-agent trace.

The selected workflow order, ``--limit``, ``--max-steps-per-workflow``, and
optional failure list intentionally match ``run_swe_trace.py``. The resulting
JSONL is a compact target-output reference; it is not another replay trace.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DEFAULT_TRACE = SCRIPT_DIR / "mini_swe_qwen25_coder_32b_50workflows.jsonl"
DEFAULT_OUTPUT = BENCHMARK_DIR / "outputs" / "swe_bench" / "mini_swe_teacher_forcing.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{line_no}")
            rows.append(value)
    return rows


def request_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return (
        str(row.get("workflow_id", "")),
        str(row.get("step_id", "")),
        str(row.get("call_id", "")),
    )


def group_workflows(
    rows: list[dict[str, Any]],
    *,
    limit_workflows: int | None,
    max_steps_per_workflow: int | None,
) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("workflow_id", "")), []).append(row)

    workflows: list[list[dict[str, Any]]] = []
    for workflow_id in sorted(
        grouped, key=lambda value: int(value) if value.isdigit() else value
    ):
        workflow = sorted(
            grouped[workflow_id],
            key=lambda row: (int(row.get("step_id") or 0), int(row.get("call_id") or 0)),
        )
        if max_steps_per_workflow is not None:
            workflow = workflow[:max_steps_per_workflow]
        if workflow:
            workflows.append(workflow)
        if limit_workflows is not None and len(workflows) >= limit_workflows:
            break
    return workflows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a mini-swe-agent replay trace into fixed target outputs "
            "for SGLang teacher-forced NGRAM replay."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python SD_benchmark/swe_bench/build_teacher_forcing_trace.py "
            "--trace-jsonl SD_benchmark/swe_bench/mini_swe_qwen25_coder_32b_50workflows.jsonl "
            "--output SD_benchmark/swe_bench/mini_swe_teacher_forcing.jsonl\n\n"
            "Use the same --limit, --max-steps-per-workflow, and --skip-failure-list "
            "selection in this builder and run_swe_trace.py when replaying a subset."
        ),
    )
    parser.add_argument("--trace-jsonl", type=Path, default=DEFAULT_TRACE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum selected workflows, matching run_swe_trace.py.",
    )
    parser.add_argument(
        "--max-steps-per-workflow",
        type=int,
        default=None,
        help="Maximum selected LLM calls per workflow, matching run_swe_trace.py.",
    )
    parser.add_argument(
        "--skip-failure-list",
        type=Path,
        default=None,
        help="Optional failure JSONL produced by run_swe_trace.py.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    if args.max_steps_per_workflow is not None and args.max_steps_per_workflow < 1:
        raise SystemExit("--max-steps-per-workflow must be >= 1")

    trace_path = args.trace_jsonl.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    workflows = group_workflows(
        read_jsonl(trace_path),
        limit_workflows=args.limit,
        max_steps_per_workflow=args.max_steps_per_workflow,
    )

    skipped = 0
    if args.skip_failure_list:
        failure_keys = {request_key(row) for row in read_jsonl(args.skip_failure_list)}
        filtered: list[list[dict[str, Any]]] = []
        for workflow in workflows:
            kept = [row for row in workflow if request_key(row) not in failure_keys]
            skipped += len(workflow) - len(kept)
            if kept:
                filtered.append(kept)
        workflows = filtered

    selected_rows = [row for workflow in workflows for row in workflow]
    if not selected_rows:
        raise SystemExit(f"No rows selected from {trace_path}")

    references: list[dict[str, Any]] = []
    seen_compat_keys: set[tuple[str, int]] = set()
    for row in selected_rows:
        workflow_id = str(row.get("workflow_id", ""))
        try:
            step_id = int(row["step_id"])
            call_id = int(row["call_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid workflow/step/call ID in {request_key(row)}") from exc
        output = str(row.get("output") or "")
        if not output:
            raise ValueError(f"Source output is empty for {request_key(row)}")

        # These fields are compatible with the generic teacher-forcing loader.
        # mini-swe-agent has one call per workflow step; reject ambiguity rather
        # than silently mapping two calls onto the same compatibility key.
        compat_key = (workflow_id, step_id)
        if compat_key in seen_compat_keys:
            raise ValueError(
                "Multiple calls share a workflow_id/step_id; use call_id-aware "
                "SWE teacher forcing instead."
            )
        seen_compat_keys.add(compat_key)
        references.append(
            {
                "question_id": workflow_id,
                "turn_id": step_id,
                "workflow_id": workflow_id,
                "step_id": step_id,
                "call_id": call_id,
                "output": output,
                "error": False,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in references:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"trace_path: {trace_path}")
    print(f"output: {output_path}")
    print(f"workflows: {len(workflows)}")
    print(f"calls: {len(references)}")
    print(f"skipped_failure_calls: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
