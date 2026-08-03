"""Backfill SGLang speculative metrics into SD benchmark results.

The batch runner writes these fields for new runs. Use this script for older
batch directories whose `server_info.json` already contains `/server_info`
payloads with `avg_spec_accept_length` or `spec_metrics_summary`.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


SPEC_FIELDS = [
    "accept_len_mean",
    "spec_draft_attempts_total",
    "spec_draft_tokens_total",
    "spec_verified_tokens_total",
    "spec_accepted_tokens_total",
    "spec_rejected_tokens_total",
    "spec_true_accept_rate",
    "spec_true_mean_accept_len",
    "spec_zero_accept_ratio",
    "spec_lifetime_accept_len",
    "spec_lifetime_accept_rate",
]

SPEC_KEY_MAP = {
    "spec_draft_attempts_total": "draft_attempts_total",
    "spec_draft_tokens_total": "draft_tokens_total",
    "spec_verified_tokens_total": "verified_tokens_total",
    "spec_accepted_tokens_total": "accepted_tokens_total",
    "spec_rejected_tokens_total": "rejected_tokens_total",
    "spec_true_accept_rate": "acceptance_rate",
    "spec_true_mean_accept_len": "mean_accepted_tokens",
    "spec_zero_accept_ratio": "zero_accept_ratio",
}


def resolve_path(raw_path: str, *, csv_path: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    candidates = [path]

    text = raw_path.replace("\\", "/")
    match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if match:
        drive = match.group(1).upper()
        rest = match.group(2).replace("/", "\\")
        candidates.append(Path(f"{drive}:\\{rest}"))

    if not path.is_absolute():
        candidates.append((csv_path.parent / path).resolve())

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0] if candidates else None


def read_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
        return payload if isinstance(payload, dict) else {}
    except json.JSONDecodeError:
        return {}


def first_internal_state(server_info: dict[str, Any]) -> dict[str, Any]:
    states = server_info.get("internal_states") or []
    for state in states:
        if isinstance(state, dict):
            return state
    return server_info


def spec_counters(server_info: dict[str, Any]) -> dict[str, Any]:
    state = first_internal_state(server_info)
    summary = server_info.get("spec_metrics_summary") or state.get(
        "spec_metrics_summary"
    ) or {}
    spec = summary.get("SpecMetrics") or {}
    return spec if isinstance(spec, dict) else {}


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def read_log_metric(server_log_path: Path | None, key: str) -> str:
    if server_log_path is None or not server_log_path.is_file():
        return ""
    log_text = server_log_path.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(rf"(?m)^{re.escape(key)}=([^\s]+)", log_text)
    return matches[-1] if matches else ""


def lifetime_metrics(server_info: dict[str, Any], *, draft_tokens_hint: str) -> dict[str, str]:
    state = first_internal_state(server_info)
    accept_len = text(
        state.get("avg_spec_accept_length", server_info.get("avg_spec_accept_length", ""))
    )
    draft_tokens = state.get(
        "speculative_num_draft_tokens",
        server_info.get("speculative_num_draft_tokens", draft_tokens_hint),
    )
    accept_rate = ""
    try:
        if accept_len != "" and float(draft_tokens) > 0:
            accept_rate = str(float(accept_len) / float(draft_tokens))
    except (TypeError, ValueError):
        pass
    return {
        "accept_len_mean": accept_len,
        "spec_lifetime_accept_len": accept_len,
        "spec_lifetime_accept_rate": accept_rate,
    }


def backfill_row(row: dict[str, str], *, csv_path: Path) -> bool:
    server_info_path = resolve_path(row.get("server_info_path", ""), csv_path=csv_path)
    server_log_path = resolve_path(row.get("server_log", ""), csv_path=csv_path)
    server_info = read_json(server_info_path)
    counters = spec_counters(server_info)
    filled = False

    for field, source_key in SPEC_KEY_MAP.items():
        value = text(counters.get(source_key, ""))
        if not value:
            value = read_log_metric(server_log_path, source_key)
        row[field] = value
        filled = filled or bool(value)

    for field, value in lifetime_metrics(
        server_info, draft_tokens_hint=row.get("ngram_draft_tokens", "")
    ).items():
        row[field] = value
        filled = filled or bool(value)

    summary_path = resolve_path(
        row.get("output_dir", "") + "/summary.json" if row.get("output_dir") else "",
        csv_path=csv_path,
    )
    if summary_path and summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            for field in SPEC_FIELDS:
                summary[field] = row.get(field, "")
            summary_path.write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    return filled


def backfill_csv(input_path: Path, output_path: Path) -> int:
    with input_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    for field in SPEC_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)

    filled_rows = 0
    for row in rows:
        if backfill_row(row, csv_path=input_path):
            filled_rows += 1

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return filled_rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Backfill SGLang speculative metrics into SD_benchmark batch_results.csv."
    )
    parser.add_argument("batch_results", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="Output CSV path. Defaults to <input>.with_spec_metrics.csv.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input CSV after writing a .bak copy.",
    )
    args = parser.parse_args()

    input_path = args.batch_results.expanduser().resolve()
    if not input_path.is_file():
        raise SystemExit(f"batch_results.csv does not exist: {input_path}")
    if args.output and args.in_place:
        raise SystemExit("Use either --output or --in-place, not both.")

    output_path = (
        args.output.expanduser().resolve()
        if args.output
        else input_path.with_name(input_path.stem + ".with_spec_metrics.csv")
    )
    if args.in_place:
        backup_path = input_path.with_suffix(input_path.suffix + ".bak")
        backup_path.write_text(input_path.read_text(encoding="utf-8-sig"), encoding="utf-8")
        print(f"backup: {backup_path}")
        output_path = input_path

    filled_rows = backfill_csv(input_path, output_path)
    print(f"input: {input_path}")
    print(f"output: {output_path}")
    print(f"rows_with_spec_metrics: {filled_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
