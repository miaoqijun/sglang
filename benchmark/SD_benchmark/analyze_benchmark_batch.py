"""Summarize SD benchmark batch results with baseline-relative speedups."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def baseline_key(row: dict[str, str]) -> tuple[str, str]:
    return (row.get("benchmark", ""), row.get("max_concurrency", ""))


def summarize(rows: list[dict[str, str]], *, baseline_variant: str) -> list[dict[str, Any]]:
    baselines: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        if row.get("variant") == baseline_variant:
            baselines[baseline_key(row)] = row

    out: list[dict[str, Any]] = []
    for row in rows:
        base = baselines.get(baseline_key(row), {})
        wall = as_float(row.get("wall_time_s"))
        base_wall = as_float(base.get("wall_time_s"))
        tps = as_float(row.get("completion_tokens_s"))
        base_tps = as_float(base.get("completion_tokens_s"))
        total_tps = as_float(row.get("total_tokens_s"))
        base_total_tps = as_float(base.get("total_tokens_s"))
        out.append(
            {
                "benchmark": row.get("benchmark", ""),
                "variant": row.get("variant", ""),
                "max_concurrency": row.get("max_concurrency", ""),
                "wall_time_s": wall,
                "wall_time_speedup": round(base_wall / wall, 6)
                if wall and base_wall
                else "",
                "completion_tokens_s": tps,
                "completion_tps_speedup": round(tps / base_tps, 6)
                if tps and base_tps
                else "",
                "total_tokens_s": total_tps,
                "total_tps_speedup": round(total_tps / base_total_tps, 6)
                if total_tps and base_total_tps
                else "",
                "latency_p50_s": row.get("latency_p50_s", ""),
                "latency_p90_s": row.get("latency_p90_s", ""),
                "latency_p99_s": row.get("latency_p99_s", ""),
                "accept_len_mean": row.get("accept_len_mean", ""),
                "spec_true_mean_accept_len": row.get("spec_true_mean_accept_len", ""),
                "spec_true_accept_rate": row.get("spec_true_accept_rate", ""),
                "spec_zero_accept_ratio": row.get("spec_zero_accept_ratio", ""),
                "spec_draft_attempts_total": row.get("spec_draft_attempts_total", ""),
                "spec_accepted_tokens_total": row.get("spec_accepted_tokens_total", ""),
                "spec_lifetime_accept_len": row.get("spec_lifetime_accept_len", ""),
                "spec_lifetime_accept_rate": row.get("spec_lifetime_accept_rate", ""),
                "gpu_util_avg": row.get("gpu_util_avg", ""),
                "gpu_util_max": row.get("gpu_util_max", ""),
                "gpu_mem_used_max_mb": row.get("gpu_mem_used_max_mb", ""),
                "errors": row.get("errors", ""),
            }
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute baseline-relative wall-time and throughput speedups from batch_results.csv.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python SD_benchmark/analyze_benchmark_batch.py "
            "SD_benchmark/outputs/benchmark_sglang_batch/<run>/batch_results.csv\n\n"
            "Writes perf_summary.csv and perf_summary.json next to the input by default."
        ),
    )
    parser.add_argument("batch_results", type=Path, help="Batch CSV produced by a launcher.")
    parser.add_argument("--baseline-variant", default="baseline", help="Variant used as the speedup denominator.")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional perf-summary CSV path.")
    parser.add_argument("--json-output", type=Path, default=None, help="Optional perf-summary JSON path.")
    args = parser.parse_args()

    batch_results = args.batch_results.expanduser().resolve()
    rows = read_csv(batch_results)
    summary_rows = summarize(rows, baseline_variant=args.baseline_variant)

    csv_output = args.csv_output or batch_results.with_name("perf_summary.csv")
    json_output = args.json_output or batch_results.with_name("perf_summary.json")
    write_csv(csv_output, summary_rows)
    json_output.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"batch_results: {batch_results}")
    print(f"csv_output: {csv_output}")
    print(f"json_output: {json_output}")
    for row in summary_rows:
        print(
            f"- {row['benchmark']} c={row['max_concurrency']} {row['variant']} "
            f"wall_speedup={row['wall_time_speedup']} "
            f"tps_speedup={row['completion_tps_speedup']} "
            f"accept_len={row['accept_len_mean']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
