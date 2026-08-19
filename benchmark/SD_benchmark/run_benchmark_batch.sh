#!/usr/bin/env bash
set -euo pipefail

# Batch driver for local non-workflow benchmarks against SGLang.
#
# This script can queue multiple benchmarks in one invocation, but it runs them
# sequentially and restarts SGLang for every benchmark/variant/concurrency run.
#
# Outputs:
#   ${BATCH_DIR}/batch_results.csv
#   ${BATCH_DIR}/<variant>_c<concurrency>/turn_traces.jsonl
#   ${BATCH_DIR}/<variant>_c<concurrency>/summary.json
#   ${BATCH_DIR}/<variant>_c<concurrency>/server_info.json
#   ${BATCH_DIR}/<variant>_c<concurrency>/server.log
#   ${BATCH_DIR}/<variant>_c<concurrency>/gpu.csv

usage() {
  cat <<'EOF'
Usage:
  BENCHMARKS="mt_bench spec_bench HumanEval" VARIANTS="baseline ngram_d8" \
  CONCURRENCIES="1 4 16" bash SD_benchmark/run_benchmark_batch.sh

Runs one SGLang server per benchmark/variant/concurrency point. Each point is
fresh, so server-side NGRAM history does not carry across comparisons.

Key environment variables:
  BENCHMARKS       Space- or comma-separated: HumanEval, mt_bench, spec_bench,
                   swe_bench. Default: mt_bench.
  VARIANTS         baseline, ngram_d<N>, ngram_prob_d<N>, or ngram_backmatch_d<N>.
  CONCURRENCIES    Client concurrency values. Default: "1 4 16 32 64 100".
  MODEL_PATH       Target model path.
  SGLANG_DIR       SGLang checkout. SGLANG_ENV selects the conda environment.
  BENCH_ENV        Optional separate client environment; defaults to SGLANG_ENV.
  AS_DIR           Optional project-root override; derived from this script by default.
  MAX_TOKENS       Per-request completion limit. Default: 1024.
  MAX_TOTAL_TOKENS Server context/KV-cache budget when supported by SGLang.
  BATCH_ROOT       Parent output directory. Default: SD_benchmark/outputs/benchmark_sglang_batch.

For swe_bench, also set SWE_TRACE_JSONL, LIMIT (workflows), and optionally
MAX_STEPS_PER_WORKFLOW. Results include turn_traces.jsonl, summary.json,
server.log, gpu.csv, and batch_results.csv.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_ENV="${SGLANG_ENV:-sglang-v0514}"
# The request runner only needs the client dependencies already installed with
# SGLang. Override this only when a separate runner environment is necessary.
BENCH_ENV="${BENCH_ENV:-${SGLANG_ENV}}"

# Keep AS_DIR as an override for copied layouts, but derive the repository root
# from this script by default so callers do not need to set it.
AS_DIR="${AS_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
SGLANG_DIR="${SGLANG_DIR:-/mnt/d/code/sglang}"
MODEL_PATH="${MODEL_PATH:-/mnt/d/code/Qwen2.5-7B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-7B-Instruct-AWQ}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1919}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:${PORT}}"
API_BASE="${API_BASE:-${SERVER_URL}/v1}"
API_KEY="${API_KEY:-dummy}"
READY_PATHS="${READY_PATHS:-/model_info /v1/models}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-900}"
SERVER_STOP_TIMEOUT="${SERVER_STOP_TIMEOUT:-120}"
START_SERVER="${START_SERVER:-1}"
FORCE_KILL_STALE_SERVER="${FORCE_KILL_STALE_SERVER:-1}"

BENCHMARK="${BENCHMARK:-mt_bench}"
BENCHMARKS_STR="${BENCHMARKS:-${BENCHMARK}}"
BENCHMARKS_STR="${BENCHMARKS_STR//,/ }"
read -r -a BENCHMARKS_ARR <<< "${BENCHMARKS_STR}"
BENCHMARK_ROOT="${BENCHMARK_ROOT:-${AS_DIR}/SD_benchmark}"
SWE_TRACE_JSONL="${SWE_TRACE_JSONL:-${AS_DIR}/SD_benchmark/outputs/swe_bench/openhands_llm_calls.jsonl}"
SWE_TOOL_MODE="${SWE_TOOL_MODE:-api}"
SWE_TOOL_SCHEMA="${SWE_TOOL_SCHEMA:-${AS_DIR}/SD_benchmark/swe_bench/tool_definitions/tools_schema.json}"
SWE_TOOL_CHOICE="${SWE_TOOL_CHOICE:-required}"
SWE_TOOL_CALL_PARSER="${SWE_TOOL_CALL_PARSER:-qwen}"
SWE_TOOL_PROMPT="${SWE_TOOL_PROMPT:-${AS_DIR}/SD_benchmark/swe_bench/tool_definitions/openhands_tools_plain_prompt.md}"
SWE_FAILURE_LIST_OUTPUT="${SWE_FAILURE_LIST_OUTPUT:-}"
SWE_SKIP_FAILURE_LIST="${SWE_SKIP_FAILURE_LIST:-}"
CATEGORIES="${CATEGORIES:-}"
LIMIT="${LIMIT:-}"
MAX_STEPS_PER_WORKFLOW="${MAX_STEPS_PER_WORKFLOW:-}"
HUMANEVAL_STYLE="${HUMANEVAL_STYLE:-completion_instruction}"

# Deterministic-ish defaults for workload comparison.
TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TIMEOUT="${TIMEOUT:-600}"
EXTRA_BODY="${EXTRA_BODY:-}"
PRINT_EVERY="${PRINT_EVERY:-10}"

MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-}"
AUTO_MAX_RUNNING_REQUESTS="${AUTO_MAX_RUNNING_REQUESTS:-0}"
AUTO_MAX_RUNNING_THRESHOLD="${AUTO_MAX_RUNNING_THRESHOLD:-48}"
EXTRA_SERVER_ARGS="${EXTRA_SERVER_ARGS:-}"

FORCE_GREEDY_VERIFY="${FORCE_GREEDY_VERIFY:-True}"
SPEC_METRICS_LOG_ON_SHUTDOWN="${SPEC_METRICS_LOG_ON_SHUTDOWN:-1}"
NGRAM_MATCH_WINDOW="${NGRAM_MATCH_WINDOW:-2}"
NGRAM_BFS_BREADTH="${NGRAM_BFS_BREADTH:-1}"
NGRAM_CAPACITY="${NGRAM_CAPACITY:-500000}"
NGRAM_BACKMATCH_CONTEXT_TOKENS="${NGRAM_BACKMATCH_CONTEXT_TOKENS:-32}"
NGRAM_BACKMATCH_MAX_CANDIDATES="${NGRAM_BACKMATCH_MAX_CANDIDATES:-16}"
NGRAM_HYBRID_SCAN_OCCURRENCES="${NGRAM_HYBRID_SCAN_OCCURRENCES:-0}"
NGRAM_SCOPE_BY_EXTRA_KEY="${NGRAM_SCOPE_BY_EXTRA_KEY:-0}"
NGRAM_DRAFT_TOKENS="${NGRAM_DRAFT_TOKENS:-8}"

CONCURRENCIES_STR="${CONCURRENCIES:-1 4 16 32 64 100}"
read -r -a CONCURRENCIES_ARR <<< "${CONCURRENCIES_STR}"

# Supported variants:
#   baseline
#   ngram_d<N> / ngram_bfs_d<N>
#   ngram_prob_d<N> / prob_d<N>
#   ngram_backmatch_d<N> / backmatch_d<N>
# For variants without an explicit d<N>, NGRAM_DRAFT_TOKENS is used.
VARIANTS_STR="${VARIANTS:-baseline ngram_d8}"
read -r -a VARIANTS_ARR <<< "${VARIANTS_STR}"

BATCH_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_ROOT="${BATCH_ROOT:-${AS_DIR}/SD_benchmark/outputs/benchmark_sglang_batch}"
BATCH_DIR_OVERRIDE="${BATCH_DIR:-}"
RESULTS_CSV_OVERRIDE="${RESULTS_CSV:-}"
BATCH_DIR=""
RESULTS_CSV=""
CURRENT_BENCHMARK=""

SERVER_PID=""
GPU_SAMPLER_PID=""

run_in_env() {
  local env_name="$1"
  shift
  conda run --no-capture-output -n "${env_name}" "$@"
}

validate_benchmarks() {
  if (( ${#BENCHMARKS_ARR[@]} == 0 )); then
    echo "ERROR: no benchmark selected" >&2
    exit 2
  fi
  for benchmark in "${BENCHMARKS_ARR[@]}"; do
    case "${benchmark}" in
      HumanEval|human_eval|humaneval|mt_bench|mtbench|spec_bench|specbench|swe_bench|swebench)
        ;;
      *)
        echo "ERROR: unsupported benchmark=${benchmark}" >&2
        exit 2
        ;;
    esac
  done
}

wait_for_server_down() {
  local start_ts
  local warned=0
  start_ts="$(date +%s)"
  while true; do
    local ready=0
    for ready_path in ${READY_PATHS}; do
      local code
      code="$(
        curl -sS -o /dev/null -w "%{http_code}" \
          -H "Authorization: Bearer ${API_KEY}" \
          "${SERVER_URL}${ready_path}" 2>/dev/null || true
      )"
      if [[ "${code}" == "200" ]]; then
        ready=1
        break
      fi
    done
    if [[ "${ready}" == "0" ]]; then
      return 0
    fi
    if [[ "${warned}" == "0" ]]; then
      echo "[bench-batch] waiting for existing server at ${SERVER_URL} to stop"
      warned=1
    fi
    if (( $(date +%s) - start_ts > SERVER_STOP_TIMEOUT )); then
      echo "[bench-batch] server still responds at ${SERVER_URL}" >&2
      return 1
    fi
    sleep 1
  done
}

matching_sglang_pids() {
  ps -eo pid=,pgid=,args= | awk -v port="${PORT}" '
    $0 ~ /python .*sglang.launch_server/ && $0 ~ "--port " port {
      print $1 ":" $2
    }
  '
}

cleanup_orphan_sglang() {
  if [[ "${FORCE_KILL_STALE_SERVER}" != "1" ]]; then
    return 0
  fi

  local matches=()
  mapfile -t matches < <(matching_sglang_pids || true)
  if (( ${#matches[@]} == 0 )); then
    return 0
  fi

  echo "[bench-batch] cleaning stale SGLang process(es) on port ${PORT}: ${matches[*]}"
  local item pid pgid
  for item in "${matches[@]}"; do
    pid="${item%%:*}"
    pgid="${item#*:}"
    kill -TERM "${pid}" 2>/dev/null || true
  done

  for _ in $(seq 1 20); do
    mapfile -t matches < <(matching_sglang_pids || true)
    if (( ${#matches[@]} == 0 )); then
      return 0
    fi
    sleep 1
  done

  echo "[bench-batch] force cleaning stale SGLang process(es) on port ${PORT}: ${matches[*]}"
  for item in "${matches[@]}"; do
    pid="${item%%:*}"
    pgid="${item#*:}"
    kill -KILL "${pid}" 2>/dev/null || true
  done
}

cleanup_server() {
  local stopped=0
  if [[ -n "${SERVER_PID}" ]]; then
    if kill -0 "${SERVER_PID}" 2>/dev/null; then
      stopped=1
      echo "[bench-batch] stopping server process group pid=${SERVER_PID}"
      kill -TERM "-${SERVER_PID}" 2>/dev/null || kill "${SERVER_PID}" 2>/dev/null || true
      for _ in $(seq 1 30); do
        if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
          break
        fi
        sleep 1
      done
      if kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -KILL "-${SERVER_PID}" 2>/dev/null || kill -KILL "${SERVER_PID}" 2>/dev/null || true
      fi
      wait "${SERVER_PID}" 2>/dev/null || true
    fi
  fi
  SERVER_PID=""
  if [[ "${stopped}" == "1" ]]; then
    if ! wait_for_server_down; then
      cleanup_orphan_sglang
      wait_for_server_down
    fi
  fi
}

cleanup_gpu_sampler() {
  if [[ -n "${GPU_SAMPLER_PID}" ]]; then
    if kill -0 "${GPU_SAMPLER_PID}" 2>/dev/null; then
      kill "${GPU_SAMPLER_PID}" 2>/dev/null || true
      wait "${GPU_SAMPLER_PID}" 2>/dev/null || true
    fi
  fi
  GPU_SAMPLER_PID=""
}

cleanup_all() {
  cleanup_gpu_sampler
  cleanup_server
}

trap cleanup_all EXIT

wait_for_server() {
  local log_file="$1"
  local start_ts
  start_ts="$(date +%s)"
  echo "[bench-batch] waiting for server ready at ${SERVER_URL}"
  while true; do
    for ready_path in ${READY_PATHS}; do
      local code
      code="$(
        curl -sS -o /dev/null -w "%{http_code}" \
          -H "Authorization: Bearer ${API_KEY}" \
          "${SERVER_URL}${ready_path}" 2>/dev/null || true
      )"
      if [[ "${code}" == "200" ]]; then
        echo "[bench-batch] server ready via ${ready_path}"
        return 0
      fi
    done
    if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "[bench-batch] server exited before ready; tail log:" >&2
      tail -80 "${log_file}" >&2 || true
      return 1
    fi
    if (( $(date +%s) - start_ts > SERVER_READY_TIMEOUT )); then
      echo "[bench-batch] timed out waiting for server; tail log:" >&2
      tail -120 "${log_file}" >&2 || true
      return 1
    fi
    sleep 2
  done
}

variant_spec_args() {
  local variant="$1"
  local draft="${NGRAM_DRAFT_TOKENS}"
  local match_type="BFS"
  case "${variant}" in
    baseline)
      return 0
      ;;
    ngram|ngram_bfs|ngram_cold)
      match_type="BFS"
      ;;
    ngram_d*|ngram_bfs_d*|ngram_cold_d*)
      draft="${variant##*_d}"
      match_type="BFS"
      ;;
    ngram_prob|prob)
      match_type="PROB"
      ;;
    ngram_prob_d*|prob_d*)
      draft="${variant##*_d}"
      match_type="PROB"
      ;;
    ngram_backmatch|backmatch)
      match_type="BACKMATCH"
      ;;
    ngram_backmatch_d*|backmatch_d*)
      draft="${variant##*_d}"
      match_type="BACKMATCH"
      ;;
    *)
      echo "ERROR: unsupported VARIANT=${variant}" >&2
      exit 2
      ;;
  esac
  if ! [[ "${draft}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: could not parse draft length for VARIANT=${variant}; got draft=${draft}" >&2
    exit 2
  fi

  echo \
    "--speculative-algorithm NGRAM" \
    "--speculative-ngram-min-match-window-size ${NGRAM_MATCH_WINDOW}" \
    "--speculative-ngram-max-match-window-size ${NGRAM_MATCH_WINDOW}" \
    "--speculative-ngram-min-bfs-breadth ${NGRAM_BFS_BREADTH}" \
    "--speculative-ngram-max-bfs-breadth ${NGRAM_BFS_BREADTH}" \
    "--speculative-ngram-match-type ${match_type}" \
    "--speculative-num-draft-tokens ${draft}" \
    "--speculative-ngram-capacity ${NGRAM_CAPACITY}" \
    "--speculative-ngram-backmatch-context-tokens ${NGRAM_BACKMATCH_CONTEXT_TOKENS}" \
    "--speculative-ngram-backmatch-max-candidates ${NGRAM_BACKMATCH_MAX_CANDIDATES}" \
    "--speculative-ngram-hybrid-scan-occurrences ${NGRAM_HYBRID_SCAN_OCCURRENCES}"
}

start_server() {
  local variant="$1"
  local concurrency="$2"
  local run_dir="$3"
  local log_file="${run_dir}/server.log"
  local spec_args
  spec_args="$(variant_spec_args "${variant}")"
  local spec_args_arr=()
  if [[ -n "${spec_args}" ]]; then
    read -r -a spec_args_arr <<< "${spec_args}"
  fi

  local max_running_args=()
  if [[ -n "${MAX_RUNNING_REQUESTS}" ]]; then
    max_running_args+=(--max-running-requests "${MAX_RUNNING_REQUESTS}")
  elif [[ "${AUTO_MAX_RUNNING_REQUESTS}" == "1" && "${concurrency}" =~ ^[0-9]+$ && "${concurrency}" -gt "${AUTO_MAX_RUNNING_THRESHOLD}" ]]; then
    max_running_args+=(--max-running-requests "${concurrency}")
  fi

  local max_total_args=()
  if [[ -n "${MAX_TOTAL_TOKENS}" ]]; then
    max_total_args+=(--max-total-tokens "${MAX_TOTAL_TOKENS}")
  fi

  local spec_metrics_args=()
  if [[ "${SPEC_METRICS_LOG_ON_SHUTDOWN}" == "1" ]]; then
    spec_metrics_args+=(--spec-metrics-log-on-shutdown)
  fi

  local scope_args=()
  if [[ "${NGRAM_SCOPE_BY_EXTRA_KEY}" == "1" ]]; then
    scope_args+=(--speculative-ngram-scope-by-extra-key)
  fi
  local tool_call_parser_args=()
  if [[ "${CURRENT_BENCHMARK}" == "swe_bench" || "${CURRENT_BENCHMARK}" == "swebench" ]]; then
    if [[ "${SWE_TOOL_MODE}" == "api" && -n "${SWE_TOOL_CALL_PARSER}" ]]; then
      tool_call_parser_args+=(--tool-call-parser "${SWE_TOOL_CALL_PARSER}")
    fi
  fi
  local extra_server_args_arr=()
  if [[ -n "${EXTRA_SERVER_ARGS}" ]]; then
    read -r -a extra_server_args_arr <<< "${EXTRA_SERVER_ARGS}"
  fi

  if ! wait_for_server_down; then
    cleanup_orphan_sglang
    wait_for_server_down
  fi
  echo "[bench-batch] starting server variant=${variant} c=${concurrency}"
  setsid bash -c '
    set -euo pipefail
    SGLANG_DIR="$1"
    SGLANG_ENV="$2"
    MODEL_PATH="$3"
    HOST="$4"
    PORT="$5"
    SERVED_MODEL_NAME="$6"
    API_KEY="$7"
    FORCE_GREEDY_VERIFY="$8"
    shift 8
    cd "${SGLANG_DIR}"
    export SGLANG_NGRAM_FORCE_GREEDY_VERIFY="${FORCE_GREEDY_VERIFY}"
    exec conda run --no-capture-output -n "${SGLANG_ENV}" python -m sglang.launch_server \
      --model-path "${MODEL_PATH}" \
      --host "${HOST}" \
      --port "${PORT}" \
      --served-model-name "${SERVED_MODEL_NAME}" \
      --api-key "${API_KEY}" \
      "$@"
  ' _ "${SGLANG_DIR}" "${SGLANG_ENV}" "${MODEL_PATH}" "${HOST}" "${PORT}" \
      "${SERVED_MODEL_NAME}" "${API_KEY}" "${FORCE_GREEDY_VERIFY}" \
      "${max_total_args[@]}" "${max_running_args[@]}" "${spec_args_arr[@]}" \
      "${scope_args[@]}" "${spec_metrics_args[@]}" "${tool_call_parser_args[@]}" "${extra_server_args_arr[@]}" \
      >"${log_file}" 2>&1 &
  SERVER_PID="$!"
  echo "[bench-batch] server pid=${SERVER_PID} log=${log_file}"
  wait_for_server "${log_file}"
}

start_gpu_sampler() {
  local output="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  (
    echo "timestamp,gpu_util_percent,mem_used_mb,mem_total_mb"
    while true; do
      nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null || true
      sleep 1
    done
  ) >"${output}" &
  GPU_SAMPLER_PID="$!"
}

save_server_info() {
  local output="$1"
  local tmp="${output}.tmp"
  if curl -fsS -H "Authorization: Bearer ${API_KEY}" "${SERVER_URL}/server_info" \
    >"${tmp}" 2>/dev/null; then
    mv "${tmp}" "${output}"
    return 0
  fi
  curl -sS -H "Authorization: Bearer ${API_KEY}" "${SERVER_URL}/model_info" \
    >"${tmp}" 2>/dev/null || true
  mv "${tmp}" "${output}" 2>/dev/null || true
}

append_result_row() {
  local summary_path="$1"
  local variant="$2"
  local concurrency="$3"
  local server_log="$4"
  local server_info="$5"
  local gpu_log="$6"

  run_in_env "${BENCH_ENV}" python - "${summary_path}" "${RESULTS_CSV}" \
    "${variant}" "${concurrency}" "${server_log}" "${server_info}" "${gpu_log}" \
    "${NGRAM_DRAFT_TOKENS}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path, csv_path, variant, concurrency, server_log, server_info, gpu_log, default_draft = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
fields = [
    "benchmark",
    "variant",
    "max_concurrency",
    "spec_algorithm",
    "ngram_match_type",
    "ngram_draft_tokens",
    "model",
    "temperature",
    "top_p",
    "seed",
    "max_tokens",
    "tool_mode",
    "tool_choice",
    "tool_count",
    "items",
    "turns",
    "errors",
    "expected_tool_call_turns",
    "generated_tool_call_turns",
    "generated_tool_calls_total",
    "generated_tool_name_match_turns",
    "wall_time_s",
    "requests_s",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "completion_tokens_s",
    "total_tokens_s",
    "latency_avg_s",
    "latency_max_s",
    "latency_p50_s",
    "latency_p90_s",
    "latency_p99_s",
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
    "gpu_util_avg",
    "gpu_util_max",
    "gpu_mem_used_max_mb",
    "trace_output",
    "output_dir",
    "server_log",
    "server_info_path",
    "gpu_log",
]
row = {field: summary.get(field, "") for field in fields}
row["variant"] = variant
row["max_concurrency"] = concurrency
row["output_dir"] = str(Path(summary_path).parent)
row["server_log"] = server_log
row["server_info_path"] = server_info
row["gpu_log"] = gpu_log

def parse_variant(value, default_draft):
    if value == "baseline":
        return "", "", ""
    match_type = "BFS"
    if "backmatch" in value:
        match_type = "BACKMATCH"
    elif "prob" in value:
        match_type = "PROB"
    draft = ""
    marker = "_d"
    if marker in value:
        tail = value.rsplit(marker, 1)[-1]
        if tail.isdigit():
            draft = tail
    if not draft:
        draft = default_draft
    return "NGRAM", match_type, draft

row["spec_algorithm"], row["ngram_match_type"], row["ngram_draft_tokens"] = parse_variant(
    variant, default_draft
)

def read_json(path_text):
    if not path_text:
        return {}
    path = Path(path_text)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return {}

def first_internal_state(server_info_obj):
    states = server_info_obj.get("internal_states") or []
    for state in states:
        if isinstance(state, dict):
            return state
    return server_info_obj

server_info_obj = read_json(server_info)
internal_state = first_internal_state(server_info_obj)
spec_summary = server_info_obj.get("spec_metrics_summary") or {}
if not spec_summary:
    spec_summary = internal_state.get("spec_metrics_summary") or {}
spec_counters = spec_summary.get("SpecMetrics") or {}

def metric_text(value):
    if value is None:
        return ""
    return str(value)

def log_key_value(key):
    path = Path(server_log)
    if not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    import re
    matches = re.findall(rf"(?m)^{re.escape(key)}=([^\s]+)", text)
    return matches[-1] if matches else ""

def counter_value(key):
    value = metric_text(spec_counters.get(key, ""))
    return value or log_key_value(key)

lifetime_accept_len = metric_text(
    internal_state.get("avg_spec_accept_length", server_info_obj.get("avg_spec_accept_length", ""))
)
draft_tokens_value = internal_state.get(
    "speculative_num_draft_tokens",
    server_info_obj.get("speculative_num_draft_tokens", row["ngram_draft_tokens"]),
)
lifetime_accept_rate = ""
try:
    if lifetime_accept_len != "" and float(draft_tokens_value) > 0:
        lifetime_accept_rate = str(float(lifetime_accept_len) / float(draft_tokens_value))
except (TypeError, ValueError):
    pass

row["accept_len_mean"] = lifetime_accept_len
row["spec_draft_attempts_total"] = counter_value("draft_attempts_total")
row["spec_draft_tokens_total"] = counter_value("draft_tokens_total")
row["spec_verified_tokens_total"] = counter_value("verified_tokens_total")
row["spec_accepted_tokens_total"] = counter_value("accepted_tokens_total")
row["spec_rejected_tokens_total"] = counter_value("rejected_tokens_total")
row["spec_true_accept_rate"] = counter_value("acceptance_rate")
row["spec_true_mean_accept_len"] = counter_value("mean_accepted_tokens")
row["spec_zero_accept_ratio"] = counter_value("zero_accept_ratio")
row["spec_lifetime_accept_len"] = lifetime_accept_len
row["spec_lifetime_accept_rate"] = lifetime_accept_rate

def summarize_gpu(path_text):
    path = Path(path_text)
    if not path.is_file():
        return {}
    gpu_utils = []
    mem_used = []
    try:
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                try:
                    gpu_utils.append(float((rec.get("gpu_util_percent") or "").strip()))
                except ValueError:
                    pass
                try:
                    mem_used.append(float((rec.get("mem_used_mb") or "").strip()))
                except ValueError:
                    pass
    except Exception:
        return {}
    out = {}
    if gpu_utils:
        out["gpu_util_avg"] = round(sum(gpu_utils) / len(gpu_utils), 6)
        out["gpu_util_max"] = round(max(gpu_utils), 6)
    if mem_used:
        out["gpu_mem_used_max_mb"] = round(max(mem_used), 6)
    return out

row.update(summarize_gpu(gpu_log))

summary.update(
    {
        "variant": variant,
        "max_concurrency": concurrency,
        "spec_algorithm": row["spec_algorithm"],
        "ngram_match_type": row["ngram_match_type"],
        "ngram_draft_tokens": row["ngram_draft_tokens"],
        "tool_mode": row.get("tool_mode", ""),
        "tool_choice": row.get("tool_choice", ""),
        "tool_count": row.get("tool_count", ""),
        "expected_tool_call_turns": row.get("expected_tool_call_turns", ""),
        "generated_tool_call_turns": row.get("generated_tool_call_turns", ""),
        "generated_tool_calls_total": row.get("generated_tool_calls_total", ""),
        "generated_tool_name_match_turns": row.get("generated_tool_name_match_turns", ""),
        "accept_len_mean": row["accept_len_mean"],
        "spec_draft_attempts_total": row["spec_draft_attempts_total"],
        "spec_draft_tokens_total": row["spec_draft_tokens_total"],
        "spec_verified_tokens_total": row["spec_verified_tokens_total"],
        "spec_accepted_tokens_total": row["spec_accepted_tokens_total"],
        "spec_rejected_tokens_total": row["spec_rejected_tokens_total"],
        "spec_true_accept_rate": row["spec_true_accept_rate"],
        "spec_true_mean_accept_len": row["spec_true_mean_accept_len"],
        "spec_zero_accept_ratio": row["spec_zero_accept_ratio"],
        "spec_lifetime_accept_len": row["spec_lifetime_accept_len"],
        "spec_lifetime_accept_rate": row["spec_lifetime_accept_rate"],
        "gpu_util_avg": row.get("gpu_util_avg", ""),
        "gpu_util_max": row.get("gpu_util_max", ""),
        "gpu_mem_used_max_mb": row.get("gpu_mem_used_max_mb", ""),
        "server_log": server_log,
        "server_info_path": server_info,
        "gpu_log": gpu_log,
    }
)
Path(summary_path).write_text(
    json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
)

path = Path(csv_path)
path.parent.mkdir(parents=True, exist_ok=True)
write_header = not path.exists()
with path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    if write_header:
        writer.writeheader()
    writer.writerow(row)
PY
}

run_one() {
  local variant="$1"
  local concurrency="$2"
  local run_dir="${BATCH_DIR}/${variant}_c${concurrency}"
  local server_info="${run_dir}/server_info.json"
  local gpu_log="${run_dir}/gpu.csv"
  local server_log="${run_dir}/server.log"
  mkdir -p "${run_dir}"

  cleanup_gpu_sampler
  cleanup_server

  if [[ "${START_SERVER}" == "1" ]]; then
    start_server "${variant}" "${concurrency}" "${run_dir}"
    save_server_info "${server_info}"
  else
    server_log=""
    save_server_info "${server_info}"
  fi

  start_gpu_sampler "${gpu_log}"

  local cmd=()
  case "${CURRENT_BENCHMARK}" in
    swe_bench|swebench)
      cmd=(
        python "${AS_DIR}/SD_benchmark/swe_bench/run_swe_trace.py"
        --trace-jsonl "${SWE_TRACE_JSONL}"
        --server-url "${API_BASE}"
        --model "${SERVED_MODEL_NAME}"
        --api-key "${API_KEY}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --seed "${SEED}"
        --max-tokens "${MAX_TOKENS}"
        --concurrency "${concurrency}"
        --timeout "${TIMEOUT}"
        --output-dir "${run_dir}"
        --server-info-output "${server_info}"
        --print-every "${PRINT_EVERY}"
      )
      if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
      fi
      if [[ -n "${MAX_STEPS_PER_WORKFLOW}" ]]; then
        cmd+=(--max-steps-per-workflow "${MAX_STEPS_PER_WORKFLOW}")
      fi
      cmd+=(--tool-mode "${SWE_TOOL_MODE}")
      if [[ "${SWE_TOOL_MODE}" == "api" ]]; then
        cmd+=(--tool-schema "${SWE_TOOL_SCHEMA}" --tool-choice "${SWE_TOOL_CHOICE}")
      elif [[ "${SWE_TOOL_MODE}" == "plain" ]]; then
        cmd+=(--tool-prompt "${SWE_TOOL_PROMPT}")
      fi
      if [[ -n "${SWE_FAILURE_LIST_OUTPUT}" ]]; then
        cmd+=(--failure-list-output "${SWE_FAILURE_LIST_OUTPUT}")
      fi
      if [[ -n "${SWE_SKIP_FAILURE_LIST}" ]]; then
        cmd+=(--skip-failure-list "${SWE_SKIP_FAILURE_LIST}")
      fi
      if [[ -n "${EXTRA_BODY}" ]]; then
        cmd+=(--extra-body "${EXTRA_BODY}")
      fi
      ;;
    *)
      cmd=(
        python "${AS_DIR}/SD_benchmark/run_benchmark.py"
        --benchmark "${CURRENT_BENCHMARK}"
        --benchmark-root "${BENCHMARK_ROOT}"
        --server-url "${API_BASE}"
        --model "${SERVED_MODEL_NAME}"
        --api-key "${API_KEY}"
        --temperature "${TEMPERATURE}"
        --top-p "${TOP_P}"
        --seed "${SEED}"
        --max-tokens "${MAX_TOKENS}"
        --concurrency "${concurrency}"
        --timeout "${TIMEOUT}"
        --humaneval-style "${HUMANEVAL_STYLE}"
        --output-dir "${run_dir}"
        --print-every "${PRINT_EVERY}"
      )
      if [[ -n "${CATEGORIES}" ]]; then
        cmd+=(--categories "${CATEGORIES}")
      fi
      if [[ -n "${LIMIT}" ]]; then
        cmd+=(--limit "${LIMIT}")
      fi
      if [[ -n "${EXTRA_BODY}" ]]; then
        cmd+=(--extra-body "${EXTRA_BODY}")
      fi
      ;;
  esac

  echo "[bench-batch] running benchmark=${CURRENT_BENCHMARK} variant=${variant} c=${concurrency}"
  run_in_env "${BENCH_ENV}" "${cmd[@]}" | tee "${run_dir}/replay.log"

  cleanup_gpu_sampler
  if [[ "${START_SERVER}" == "1" ]]; then
    save_server_info "${server_info}"
    cleanup_server
  fi

  append_result_row "${run_dir}/summary.json" "${variant}" "${concurrency}" \
    "${server_log}" "${server_info}" "${gpu_log}"
}

run_benchmark_batch() {
  local benchmark="$1"
  CURRENT_BENCHMARK="${benchmark}"
  if [[ -n "${BATCH_DIR_OVERRIDE}" && ${#BENCHMARKS_ARR[@]} -gt 1 ]]; then
    BATCH_DIR="${BATCH_DIR_OVERRIDE}/${benchmark}"
  elif [[ -n "${BATCH_DIR_OVERRIDE}" ]]; then
    BATCH_DIR="${BATCH_DIR_OVERRIDE}"
  else
    BATCH_DIR="${BATCH_ROOT}/${benchmark}/${BATCH_TS}"
  fi
  if [[ -n "${RESULTS_CSV_OVERRIDE}" && ${#BENCHMARKS_ARR[@]} -eq 1 ]]; then
    RESULTS_CSV="${RESULTS_CSV_OVERRIDE}"
  else
    RESULTS_CSV="${BATCH_DIR}/batch_results.csv"
  fi

  mkdir -p "${BATCH_DIR}"
  echo "[bench-batch] batch_dir=${BATCH_DIR}"
  echo "[bench-batch] benchmark=${CURRENT_BENCHMARK}"
  echo "[bench-batch] variants=${VARIANTS_STR}"
  echo "[bench-batch] concurrencies=${CONCURRENCIES_STR}"
  echo "[bench-batch] temperature=${TEMPERATURE} top_p=${TOP_P} seed=${SEED} max_tokens=${MAX_TOKENS}"

  for variant in "${VARIANTS_ARR[@]}"; do
    for concurrency in "${CONCURRENCIES_ARR[@]}"; do
      run_one "${variant}" "${concurrency}"
    done
  done

  run_in_env "${BENCH_ENV}" python "${AS_DIR}/SD_benchmark/analyze_benchmark_batch.py" "${RESULTS_CSV}" --csv-output "${BATCH_DIR}/perf_summary.csv" --json-output "${BATCH_DIR}/perf_summary.json"

  echo "[bench-batch] done: ${BATCH_DIR}"
  echo "[bench-batch] results: ${RESULTS_CSV}"
  echo "[bench-batch] perf_summary: ${BATCH_DIR}/perf_summary.csv"
}

main() {
  validate_benchmarks
  echo "[bench-batch] benchmarks=${BENCHMARKS_ARR[*]}"
  for benchmark in "${BENCHMARKS_ARR[@]}"; do
    run_benchmark_batch "${benchmark}"
  done
}

main "$@"
