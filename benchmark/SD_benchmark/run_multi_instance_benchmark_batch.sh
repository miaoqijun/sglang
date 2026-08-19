#!/usr/bin/env bash
set -euo pipefail

# Batch launcher for the coauthor's multi-instance SGLang gateway experiment.
# Every benchmark / variant / concurrency run gets fresh workers, a fresh L2
# namespace, and a fresh gateway so history does not leak across comparisons.
#
# Supported variants:
#   baseline
#   ngram_d<N> / ngram_bfs_d<N>
#   ngram_prob_d<N> / prob_d<N>
# Append _no_l2 to any NGRAM variant to disable only cross-instance history,
# e.g. ngram_d8_no_l2 or ngram_prob_d4_no_l2.
#
# Run this script from an activated SGLang environment. It deliberately does
# not use the single-instance run_benchmark_batch.sh launcher.

usage() {
  cat <<'EOF'
Usage:
  MODEL_PATH=/path/to/model BENCHMARKS="mt_bench" \
  VARIANTS="baseline ngram_d8 ngram_d8_no_l2" CONCURRENCIES="1 4 16" \
  GPU_IDS="0 1" bash SD_benchmark/run_multi_instance_benchmark_batch.sh

Starts two SGLang workers and a round-robin sglang_router for each experiment
point. Workers and the gateway are restarted for every point.

Key environment variables:
  BENCHMARKS       HumanEval, mt_bench, spec_bench, or swe_bench.
  VARIANTS         baseline, ngram_d<N>, or ngram_prob_d<N>. Add _no_l2 to
                   disable only shared cross-instance NGRAM history.
  CONCURRENCIES    Client concurrency values. Default: "1 4".
  GPU_IDS          Exactly two GPU IDs, e.g. "0 1". "0 0" is smoke-test only.
  MODEL_PATH       Target model path.
  WORKER_MEM_FRACTION_STATIC / WORKER_MAX_TOTAL_TOKENS  Worker memory settings.
  BATCH_ROOT       Parent output directory.

The modified gateway must include patches/sgl_model_gateway_return_meta_info.patch
to report raw SGLang speculative counters. Results are written to BATCH_ROOT.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

AS_DIR="${AS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SGLANG_BIN="${SGLANG_BIN:-sglang}"
MODEL_PATH="${MODEL_PATH:-${HOME}/swq/models/Qwen2.5-14B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen2.5-14B-Instruct}"

BENCHMARKS_STR="${BENCHMARKS:-mt_bench}"
BENCHMARKS_STR="${BENCHMARKS_STR//,/ }"
read -r -a BENCHMARKS_ARR <<< "${BENCHMARKS_STR}"
VARIANTS_STR="${VARIANTS:-baseline ngram_d8}"
read -r -a VARIANTS_ARR <<< "${VARIANTS_STR}"
CONCURRENCIES_STR="${CONCURRENCIES:-1 4}"
read -r -a CONCURRENCIES_ARR <<< "${CONCURRENCIES_STR}"

# Conda compiler activation commonly exports HOST=x86_64-conda-linux-gnu.
# Do not reuse that variable for the HTTP listener address.
SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
GATEWAY_PORT="${GATEWAY_PORT:-1919}"
WORKER0_PORT="${WORKER0_PORT:-19191}"
WORKER1_PORT="${WORKER1_PORT:-19192}"
API_KEY="${API_KEY:-dummy}"
# Backward-compatible single-GPU default. Set GPU_IDS="0 1" to bind the
# two workers to different GPUs, or GPU_IDS="0 0" for a functional single-GPU
# two-worker setup.
GPU_ID="${GPU_ID:-0}"
GPU_IDS_STR="${GPU_IDS:-${GPU_ID} ${GPU_ID}}"
GPU_IDS_STR="${GPU_IDS_STR//,/ }"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

WORKER_MEM_FRACTION_STATIC="${WORKER_MEM_FRACTION_STATIC:-0.40}"
WORKER_MAX_TOTAL_TOKENS="${WORKER_MAX_TOTAL_TOKENS:-32768}"
WORKER_MAX_RUNNING_REQUESTS="${WORKER_MAX_RUNNING_REQUESTS:-}"
NGRAM_CAPACITY="${NGRAM_CAPACITY:-500000}"
NGRAM_MAX_TRIE_DEPTH="${NGRAM_MAX_TRIE_DEPTH:-18}"
NGRAM_BFS_BREADTH="${NGRAM_BFS_BREADTH:-1}"
L2_BACKEND="${L2_BACKEND:-mmap}"
L2_MMAP_CAPACITY="${L2_MMAP_CAPACITY:-65536}"

TEMPERATURE="${TEMPERATURE:-0}"
TOP_P="${TOP_P:-1.0}"
SEED="${SEED:-0}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
TIMEOUT="${TIMEOUT:-600}"
LIMIT="${LIMIT:-}"
CATEGORIES="${CATEGORIES:-}"
HUMANEVAL_STYLE="${HUMANEVAL_STYLE:-completion_instruction}"
PRINT_EVERY="${PRINT_EVERY:-10}"
SWE_TRACE_JSONL="${SWE_TRACE_JSONL:-${AS_DIR}/SD_benchmark/swe_bench/mini_swe_qwen25_coder_32b_50workflows.jsonl}"
SWE_TOOL_MODE="${SWE_TOOL_MODE:-none}"
MAX_STEPS_PER_WORKFLOW="${MAX_STEPS_PER_WORKFLOW:-}"
# Optional NGRAM teacher-forced replay. The trace must cover the selected
# benchmark items and be produced with a tokenizer compatible with MODEL_PATH.
TEACHER_FORCING_TRACE="${TEACHER_FORCING_TRACE:-}"
TEACHER_FORCING_TOKENIZER="${TEACHER_FORCING_TOKENIZER:-${MODEL_PATH}}"

# These defaults address the Ubuntu 20.04/CUDA JIT compiler combination used
# on the H200 node. Override them if the cluster provides a different compiler.
CC_BIN="${CC_BIN:-/usr/bin/gcc-10}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-10}"
CUDAHOSTCXX_BIN="${CUDAHOSTCXX_BIN:-${CXX_BIN}}"
NVCC_PREPEND_FLAGS_VALUE="${NVCC_PREPEND_FLAGS_VALUE:--ccbin ${CXX_BIN}}"

BATCH_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_ROOT="${BATCH_ROOT:-${AS_DIR}/SD_benchmark/outputs/multi_instance_gateway_batch}"
BATCH_DIR="${BATCH_DIR:-${BATCH_ROOT}/${BATCH_TS}}"
RESULTS_CSV="${RESULTS_CSV:-${BATCH_DIR}/batch_results.csv}"

WORKER0_PID=""
WORKER1_PID=""
GATEWAY_PID=""
GPU_SAMPLER_PID=""

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_path() {
  [[ -e "$1" ]] || die "Required path does not exist: $1"
}

wait_for_http() {
  local url="$1"
  local name="$2"
  local pid="$3"
  local timeout_s="${WORKER_READY_TIMEOUT:-900}"
  local started
  started="$(date +%s)"

  while true; do
    if curl -fsS -H "Authorization: Bearer ${API_KEY}" "$url" >/dev/null 2>&1; then
      echo "[multi-batch] ${name} ready: ${url}"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[multi-batch] ${name} exited before becoming ready" >&2
      return 1
    fi
    if (( $(date +%s) - started > timeout_s )); then
      echo "[multi-batch] timed out waiting for ${name}: ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

stop_pid_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
}

stop_services() {
  stop_pid_group "$GATEWAY_PID"
  stop_pid_group "$WORKER1_PID"
  stop_pid_group "$WORKER0_PID"
  GATEWAY_PID=""
  WORKER1_PID=""
  WORKER0_PID=""
}

stop_gpu_sampler() {
  [[ -n "$GPU_SAMPLER_PID" ]] || return 0
  kill "$GPU_SAMPLER_PID" 2>/dev/null || true
  wait "$GPU_SAMPLER_PID" 2>/dev/null || true
  GPU_SAMPLER_PID=""
}

cleanup_all() {
  stop_gpu_sampler
  stop_services
}
trap cleanup_all EXIT

start_gpu_sampler() {
  local output="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  (
    echo "timestamp,gpu_id,gpu_util_percent,mem_used_mb,mem_total_mb"
    while true; do
      local sampled=()
      local gpu seen
      for gpu in "${GPU_IDS[@]}"; do
        seen=0
        for previous in "${sampled[@]}"; do
          [[ "$previous" == "$gpu" ]] && seen=1 && break
        done
        (( seen == 1 )) && continue
        sampled+=("$gpu")
        nvidia-smi -i "$gpu" \
          --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
          --format=csv,noheader,nounits 2>/dev/null \
          | sed "s/^[[:space:]]*\([^,]*\),/\1,${gpu},/" || true
      done
      sleep 1
    done
  ) >"$output" &
  GPU_SAMPLER_PID="$!"
}

parse_variant() {
  local variant="$1"
  SPEC_ARGS=()
  USE_L2=0
  if [[ "$variant" == "baseline" ]]; then
    return 0
  fi

  if [[ "$variant" == *_no_l2 ]]; then
    # _no_l2 disables cross-worker reuse only; each worker still maintains its
    # ordinary in-process NGRAM cache for the duration of this point.
    variant="${variant%_no_l2}"
  else
    USE_L2=1
  fi

  local match_type
  local draft
  if [[ "$variant" =~ ^(ngram|ngram_bfs)_d([0-9]+)$ ]]; then
    match_type="BFS"
    draft="${BASH_REMATCH[2]}"
  elif [[ "$variant" =~ ^(ngram_prob|prob)_d([0-9]+)$ ]]; then
    match_type="PROB"
    draft="${BASH_REMATCH[2]}"
  else
    die "Unsupported variant=${variant}. Use baseline, ngram_d<N>, ngram_prob_d<N>, or append _no_l2."
  fi

  SPEC_ARGS=(
    --speculative-algorithm NGRAM
    --speculative-ngram-min-bfs-breadth "$NGRAM_BFS_BREADTH"
    --speculative-ngram-max-bfs-breadth "$NGRAM_BFS_BREADTH"
    --speculative-ngram-match-type "$match_type"
    --speculative-ngram-max-trie-depth "$NGRAM_MAX_TRIE_DEPTH"
    --speculative-num-draft-tokens "$draft"
    --speculative-ngram-capacity "$NGRAM_CAPACITY"
  )
}

start_worker() {
  local worker_id="$1"
  local port="$2"
  local gpu_id="$3"
  local l2_path="$4"
  local namespace="$5"
  local log_path="$6"
  local pid_var="$7"
  local args=(
    serve
    --model-path "$MODEL_PATH"
    --host "$SERVER_HOST"
    --port "$port"
    --served-model-name "$SERVED_MODEL_NAME"
    --api-key "$API_KEY"
    --mem-fraction-static "$WORKER_MEM_FRACTION_STATIC"
    --max-total-tokens "$WORKER_MAX_TOTAL_TOKENS"
  )
  if [[ -n "$WORKER_MAX_RUNNING_REQUESTS" ]]; then
    args+=(--max-running-requests "$WORKER_MAX_RUNNING_REQUESTS")
  fi
  if (( ${#SPEC_ARGS[@]} > 0 )); then
    args+=("${SPEC_ARGS[@]}")
  fi
  if (( USE_L2 == 1 )); then
    # Both workers share a namespace but have different instance IDs. The
    # per-point path and namespace prevent history leaking into later variants.
    args+=(
      --speculative-ngram-l2-history-path "$l2_path"
      --speculative-ngram-l2-backend "$L2_BACKEND"
      --speculative-ngram-l2-mmap-capacity "$L2_MMAP_CAPACITY"
      --speculative-ngram-l2-namespace "$namespace"
      --speculative-ngram-l2-instance-id "$worker_id"
    )
  fi

  echo "[multi-batch] starting ${worker_id} on port ${port}, GPU ${gpu_id}"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    CC="$CC_BIN" \
    CXX="$CXX_BIN" \
    CUDAHOSTCXX="$CUDAHOSTCXX_BIN" \
    NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS_VALUE" \
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY=True \
    "$SGLANG_BIN" "${args[@]}" >"$log_path" 2>&1 &
  printf -v "$pid_var" '%s' "$!"
}

save_server_info() {
  local url="$1"
  local output="$2"
  curl -fsS -H "Authorization: Bearer ${API_KEY}" "${url}/server_info" >"$output" 2>/dev/null || true
}

append_result_row() {
  local summary_path="$1"
  local benchmark="$2"
  local variant="$3"
  local concurrency="$4"
  local run_dir="$5"

  "$PYTHON_BIN" - "$summary_path" "$RESULTS_CSV" "$benchmark" "$variant" "$concurrency" "$run_dir" "${GPU_IDS[*]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path, csv_path, benchmark, variant, concurrency, run_dir, worker_gpu_ids = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))

def gpu_metrics():
    path = Path(run_dir) / "gpu.csv"
    values = []
    memory = []
    if not path.exists():
        return "", "", ""
    with path.open(encoding="utf-8", errors="replace") as handle:
        rows = csv.DictReader(handle)
        for row in rows:
            try:
                values.append(float(row["gpu_util_percent"]))
                memory.append(float(row["mem_used_mb"]))
            except (KeyError, TypeError, ValueError):
                continue
    if not values:
        return "", "", ""
    return round(sum(values) / len(values), 4), max(values), max(memory)

fields = [
    "benchmark", "variant", "concurrency", "worker_gpu_ids", "wall_time_s", "requests_s",
    "turns", "errors", "prompt_tokens", "completion_tokens", "total_tokens",
    "completion_tokens_s", "total_tokens_s", "latency_avg_s", "latency_max_s",
    "latency_p50_s", "latency_p90_s", "latency_p99_s",
    "teacher_forced_turns", "teacher_forcing_mismatches",
    "spec_metric_turns", "spec_verify_ct", "spec_num_correct_drafts",
    "spec_num_proposed_drafts", "spec_accept_length",
    "spec_draft_accept_length", "spec_accept_rate",
    "gpu_util_avg", "gpu_util_max", "gpu_mem_used_max_mb",
    "output_dir", "worker0_log", "worker1_log", "gateway_log", "gpu_log",
]
row = {field: summary.get(field, "") for field in fields}
gpu_util_avg, gpu_util_max, gpu_mem_used_max_mb = gpu_metrics()
row.update({
    "benchmark": benchmark,
    "variant": variant,
    "concurrency": concurrency,
    "worker_gpu_ids": worker_gpu_ids,
    "gpu_util_avg": gpu_util_avg,
    "gpu_util_max": gpu_util_max,
    "gpu_mem_used_max_mb": gpu_mem_used_max_mb,
    "output_dir": run_dir,
    "worker0_log": str(Path(run_dir) / "worker0.log"),
    "worker1_log": str(Path(run_dir) / "worker1.log"),
    "gateway_log": str(Path(run_dir) / "gateway.log"),
    "gpu_log": str(Path(run_dir) / "gpu.csv"),
})

path = Path(csv_path)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    if handle.tell() == 0:
        writer.writeheader()
    writer.writerow(row)
PY
}

run_one() {
  local benchmark="$1"
  local variant="$2"
  local concurrency="$3"
  local run_dir="${BATCH_DIR}/${benchmark}/${variant}_c${concurrency}"
  local l2_path="${run_dir}/l2_history"
  local namespace="${benchmark}_${variant}_c${concurrency}_${BATCH_TS}"
  local runner=()
  if [[ "$benchmark" == "swe_bench" || "$benchmark" == "swebench" ]]; then
    require_path "$SWE_TRACE_JSONL"
    runner=(
      "$PYTHON_BIN" "${AS_DIR}/SD_benchmark/swe_bench/run_swe_trace.py"
      --trace-jsonl "$SWE_TRACE_JSONL"
      --server-url "http://127.0.0.1:${GATEWAY_PORT}/v1"
      --model "$SERVED_MODEL_NAME"
      --api-key "$API_KEY"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --seed "$SEED"
      --max-tokens "$MAX_TOKENS"
      --concurrency "$concurrency"
      --timeout "$TIMEOUT"
      --tool-mode "$SWE_TOOL_MODE"
      --print-every "$PRINT_EVERY"
      --output-dir "$run_dir"
    )
    [[ -n "$LIMIT" ]] && runner+=(--limit "$LIMIT")
    [[ -n "$MAX_STEPS_PER_WORKFLOW" ]] && runner+=(--max-steps-per-workflow "$MAX_STEPS_PER_WORKFLOW")
  else
    runner=(
      "$PYTHON_BIN" "${AS_DIR}/SD_benchmark/run_benchmark.py"
      --benchmark "$benchmark"
      --benchmark-root "${AS_DIR}/SD_benchmark"
      --server-url "http://127.0.0.1:${GATEWAY_PORT}/v1"
      --model "$SERVED_MODEL_NAME"
      --api-key "$API_KEY"
      --temperature "$TEMPERATURE"
      --top-p "$TOP_P"
      --seed "$SEED"
      --max-tokens "$MAX_TOKENS"
      --concurrency "$concurrency"
      --timeout "$TIMEOUT"
      --humaneval-style "$HUMANEVAL_STYLE"
      --print-every "$PRINT_EVERY"
      --output-dir "$run_dir"
    )
    [[ -n "$LIMIT" ]] && runner+=(--limit "$LIMIT")
    [[ -n "$CATEGORIES" ]] && runner+=(--categories "$CATEGORIES")
    if [[ -n "$TEACHER_FORCING_TRACE" ]]; then
      require_path "$TEACHER_FORCING_TRACE"
      require_path "$TEACHER_FORCING_TOKENIZER"
      runner+=(
        --teacher-forcing-trace "$TEACHER_FORCING_TRACE"
        --tokenizer "$TEACHER_FORCING_TOKENIZER"
      )
    fi
  fi

  # Keep every service artifact and, when enabled, its shared history under
  # this point's directory so cleanup cannot affect another comparison point.
  mkdir -p "$run_dir" "$l2_path"
  parse_variant "$variant"
  if (( ${#SPEC_ARGS[@]} > 0 )); then
    runner+=(--collect-sglang-spec-metrics)
  fi
  echo "[multi-batch] benchmark=${benchmark} variant=${variant} concurrency=${concurrency}"

  start_worker worker-0 "$WORKER0_PORT" "${GPU_IDS[0]}" "$l2_path" "$namespace" "$run_dir/worker0.log" WORKER0_PID
  wait_for_http "http://127.0.0.1:${WORKER0_PORT}/v1/models" worker-0 "$WORKER0_PID"
  start_worker worker-1 "$WORKER1_PORT" "${GPU_IDS[1]}" "$l2_path" "$namespace" "$run_dir/worker1.log" WORKER1_PID
  wait_for_http "http://127.0.0.1:${WORKER1_PORT}/v1/models" worker-1 "$WORKER1_PID"

  echo "[multi-batch] starting gateway on port ${GATEWAY_PORT}"
  setsid "$PYTHON_BIN" -m sglang_router.launch_router \
    --worker-urls "http://127.0.0.1:${WORKER0_PORT}" "http://127.0.0.1:${WORKER1_PORT}" \
    --policy round_robin \
    --host "$SERVER_HOST" \
    --port "$GATEWAY_PORT" >"$run_dir/gateway.log" 2>&1 &
  GATEWAY_PID="$!"
  wait_for_http "http://127.0.0.1:${GATEWAY_PORT}/v1/models" gateway "$GATEWAY_PID"

  save_server_info "http://127.0.0.1:${WORKER0_PORT}" "$run_dir/worker0_server_info_before.json"
  save_server_info "http://127.0.0.1:${WORKER1_PORT}" "$run_dir/worker1_server_info_before.json"
  start_gpu_sampler "$run_dir/gpu.csv"

  "${runner[@]}" | tee "$run_dir/replay.log"

  stop_gpu_sampler
  save_server_info "http://127.0.0.1:${WORKER0_PORT}" "$run_dir/worker0_server_info_after.json"
  save_server_info "http://127.0.0.1:${WORKER1_PORT}" "$run_dir/worker1_server_info_after.json"
  stop_services
  append_result_row "$run_dir/summary.json" "$benchmark" "$variant" "$concurrency" "$run_dir"
}

main() {
  [[ ${#GPU_IDS[@]} -eq 2 ]] || die "GPU_IDS must contain exactly two IDs, e.g. GPU_IDS='0 1'."
  require_path "${AS_DIR}/SD_benchmark/run_benchmark.py"
  require_path "$MODEL_PATH"
  require_path "$CC_BIN"
  require_path "$CXX_BIN"
  command -v "$SGLANG_BIN" >/dev/null 2>&1 || die "SGLANG_BIN is not executable: $SGLANG_BIN"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "PYTHON_BIN is not executable: $PYTHON_BIN"

  mkdir -p "$BATCH_DIR"
  echo "[multi-batch] batch_dir=${BATCH_DIR}"
  echo "[multi-batch] benchmarks=${BENCHMARKS_ARR[*]}"
  echo "[multi-batch] variants=${VARIANTS_ARR[*]}"
  echo "[multi-batch] concurrencies=${CONCURRENCIES_ARR[*]}"
  echo "[multi-batch] worker_gpu_ids=${GPU_IDS[*]}"

  local benchmark variant concurrency
  for benchmark in "${BENCHMARKS_ARR[@]}"; do
    for variant in "${VARIANTS_ARR[@]}"; do
      for concurrency in "${CONCURRENCIES_ARR[@]}"; do
        run_one "$benchmark" "$variant" "$concurrency"
      done
    done
  done
  echo "[multi-batch] results=${RESULTS_CSV}"
}

main "$@"
