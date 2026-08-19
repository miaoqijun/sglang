#!/usr/bin/env bash
set -euo pipefail

# Batch launcher for official SPEED-Bench through the coauthor SGLang gateway.
# Every (SPEED config, variant, concurrency) combination starts fresh workers
# and a fresh router, then invokes the official run.py with SGLANG_REMOTE.
# This prevents NGRAM history and server state from leaking across variants.
#
# The official checkout must first receive the explicit SGLANG_REMOTE adapter:
#   python SD_benchmark/speed_bench/install_official_runner_adapter.py \
#     /path/to/specdec_bench
#
# Common examples:
#   SPEED_CONFIGS="throughput_1k" VARIANTS="baseline ngram_d4 ngram_d8" \
#     CONCURRENCIES="1 4" bash SD_benchmark/speed_bench/run_speed_bench_batch.sh
#   SPEED_CONFIGS="throughput_1k throughput_32k" VARIANTS="ngram_prob_d8" \
#     NGRAM_BFS_BREADTH=2 NGRAM_MAX_TRIE_DEPTH=24 \
#     bash SD_benchmark/speed_bench/run_speed_bench_batch.sh
#
# Variant names choose draft length and, unless NGRAM_MATCH_TYPE is set, match
# type: baseline, ngram_d<N>/ngram_bfs_d<N>, ngram_prob_d<N>.  Append _no_l2
# to disable shared L2 only, e.g. ngram_d8_no_l2.

usage() {
  cat <<'EOF'
Usage:
  OFFICIAL_RUNNER_DIR=/path/to/specdec_bench MODEL_PATH=/path/to/model \
  SPEED_CONFIGS="throughput_1k" VARIANTS="baseline ngram_d8" \
  CONCURRENCIES="1 4" GPU_IDS="0 1" \
  bash SD_benchmark/speed_bench/run_speed_bench_batch.sh

Runs NVIDIA's official SPEED-Bench runner through two SGLang workers and a
round-robin gateway. Install the remote adapter first with
install_official_runner_adapter.py.

Key environment variables:
  OFFICIAL_RUNNER_DIR  Official NVIDIA examples/specdec_bench checkout.
  SPEED_CONFIGS        Prepared dataset configurations, e.g. throughput_1k.
  VARIANTS             baseline, ngram_d<N>, ngram_prob_d<N>; append _no_l2.
  CONCURRENCIES        Official runner --concurrency values.
  NUM_REQUESTS         Number of official requests selected per point.
  OUTPUT_LENGTH        Official runner generation length. Default: 1024.
  GPU_IDS              Exactly two worker GPU IDs, e.g. "0 1".
  RUNNER_PYTHON_BIN    Optional Python executable for the official runner.
  BATCH_ROOT           Parent output directory.

Each point gets fresh workers, router, L2 namespace, and official-runner output.
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AS_DIR="${AS_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
# Workers and router use PYTHON_BIN. The official runner can use a separate
# environment with its datasets/metrics dependencies installed.
RUNNER_PYTHON_BIN="${RUNNER_PYTHON_BIN:-${PYTHON_BIN}}"
SGLANG_BIN="${SGLANG_BIN:-sglang}"

OFFICIAL_RUNNER_DIR="${OFFICIAL_RUNNER_DIR:-${AS_DIR}/SD_benchmark/speedbench-official}"
MODEL_PATH="${MODEL_PATH:-${HOME}/swq/models/Qwen2.5-14B-Instruct}"
TOKENIZER_PATH="${TOKENIZER_PATH:-${MODEL_PATH}}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
# The upstream example normally writes data/ below its working directory. A
# copied checkout may instead retain the original specdec_bench/data layout.
# Set SPEED_DATA_ROOT explicitly if neither layout applies.
SPEED_DATA_ROOT="${SPEED_DATA_ROOT:-}"

SPEED_CONFIGS_STR="${SPEED_CONFIGS:-throughput_1k}"
SPEED_CONFIGS_STR="${SPEED_CONFIGS_STR//,/ }"
read -r -a SPEED_CONFIGS <<< "${SPEED_CONFIGS_STR}"
VARIANTS_STR="${VARIANTS:-baseline ngram_d8}"
VARIANTS_STR="${VARIANTS_STR//,/ }"
read -r -a VARIANTS <<< "${VARIANTS_STR}"
CONCURRENCIES_STR="${CONCURRENCIES:-1}"
CONCURRENCIES_STR="${CONCURRENCIES_STR//,/ }"
read -r -a CONCURRENCIES <<< "${CONCURRENCIES_STR}"

NUM_REQUESTS="${NUM_REQUESTS:-20}"
OUTPUT_LENGTH="${OUTPUT_LENGTH:-1024}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-4096}"
TP_SIZE="${TP_SIZE:-1}"
EP_SIZE="${EP_SIZE:-1}"
SHOW_PROGRESS="${SHOW_PROGRESS:-1}"

SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
API_KEY="${API_KEY:-dummy}"
GATEWAY_PORT="${GATEWAY_PORT:-1919}"
WORKER0_PORT="${WORKER0_PORT:-19191}"
WORKER1_PORT="${WORKER1_PORT:-19192}"
GPU_IDS_STR="${GPU_IDS:-0 0}"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"
[[ "${#GPU_IDS[@]}" -eq 2 ]] || { echo "GPU_IDS must contain exactly two IDs, e.g. '0 0' or '0 1'." >&2; exit 2; }

WORKER_MEM_FRACTION_STATIC="${WORKER_MEM_FRACTION_STATIC:-0.42}"
WORKER_MAX_TOTAL_TOKENS="${WORKER_MAX_TOTAL_TOKENS:-32768}"
WORKER_MAX_RUNNING_REQUESTS="${WORKER_MAX_RUNNING_REQUESTS:-}"
WORKER_READY_TIMEOUT="${WORKER_READY_TIMEOUT:-900}"

# NGRAM knobs. A nonempty NGRAM_MATCH_TYPE overrides the strategy implied by
# the variant name, which is convenient for sweeps over one draft length.
NGRAM_MATCH_TYPE="${NGRAM_MATCH_TYPE:-}"
NGRAM_CAPACITY="${NGRAM_CAPACITY:-500000}"
NGRAM_MAX_TRIE_DEPTH="${NGRAM_MAX_TRIE_DEPTH:-18}"
NGRAM_BFS_BREADTH="${NGRAM_BFS_BREADTH:-1}"
NGRAM_FORCE_GREEDY_VERIFY="${NGRAM_FORCE_GREEDY_VERIFY:-True}"
# Space-separated additional worker flags for branch-specific NGRAM features.
# Example: NGRAM_EXTRA_ARGS="--speculative-ngram-l2-read-only".
NGRAM_EXTRA_ARGS_STR="${NGRAM_EXTRA_ARGS:-}"
read -r -a NGRAM_EXTRA_ARGS <<< "${NGRAM_EXTRA_ARGS_STR}"
L2_ENABLED="${L2_ENABLED:-1}"
L2_BACKEND="${L2_BACKEND:-mmap}"
L2_MMAP_CAPACITY="${L2_MMAP_CAPACITY:-65536}"

CC_BIN="${CC_BIN:-/usr/bin/gcc-10}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-10}"
CUDAHOSTCXX_BIN="${CUDAHOSTCXX_BIN:-${CXX_BIN}}"
NVCC_PREPEND_FLAGS_VALUE="${NVCC_PREPEND_FLAGS_VALUE:--ccbin ${CXX_BIN}}"

BATCH_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_ROOT="${BATCH_ROOT:-${AS_DIR}/SD_benchmark/outputs/speed_bench_multi_instance}"
BATCH_DIR="${BATCH_DIR:-${BATCH_ROOT}/${BATCH_TS}}"
RESULTS_CSV="${RESULTS_CSV:-${BATCH_DIR}/batch_results.csv}"

WORKER0_PID=""
WORKER1_PID=""
GATEWAY_PID=""
SPEC_ARGS=()
USE_L2=0

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_path() {
  [[ -e "$1" ]] || die "Required path does not exist: $1"
}

stop_process_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "$pid" 2>/dev/null || return 0
      sleep 1
    done
    kill -KILL -- "-${pid}" 2>/dev/null || true
  fi
}

stop_services() {
  stop_process_group "$GATEWAY_PID"
  stop_process_group "$WORKER1_PID"
  stop_process_group "$WORKER0_PID"
  GATEWAY_PID=""
  WORKER1_PID=""
  WORKER0_PID=""
}
trap stop_services EXIT INT TERM

wait_for_http() {
  local url="$1"
  local name="$2"
  local pid="$3"
  local started
  started="$(date +%s)"
  while true; do
    if curl -fsS -H "Authorization: Bearer ${API_KEY}" "$url" >/dev/null 2>&1; then
      echo "[speed-batch] ${name} ready: ${url}"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || die "${name} exited before becoming ready"
    (( $(date +%s) - started <= WORKER_READY_TIMEOUT )) || die "Timed out waiting for ${name}: ${url}"
    sleep 1
  done
}

parse_variant() {
  local value="$1"
  SPEC_ARGS=()
  USE_L2=0
  if [[ "$value" == "baseline" ]]; then
    return 0
  fi

  if [[ "$value" == *_no_l2 ]]; then
    value="${value%_no_l2}"
  elif [[ "$L2_ENABLED" == "1" ]]; then
    USE_L2=1
  fi

  local implied_match draft
  if [[ "$value" =~ ^(ngram|ngram_bfs)_d([0-9]+)$ ]]; then
    implied_match="BFS"
    draft="${BASH_REMATCH[2]}"
  elif [[ "$value" =~ ^(ngram_prob|prob)_d([0-9]+)$ ]]; then
    implied_match="PROB"
    draft="${BASH_REMATCH[2]}"
  else
    die "Unsupported VARIANT=${1}; use baseline, ngram_d<N>, ngram_prob_d<N>, optionally suffixed _no_l2."
  fi

  local match_type="${NGRAM_MATCH_TYPE:-${implied_match}}"
  SPEC_ARGS=(
    --speculative-algorithm NGRAM
    --speculative-ngram-min-bfs-breadth "$NGRAM_BFS_BREADTH"
    --speculative-ngram-max-bfs-breadth "$NGRAM_BFS_BREADTH"
    --speculative-ngram-match-type "$match_type"
    --speculative-ngram-max-trie-depth "$NGRAM_MAX_TRIE_DEPTH"
    --speculative-num-draft-tokens "$draft"
    --speculative-ngram-capacity "$NGRAM_CAPACITY"
  )
  if (( ${#NGRAM_EXTRA_ARGS[@]} > 0 )); then
    SPEC_ARGS+=("${NGRAM_EXTRA_ARGS[@]}")
  fi
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
  [[ -n "$WORKER_MAX_RUNNING_REQUESTS" ]] && args+=(--max-running-requests "$WORKER_MAX_RUNNING_REQUESTS")
  (( ${#SPEC_ARGS[@]} > 0 )) && args+=("${SPEC_ARGS[@]}")
  if (( USE_L2 == 1 )); then
    args+=(
      --speculative-ngram-l2-history-path "$l2_path"
      --speculative-ngram-l2-backend "$L2_BACKEND"
      --speculative-ngram-l2-mmap-capacity "$L2_MMAP_CAPACITY"
      --speculative-ngram-l2-namespace "$namespace"
      --speculative-ngram-l2-instance-id "$worker_id"
    )
  fi

  echo "[speed-batch] starting ${worker_id}: gpu=${gpu_id} port=${port}"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    CC="$CC_BIN" \
    CXX="$CXX_BIN" \
    CUDAHOSTCXX="$CUDAHOSTCXX_BIN" \
    NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS_VALUE" \
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY="$NGRAM_FORCE_GREEDY_VERIFY" \
    "$SGLANG_BIN" "${args[@]}" >"$log_path" 2>&1 &
  printf -v "$pid_var" '%s' "$!"
}

append_result_row() {
  local run_log="$1"
  local config="$2"
  local variant="$3"
  local concurrency="$4"
  local run_dir="$5"
  local wall_time_s="$6"

  "$PYTHON_BIN" - "$run_log" "$RESULTS_CSV" "$config" "$variant" "$concurrency" "$run_dir" "$wall_time_s" <<'PY'
import csv
import re
import sys
from pathlib import Path

log_path, csv_path, config, variant, concurrency, run_dir, wall_time_s = sys.argv[1:]
text = Path(log_path).read_text(encoding="utf-8", errors="replace")

def last_float(pattern):
    matches = re.findall(pattern, text, flags=re.MULTILINE)
    return matches[-1] if matches else ""

fields = [
    "speed_config", "variant", "concurrency", "wall_time_s",
    "output_tps", "output_tps_per_gpu", "average_accept_length",
    "request_time_mean_s", "ttft_mean_s", "generation_step_mean_s",
    "run_dir", "runner_log", "worker0_log", "worker1_log", "router_log",
]
row = {
    "speed_config": config,
    "variant": variant,
    "concurrency": concurrency,
    "wall_time_s": wall_time_s,
    "output_tps": last_float(r"^Output TPS\s+([0-9.eE+-]+)\s*$"),
    "output_tps_per_gpu": last_float(r"^Output TPS/gpu\s+([0-9.eE+-]+)\s*$"),
    "average_accept_length": last_float(r"^│ Overall Average │\s*([0-9.]+)\s*│\s*$"),
    "request_time_mean_s": last_float(r"^E2E Request Time .*?'mean': '([0-9.]+)'"),
    "ttft_mean_s": last_float(r"^TTFT Time .*?'mean': '([0-9.]+)'"),
    "generation_step_mean_s": last_float(r"^Request Generation Step Time .*?'mean': '([0-9.]+)'"),
    "run_dir": run_dir,
    "runner_log": str(Path(run_dir) / "runner.log"),
    "worker0_log": str(Path(run_dir) / "worker0.log"),
    "worker1_log": str(Path(run_dir) / "worker1.log"),
    "router_log": str(Path(run_dir) / "router.log"),
}

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
  local speed_config="$1"
  local variant="$2"
  local concurrency="$3"
  local run_dir="${BATCH_DIR}/${speed_config}/${variant}_c${concurrency}"
  # Keep L2 under this individual run directory.  A timestamped BATCH_DIR and
  # namespace prevent history from being reused by later experiment points.
  local l2_path="${run_dir}/l2_history"
  local namespace="speed_${speed_config}_${variant}_c${concurrency}_${BATCH_TS}"
  local data_root data_path
  if [[ -n "$SPEED_DATA_ROOT" ]]; then
    data_root="$SPEED_DATA_ROOT"
  elif [[ -d "${OFFICIAL_RUNNER_DIR}/data/speed" ]]; then
    data_root="${OFFICIAL_RUNNER_DIR}/data/speed"
  else
    data_root="${OFFICIAL_RUNNER_DIR}/specdec_bench/data/speed"
  fi
  data_path="${data_root}/${speed_config}"
  local started ended

  require_path "$data_path"
  mkdir -p "$run_dir"
  parse_variant "$variant"
  if (( USE_L2 == 1 )); then
    mkdir -p "$l2_path"
  fi

  echo "[speed-batch] config=${speed_config} variant=${variant} concurrency=${concurrency} l2=${USE_L2}"
  start_worker worker-0 "$WORKER0_PORT" "${GPU_IDS[0]}" "$l2_path" "$namespace" "$run_dir/worker0.log" WORKER0_PID
  wait_for_http "http://127.0.0.1:${WORKER0_PORT}/v1/models" worker-0 "$WORKER0_PID"
  start_worker worker-1 "$WORKER1_PORT" "${GPU_IDS[1]}" "$l2_path" "$namespace" "$run_dir/worker1.log" WORKER1_PID
  wait_for_http "http://127.0.0.1:${WORKER1_PORT}/v1/models" worker-1 "$WORKER1_PID"

  echo "[speed-batch] starting router on port ${GATEWAY_PORT}"
  setsid "$PYTHON_BIN" -m sglang_router.launch_router \
    --worker-urls "http://127.0.0.1:${WORKER0_PORT}" "http://127.0.0.1:${WORKER1_PORT}" \
    --policy round_robin \
    --host "$SERVER_HOST" \
    --port "$GATEWAY_PORT" >"$run_dir/router.log" 2>&1 &
  GATEWAY_PID="$!"
  wait_for_http "http://127.0.0.1:${GATEWAY_PORT}/v1/models" router "$GATEWAY_PID"
  wait_for_http "http://127.0.0.1:${GATEWAY_PORT}/health_generate" router-generate "$GATEWAY_PID"

  started="$(date +%s.%N)"
  local runner_args=(
    "$RUNNER_PYTHON_BIN" run.py
    --model_dir "$MODEL_PATH"
    --tokenizer "$TOKENIZER_PATH"
    --dataset speed
    --dataset_path "$data_path"
    --engine SGLANG_REMOTE
    --server_url "http://127.0.0.1:${GATEWAY_PORT}"
    --speculative_algorithm NONE
    --tp_size "$TP_SIZE"
    --ep_size "$EP_SIZE"
    --output_length "$OUTPUT_LENGTH"
    --num_requests "$NUM_REQUESTS"
    --concurrency "$concurrency"
    --temperature "$TEMPERATURE"
    --max_seq_len "$MAX_SEQ_LEN"
    --save_dir "$run_dir/official_output"
  )
  if [[ "$SHOW_PROGRESS" == "1" ]]; then
    runner_args+=(--show_progress)
  fi

  (
    cd "$OFFICIAL_RUNNER_DIR"
    SGLANG_API_KEY="$API_KEY" "${runner_args[@]}"
  ) 2>&1 | tee "$run_dir/runner.log"
  ended="$(date +%s.%N)"

  stop_services
  local elapsed
  elapsed="$("$PYTHON_BIN" - "$started" "$ended" <<'PY'
import sys
print(f"{float(sys.argv[2]) - float(sys.argv[1]):.6f}")
PY
)"
  append_result_row "$run_dir/runner.log" "$speed_config" "$variant" "$concurrency" "$run_dir" "$elapsed"
}

main() {
  require_path "$MODEL_PATH"
  require_path "$TOKENIZER_PATH"
  require_path "$OFFICIAL_RUNNER_DIR/run.py"
  require_path "$OFFICIAL_RUNNER_DIR/specdec_bench/models/sglang_remote.py"
  require_path "$CC_BIN"
  require_path "$CXX_BIN"
  command -v "$SGLANG_BIN" >/dev/null 2>&1 || die "SGLANG_BIN is not executable: $SGLANG_BIN"
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "PYTHON_BIN is not executable: $PYTHON_BIN"
  command -v "$RUNNER_PYTHON_BIN" >/dev/null 2>&1 || die "RUNNER_PYTHON_BIN is not executable: $RUNNER_PYTHON_BIN"
  command -v curl >/dev/null 2>&1 || die "curl is required"

  mkdir -p "$BATCH_DIR"
  echo "[speed-batch] batch_dir=${BATCH_DIR}"
  echo "[speed-batch] speed_configs=${SPEED_CONFIGS[*]}"
  echo "[speed-batch] variants=${VARIANTS[*]}"
  echo "[speed-batch] concurrencies=${CONCURRENCIES[*]}"
  echo "[speed-batch] ngram_match_override=${NGRAM_MATCH_TYPE:-variant-default} breadth=${NGRAM_BFS_BREADTH} trie_depth=${NGRAM_MAX_TRIE_DEPTH}"

  local speed_config variant concurrency
  for speed_config in "${SPEED_CONFIGS[@]}"; do
    for variant in "${VARIANTS[@]}"; do
      for concurrency in "${CONCURRENCIES[@]}"; do
        run_one "$speed_config" "$variant" "$concurrency"
      done
    done
  done
  echo "[speed-batch] results=${RESULTS_CSV}"
}

main "$@"
