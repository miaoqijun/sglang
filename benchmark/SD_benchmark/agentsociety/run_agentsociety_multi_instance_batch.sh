#!/usr/bin/env bash
set -euo pipefail

# Portable two-worker AgentSociety replay sweep for the coauthor's SGLang
# gateway branch. Each point receives fresh workers, router, L2 directory, and
# L2 namespace, so NGRAM history cannot leak across comparisons.
#
# Run from an activated SGLang/gateway environment. The gateway must include
# SD_benchmark/patches/sgl_model_gateway_return_meta_info.patch when raw
# speculative counters are needed through the router.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SGLANG_BIN="${SGLANG_BIN:-sglang}"
MODEL_PATH="${MODEL_PATH:-${HOME}/swq/models/Qwen2.5-14B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
RECORD_PATH="${RECORD_PATH:-${SCRIPT_DIR}/records/agentsociety_record.jsonl}"

SERVER_HOST="${SERVER_HOST:-0.0.0.0}"
API_KEY="${API_KEY:-dummy}"
GATEWAY_PORT="${GATEWAY_PORT:-1919}"
WORKER0_PORT="${WORKER0_PORT:-19191}"
WORKER1_PORT="${WORKER1_PORT:-19192}"
GPU_IDS_STR="${GPU_IDS:-0 0}"
GPU_IDS_STR="${GPU_IDS_STR//,/ }"
read -r -a GPU_IDS <<< "${GPU_IDS_STR}"

WORKER_MEM_FRACTION_STATIC="${WORKER_MEM_FRACTION_STATIC:-0.42}"
WORKER_MAX_TOTAL_TOKENS="${WORKER_MAX_TOTAL_TOKENS:-32768}"
WORKER_MAX_RUNNING_REQUESTS="${WORKER_MAX_RUNNING_REQUESTS:-}"
TIMEOUT="${TIMEOUT:-600}"
REPLAY_MODE="${REPLAY_MODE:-faithful}"
CALL_TYPES="${CALL_TYPES:-}"
# Deterministic decoding is the default for serving/SD comparisons. Set
# USE_RECORDED_TEMPERATURE=1 to preserve each raw record's request setting.
TEMPERATURE="${TEMPERATURE:-0}"
USE_RECORDED_TEMPERATURE="${USE_RECORDED_TEMPERATURE:-0}"
REPLAY_MAX_TOKENS="${REPLAY_MAX_TOKENS:-}"
PRINT_EVERY="${PRINT_EVERY:-100}"

CONCURRENCIES_STR="${CONCURRENCIES:-1 4 16 32}"
CONCURRENCIES_STR="${CONCURRENCIES_STR//,/ }"
read -r -a CONCURRENCIES <<< "${CONCURRENCIES_STR}"
VARIANTS_STR="${VARIANTS:-baseline ngram_d4 ngram_d8 ngram_prob_d8}"
VARIANTS_STR="${VARIANTS_STR//,/ }"
read -r -a VARIANTS <<< "${VARIANTS_STR}"

NGRAM_MATCH_TYPE="${NGRAM_MATCH_TYPE:-}"
NGRAM_CAPACITY="${NGRAM_CAPACITY:-500000}"
NGRAM_MAX_TRIE_DEPTH="${NGRAM_MAX_TRIE_DEPTH:-18}"
NGRAM_BFS_BREADTH="${NGRAM_BFS_BREADTH:-1}"
NGRAM_EXTRA_ARGS_STR="${NGRAM_EXTRA_ARGS:-}"
read -r -a NGRAM_EXTRA_ARGS <<< "${NGRAM_EXTRA_ARGS_STR}"
FORCE_GREEDY_VERIFY="${FORCE_GREEDY_VERIFY:-True}"

L2_ENABLED="${L2_ENABLED:-1}"
L2_BACKEND="${L2_BACKEND:-mmap}"
L2_MMAP_CAPACITY="${L2_MMAP_CAPACITY:-65536}"
L2_PATH_ROOT="${L2_PATH_ROOT:-}"

# These defaults cover the H200 Ubuntu/CUDA JIT environment. Override them
# when the cluster uses another compiler path.
CC_BIN="${CC_BIN:-/usr/bin/gcc-10}"
CXX_BIN="${CXX_BIN:-/usr/bin/g++-10}"
CUDAHOSTCXX_BIN="${CUDAHOSTCXX_BIN:-${CXX_BIN}}"
NVCC_PREPEND_FLAGS_VALUE="${NVCC_PREPEND_FLAGS_VALUE:--ccbin ${CXX_BIN}}"
EXTRA_WORKER_ARGS_STR="${EXTRA_WORKER_ARGS:-}"
read -r -a EXTRA_WORKER_ARGS <<< "${EXTRA_WORKER_ARGS_STR}"

BATCH_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_ROOT="${BATCH_ROOT:-${SCRIPT_DIR}/../outputs/agentsociety}"
BATCH_DIR="${BATCH_DIR:-${BATCH_ROOT}/${BATCH_TS}}"
RESULTS_CSV="${RESULTS_CSV:-${BATCH_DIR}/batch_results.csv}"

WORKER0_PID=""
WORKER1_PID=""
GATEWAY_PID=""
GPU_SAMPLER_PID=""
SPEC_ARGS=()
USE_L2=0

die() { echo "ERROR: $*" >&2; exit 1; }
require_path() { [[ -e "$1" ]] || die "Required path does not exist: $1"; }

stop_pid_group() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
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

cleanup() {
  stop_gpu_sampler
  stop_services
}
trap cleanup EXIT INT TERM

wait_for_http() {
  local url="$1" name="$2" pid="$3"
  local started
  started="$(date +%s)"
  while true; do
    if curl -fsS -H "Authorization: Bearer ${API_KEY}" "$url" >/dev/null 2>&1; then
      echo "[agentsociety-multi] ${name} ready: ${url}"
      return 0
    fi
    kill -0 "$pid" 2>/dev/null || die "${name} exited before becoming ready; see ${CURRENT_RUN_DIR}"
    (( $(date +%s) - started <= ${READY_TIMEOUT:-900} )) || die "Timed out waiting for ${name}: ${url}"
    sleep 1
  done
}

start_gpu_sampler() {
  local output="$1"
  command -v nvidia-smi >/dev/null 2>&1 || return 0
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
  local value="$1"
  SPEC_ARGS=()
  USE_L2=0
  [[ "$value" == "baseline" ]] && return 0

  if [[ "$value" == *_no_l2 ]]; then
    value="${value%_no_l2}"
  elif [[ "$L2_ENABLED" == "1" ]]; then
    USE_L2=1
  fi

  local implied draft
  if [[ "$value" =~ ^(ngram|ngram_bfs)_d([0-9]+)$ ]]; then
    implied="BFS"; draft="${BASH_REMATCH[2]}"
  elif [[ "$value" =~ ^(ngram_prob|prob)_d([0-9]+)$ ]]; then
    implied="PROB"; draft="${BASH_REMATCH[2]}"
  else
    die "Unsupported VARIANT=${1}; use baseline, ngram_d<N>, ngram_prob_d<N>, or append _no_l2."
  fi
  SPEC_ARGS=(
    --speculative-algorithm NGRAM
    --speculative-ngram-min-bfs-breadth "$NGRAM_BFS_BREADTH"
    --speculative-ngram-max-bfs-breadth "$NGRAM_BFS_BREADTH"
    --speculative-ngram-match-type "${NGRAM_MATCH_TYPE:-$implied}"
    --speculative-ngram-max-trie-depth "$NGRAM_MAX_TRIE_DEPTH"
    --speculative-num-draft-tokens "$draft"
    --speculative-ngram-capacity "$NGRAM_CAPACITY"
  )
  if (( ${#NGRAM_EXTRA_ARGS[@]} > 0 )); then
    SPEC_ARGS+=("${NGRAM_EXTRA_ARGS[@]}")
  fi
  return 0
}

start_worker() {
  local worker_id="$1" port="$2" gpu_id="$3" l2_path="$4" namespace="$5" log="$6" pid_var="$7"
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
  if (( ${#SPEC_ARGS[@]} > 0 )); then
    args+=("${SPEC_ARGS[@]}")
  fi
  if (( USE_L2 == 1 )); then
    args+=(
      --speculative-ngram-l2-history-path "$l2_path"
      --speculative-ngram-l2-backend "$L2_BACKEND"
      --speculative-ngram-l2-mmap-capacity "$L2_MMAP_CAPACITY"
      --speculative-ngram-l2-namespace "$namespace"
      --speculative-ngram-l2-instance-id "$worker_id"
    )
  fi
  if (( ${#EXTRA_WORKER_ARGS[@]} > 0 )); then
    args+=("${EXTRA_WORKER_ARGS[@]}")
  fi

  echo "[agentsociety-multi] starting ${worker_id} on port ${port}, GPU ${gpu_id}"
  setsid env \
    CUDA_VISIBLE_DEVICES="$gpu_id" \
    CC="$CC_BIN" CXX="$CXX_BIN" CUDAHOSTCXX="$CUDAHOSTCXX_BIN" \
    NVCC_PREPEND_FLAGS="$NVCC_PREPEND_FLAGS_VALUE" \
    SGLANG_NGRAM_FORCE_GREEDY_VERIFY="$FORCE_GREEDY_VERIFY" \
    "$SGLANG_BIN" "${args[@]}" >"$log" 2>&1 &
  printf -v "$pid_var" '%s' "$!"
}

save_server_info() {
  local url="$1" output="$2"
  curl -fsS -H "Authorization: Bearer ${API_KEY}" "${url}/server_info" >"$output" 2>/dev/null || true
}

append_result() {
  local summary_path="$1" variant="$2" concurrency="$3" run_dir="$4" l2_namespace="$5"
  "$PYTHON_BIN" - "$summary_path" "$RESULTS_CSV" "$variant" "$concurrency" "$run_dir" "$l2_namespace" "$USE_L2" "${GPU_IDS[*]}" <<'PY'
import csv
import json
import sys
from pathlib import Path

summary_path, csv_path, variant, concurrency, run_dir, namespace, l2_enabled, worker_gpu_ids = sys.argv[1:]
summary_file = Path(summary_path)
summary = json.loads(summary_file.read_text(encoding="utf-8"))
gpu_path = Path(run_dir) / "gpu.csv"
utilization, memory = [], []
if gpu_path.exists():
    with gpu_path.open(encoding="utf-8", errors="replace") as handle:
        for row in csv.DictReader(handle):
            try:
                utilization.append(float(row["gpu_util_percent"]))
                memory.append(float(row["mem_used_mb"]))
            except (KeyError, TypeError, ValueError):
                pass
gpu = {
    "gpu_util_avg": round(sum(utilization) / len(utilization), 6) if utilization else None,
    "gpu_util_max": max(utilization) if utilization else None,
    "gpu_mem_used_max_mb": max(memory) if memory else None,
    "gpu_log": str(gpu_path),
}
summary.update({
    "deployment": "two_worker_gateway",
    "concurrency": int(concurrency),
    "worker_gpu_ids": worker_gpu_ids,
    "l2_enabled": l2_enabled == "1",
    "l2_namespace": namespace if l2_enabled == "1" else None,
    **gpu,
})
summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

fields = [
    "benchmark", "variant", "concurrency", "max_concurrency", "deployment", "worker_gpu_ids", "l2_enabled", "l2_namespace",
    "wall_time_s", "requests_s", "turns", "errors", "prompt_tokens", "completion_tokens", "total_tokens",
    "completion_tokens_s", "total_tokens_s", "latency_avg_s", "latency_p50_s", "latency_p90_s", "latency_p99_s",
    "spec_metric_turns", "spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts",
    "spec_accept_length", "spec_draft_accept_length", "spec_accept_rate",
    "gpu_util_avg", "gpu_util_max", "gpu_mem_used_max_mb",
    "output_dir", "trace_output", "worker0_log", "worker1_log", "gateway_log", "gpu_log",
]
row = {field: summary.get(field, "") for field in fields}
row.update({
    "variant": variant,
    "concurrency": concurrency,
    "max_concurrency": concurrency,
    "output_dir": run_dir,
    "worker0_log": str(Path(run_dir) / "worker0.log"),
    "worker1_log": str(Path(run_dir) / "worker1.log"),
    "gateway_log": str(Path(run_dir) / "gateway.log"),
})
csv_file = Path(csv_path)
csv_file.parent.mkdir(parents=True, exist_ok=True)
with csv_file.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    if handle.tell() == 0:
        writer.writeheader()
    writer.writerow(row)
PY
}

run_one() {
  local variant="$1" concurrency="$2"
  CURRENT_RUN_DIR="${BATCH_DIR}/${variant}_c${concurrency}"
  mkdir -p "$CURRENT_RUN_DIR"
  parse_variant "$variant"
  local l2_path="${L2_PATH_ROOT:-${CURRENT_RUN_DIR}/l2_history}"
  local namespace="agentsociety_${variant}_c${concurrency}_${BATCH_TS}"
  mkdir -p "$l2_path"

  start_worker worker-0 "$WORKER0_PORT" "${GPU_IDS[0]}" "$l2_path" "$namespace" "$CURRENT_RUN_DIR/worker0.log" WORKER0_PID
  wait_for_http "http://127.0.0.1:${WORKER0_PORT}/v1/models" worker-0 "$WORKER0_PID"
  start_worker worker-1 "$WORKER1_PORT" "${GPU_IDS[1]}" "$l2_path" "$namespace" "$CURRENT_RUN_DIR/worker1.log" WORKER1_PID
  wait_for_http "http://127.0.0.1:${WORKER1_PORT}/v1/models" worker-1 "$WORKER1_PID"

  echo "[agentsociety-multi] starting round-robin gateway on port ${GATEWAY_PORT}"
  setsid "$PYTHON_BIN" -m sglang_router.launch_router \
    --worker-urls "http://127.0.0.1:${WORKER0_PORT}" "http://127.0.0.1:${WORKER1_PORT}" \
    --policy round_robin --host "$SERVER_HOST" --port "$GATEWAY_PORT" \
    >"$CURRENT_RUN_DIR/gateway.log" 2>&1 &
  GATEWAY_PID="$!"
  wait_for_http "http://127.0.0.1:${GATEWAY_PORT}/v1/models" gateway "$GATEWAY_PID"

  save_server_info "http://127.0.0.1:${WORKER0_PORT}" "$CURRENT_RUN_DIR/worker0_server_info_before.json"
  save_server_info "http://127.0.0.1:${WORKER1_PORT}" "$CURRENT_RUN_DIR/worker1_server_info_before.json"
  local runner=(
    "$PYTHON_BIN" "$SCRIPT_DIR/run_agentsociety_replay.py"
    --record "$RECORD_PATH"
    --server-url "http://127.0.0.1:${GATEWAY_PORT}/v1"
    --model "$SERVED_MODEL_NAME" --api-key "$API_KEY"
    --mode "$REPLAY_MODE" --max-concurrency "$concurrency"
    --timeout "$TIMEOUT" --print-every "$PRINT_EVERY"
    --output-dir "$CURRENT_RUN_DIR"
  )
  [[ -n "$CALL_TYPES" ]] && runner+=(--call-types "$CALL_TYPES")
  [[ "$USE_RECORDED_TEMPERATURE" != "1" ]] && runner+=(--temperature "$TEMPERATURE")
  [[ -n "$REPLAY_MAX_TOKENS" ]] && runner+=(--max-tokens "$REPLAY_MAX_TOKENS")
  [[ "$variant" != "baseline" ]] && runner+=(--collect-sglang-spec-metrics)

  echo "[agentsociety-multi] replay variant=${variant} concurrency=${concurrency} l2=${USE_L2}"
  start_gpu_sampler "$CURRENT_RUN_DIR/gpu.csv"
  "${runner[@]}" 2>&1 | tee "$CURRENT_RUN_DIR/replay.log"
  stop_gpu_sampler
  save_server_info "http://127.0.0.1:${WORKER0_PORT}" "$CURRENT_RUN_DIR/worker0_server_info_after.json"
  save_server_info "http://127.0.0.1:${WORKER1_PORT}" "$CURRENT_RUN_DIR/worker1_server_info_after.json"
  stop_services
  append_result "$CURRENT_RUN_DIR/summary.json" "$variant" "$concurrency" "$CURRENT_RUN_DIR" "$namespace"
}

main() {
  [[ ${#GPU_IDS[@]} -eq 2 ]] || die "GPU_IDS must contain exactly two IDs, e.g. GPU_IDS='0 1'."
  require_path "$RECORD_PATH"
  require_path "$MODEL_PATH"
  require_path "$CC_BIN"
  require_path "$CXX_BIN"
  command -v "$SGLANG_BIN" >/dev/null 2>&1 || die "SGLANG_BIN is not executable: $SGLANG_BIN"
  "$PYTHON_BIN" -c "import openai, sglang_router" >/dev/null 2>&1 || die "The active environment needs openai and sglang_router."
  mkdir -p "$BATCH_DIR"
  echo "[agentsociety-multi] batch_dir=${BATCH_DIR}"
  echo "[agentsociety-multi] record=${RECORD_PATH}"
  echo "[agentsociety-multi] variants=${VARIANTS[*]} concurrencies=${CONCURRENCIES[*]} gpu_ids=${GPU_IDS[*]}"
  local variant concurrency
  for variant in "${VARIANTS[@]}"; do
    for concurrency in "${CONCURRENCIES[@]}"; do
      run_one "$variant" "$concurrency"
    done
  done
  echo "[agentsociety-multi] results=${RESULTS_CSV}"
}

main "$@"
