#!/usr/bin/env bash
set -euo pipefail

# Portable single-server AgentSociety replay sweep. The bundled raw record and
# runner are self-contained; only an installed SGLang server environment and a
# model path are external requirements.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SGLANG_ENV="${SGLANG_ENV:-sglang-v059}"
RUNNER_ENV="${RUNNER_ENV:-${SGLANG_ENV}}"
SGLANG_DIR="${SGLANG_DIR:-/mnt/d/code/sglang}"
MODEL_PATH="${MODEL_PATH:-/mnt/d/code/Qwen2.5-14B-Instruct-AWQ}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${MODEL_PATH}")}"
RECORD_PATH="${RECORD_PATH:-${SCRIPT_DIR}/records/agentsociety_record.jsonl}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-1919}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:${PORT}/v1}"
API_KEY="${API_KEY:-dummy}"
MAX_TOTAL_TOKENS="${MAX_TOTAL_TOKENS:-32768}"
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
SPEC_METRICS_LOG_ON_SHUTDOWN="${SPEC_METRICS_LOG_ON_SHUTDOWN:-1}"
GPU_ID="${GPU_ID:-0}"
EXTRA_SERVER_ARGS_STR="${EXTRA_SERVER_ARGS:-}"
read -r -a EXTRA_SERVER_ARGS <<< "${EXTRA_SERVER_ARGS_STR}"

BATCH_TS="$(date +%Y%m%d_%H%M%S)"
BATCH_ROOT="${BATCH_ROOT:-${SCRIPT_DIR}/../outputs/agentsociety}"
BATCH_DIR="${BATCH_DIR:-${BATCH_ROOT}/${BATCH_TS}}"
RESULTS_CSV="${RESULTS_CSV:-${BATCH_DIR}/batch_results.csv}"
SERVER_PID=""
GPU_SAMPLER_PID=""
SPEC_ARGS=()

die() { echo "ERROR: $*" >&2; exit 1; }

run_in_env() { conda run --no-capture-output -n "$1" "$@"; }

stop_server() {
  [[ -n "$SERVER_PID" ]] || return 0
  if kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM -- "-$SERVER_PID" 2>/dev/null || kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "$SERVER_PID" 2>/dev/null || break
      sleep 1
    done
    kill -0 "$SERVER_PID" 2>/dev/null && kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
  fi
  SERVER_PID=""
}
stop_gpu_sampler() {
  [[ -n "$GPU_SAMPLER_PID" ]] || return 0
  kill "$GPU_SAMPLER_PID" 2>/dev/null || true
  wait "$GPU_SAMPLER_PID" 2>/dev/null || true
  GPU_SAMPLER_PID=""
}

cleanup() {
  stop_gpu_sampler
  stop_server
}
trap cleanup EXIT INT TERM

start_gpu_sampler() {
  local output="$1"
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  (
    echo "timestamp,gpu_util_percent,mem_used_mb,mem_total_mb"
    while true; do
      nvidia-smi -i "$GPU_ID" \
        --query-gpu=timestamp,utilization.gpu,memory.used,memory.total \
        --format=csv,noheader,nounits 2>/dev/null || true
      sleep 1
    done
  ) >"$output" &
  GPU_SAMPLER_PID="$!"
}

wait_for_server() {
  local started="$(date +%s)"
  while true; do
    if curl -fsS -H "Authorization: Bearer ${API_KEY}" "${SERVER_URL}/models" >/dev/null 2>&1; then
      return 0
    fi
    kill -0 "$SERVER_PID" 2>/dev/null || die "Server exited before becoming ready; see ${CURRENT_RUN_DIR}/server.log"
    (( $(date +%s) - started <= 900 )) || die "Timed out waiting for ${SERVER_URL}"
    sleep 1
  done
}

parse_variant() {
  local value="$1"
  SPEC_ARGS=()
  [[ "$value" == "baseline" ]] && return 0
  local implied draft
  if [[ "$value" =~ ^(ngram|ngram_bfs)_d([0-9]+)$ ]]; then
    implied="BFS"; draft="${BASH_REMATCH[2]}"
  elif [[ "$value" =~ ^(ngram_prob|prob)_d([0-9]+)$ ]]; then
    implied="PROB"; draft="${BASH_REMATCH[2]}"
  else
    die "Unsupported VARIANT=${value}; use baseline, ngram_d<N>, or ngram_prob_d<N>."
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
  if (( ${#NGRAM_EXTRA_ARGS[@]} > 0 )); then SPEC_ARGS+=("${NGRAM_EXTRA_ARGS[@]}"); fi
}

start_server() {
  local variant="$1"
  parse_variant "$variant"
  local spec_metrics_args=()
  if [[ "$SPEC_METRICS_LOG_ON_SHUTDOWN" == "1" ]]; then
    spec_metrics_args+=(--spec-metrics-log-on-shutdown)
  fi
  echo "[agentsociety-batch] starting server variant=${variant}"
  setsid bash -c '
    set -euo pipefail
    cd "$1"
    export SGLANG_NGRAM_FORCE_GREEDY_VERIFY="$2"
    shift 2
    exec conda run --no-capture-output -n "$1" python -m sglang.launch_server \
      --model-path "$2" --host "$3" --port "$4" --served-model-name "$5" --api-key "$6" \
      --max-total-tokens "$7" "${@:8}"
  ' _ "$SGLANG_DIR" "$FORCE_GREEDY_VERIFY" "$SGLANG_ENV" "$MODEL_PATH" "$HOST" "$PORT" \
      "$SERVED_MODEL_NAME" "$API_KEY" "$MAX_TOTAL_TOKENS" \
      "${SPEC_ARGS[@]}" \
      "${spec_metrics_args[@]}" \
      "${EXTRA_SERVER_ARGS[@]}" >"${CURRENT_RUN_DIR}/server.log" 2>&1 &
  SERVER_PID="$!"
  wait_for_server
}

append_result() {
  local summary="$1" variant="$2" concurrency="$3" run_dir="$4"
  "$PYTHON_BIN" - "$summary" "$RESULTS_CSV" "$variant" "$concurrency" "$run_dir" <<'PY'
import csv, json, sys
from pathlib import Path
summary_path, csv_path, variant, concurrency, run_dir = sys.argv[1:]
summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
gpu_path = Path(run_dir) / "gpu.csv"
gpu_util, gpu_memory = [], []
if gpu_path.exists():
    with gpu_path.open(encoding="utf-8", errors="replace") as handle:
        for item in csv.DictReader(handle):
            try:
                gpu_util.append(float(item["gpu_util_percent"]))
                gpu_memory.append(float(item["mem_used_mb"]))
            except (KeyError, TypeError, ValueError):
                pass
gpu = {
    "gpu_util_avg": round(sum(gpu_util) / len(gpu_util), 6) if gpu_util else None,
    "gpu_util_max": max(gpu_util) if gpu_util else None,
    "gpu_mem_used_max_mb": max(gpu_memory) if gpu_memory else None,
    "gpu_log": str(gpu_path),
}
summary.update({"deployment": "single_server", **gpu})
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
fields = ["benchmark", "variant", "concurrency", "max_concurrency", "deployment", "wall_time_s", "requests_s", "turns", "errors", "prompt_tokens", "completion_tokens", "total_tokens", "completion_tokens_s", "total_tokens_s", "latency_avg_s", "latency_p50_s", "latency_p90_s", "latency_p99_s", "spec_metric_turns", "spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts", "spec_accept_length", "spec_draft_accept_length", "spec_accept_rate", "gpu_util_avg", "gpu_util_max", "gpu_mem_used_max_mb", "output_dir", "trace_output", "server_log", "gpu_log"]
row = {key: summary.get(key, "") for key in fields}
row.update({"variant": variant, "concurrency": concurrency, "max_concurrency": concurrency, "output_dir": run_dir, "server_log": str(Path(run_dir) / "server.log")})
path = Path(csv_path); path.parent.mkdir(parents=True, exist_ok=True)
with path.open("a", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fields)
    if handle.tell() == 0: writer.writeheader()
    writer.writerow(row)
PY
}

run_one() {
  local variant="$1" concurrency="$2"
  CURRENT_RUN_DIR="${BATCH_DIR}/${variant}_c${concurrency}"
  mkdir -p "$CURRENT_RUN_DIR"
  start_server "$variant"
  local runner=(python "$SCRIPT_DIR/run_agentsociety_replay.py" --record "$RECORD_PATH" --server-url "$SERVER_URL" --model "$SERVED_MODEL_NAME" --api-key "$API_KEY" --mode "$REPLAY_MODE" --max-concurrency "$concurrency" --timeout "$TIMEOUT" --print-every "$PRINT_EVERY" --output-dir "$CURRENT_RUN_DIR")
  if [[ -n "$CALL_TYPES" ]]; then runner+=(--call-types "$CALL_TYPES"); fi
  if [[ "$USE_RECORDED_TEMPERATURE" != "1" ]]; then runner+=(--temperature "$TEMPERATURE"); fi
  if [[ -n "$REPLAY_MAX_TOKENS" ]]; then runner+=(--max-tokens "$REPLAY_MAX_TOKENS"); fi
  if [[ "$variant" != "baseline" ]]; then runner+=(--collect-sglang-spec-metrics); fi
  echo "[agentsociety-batch] replay variant=${variant} concurrency=${concurrency}"
  start_gpu_sampler "$CURRENT_RUN_DIR/gpu.csv"
  run_in_env "$RUNNER_ENV" "${runner[@]}" 2>&1 | tee "$CURRENT_RUN_DIR/replay.log"
  stop_gpu_sampler
  stop_server
  append_result "$CURRENT_RUN_DIR/summary.json" "$variant" "$concurrency" "$CURRENT_RUN_DIR"
}

main() {
  [[ -f "$RECORD_PATH" ]] || die "RECORD_PATH does not exist: $RECORD_PATH"
  [[ -d "$SGLANG_DIR" ]] || die "SGLANG_DIR does not exist: $SGLANG_DIR"
  [[ -e "$MODEL_PATH" ]] || die "MODEL_PATH does not exist: $MODEL_PATH"
  mkdir -p "$BATCH_DIR"
  echo "[agentsociety-batch] batch_dir=${BATCH_DIR}"
  echo "[agentsociety-batch] record=${RECORD_PATH}"
  echo "[agentsociety-batch] variants=${VARIANTS[*]} concurrencies=${CONCURRENCIES[*]}"
  local variant concurrency
  for variant in "${VARIANTS[@]}"; do
    for concurrency in "${CONCURRENCIES[@]}"; do
      run_one "$variant" "$concurrency"
    done
  done
  echo "[agentsociety-batch] results=${RESULTS_CSV}"
}

main "$@"
