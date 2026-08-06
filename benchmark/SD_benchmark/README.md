# SD Benchmark Workloads

This directory contains lightweight benchmark replay tools for evaluating
speculative decoding behavior with SGLang. The current focus is performance
workload comparison, not official benchmark scoring.

## Included Workloads

- `HumanEval/`: code-generation prompts converted to chat requests.
- `mt_bench/`: two-turn MT-Bench-style chat prompts.
- `spec_bench/`: multi-category Spec-Bench-style prompts, including writing,
  coding, math, summarization, translation, RAG, and related tasks.
- `swe_bench/`: trace-replay workload built from mini-swe-agent SWE-bench
  coding trajectories.

The runner records full request/response traces so speculative decoding
behavior can be analyzed later.

Each JSONL trace row contains:

- `request_id`
- `benchmark`
- `question_id`
- `turn_id`
- `category`
- full `messages`
- model `output`
- `prompt_tokens`, `completion_tokens`, `total_tokens`
- `latency_s`
- request parameters such as model, temperature, top-p, seed, and max tokens

For batch runs, `batch_results.csv` summarizes throughput, latency percentiles,
token counts, GPU utilization/memory, paths to per-run logs, and
`accept_len_mean` when the local SGLang build exposes it through
`/server_info`. This accepted-length field follows the SPEED-Bench-style
convention: average output tokens advanced per decode/verify step, including
the normal target token. For speculative runs, the batch output also includes
true speculative counters from SGLang's `SpecMetrics`, such as
`spec_true_mean_accept_len`, `spec_true_accept_rate`, and
`spec_zero_accept_ratio`. These fields are not derived from per-batch log
averages. The batch runner also writes
`perf_summary.csv/json` with baseline-relative wall-time and throughput
speedups.

## Scripts

`run_benchmark.py` runs one benchmark against an OpenAI-compatible chat server.

`run_benchmark_batch.sh` starts SGLang, runs one or more benchmarks
sequentially, and restarts the server for each benchmark / variant /
concurrency setting.

`backfill_spec_metrics.py` can recover accepted-length and speculative counter
columns for existing batch results when the run directories already contain
`server_info.json` captured from `/server_info`.

## Mini-SWE-Agent Trace Replay

`swe_bench/` is a **serving workload**, not an official SWE-Bench evaluation.
It replays public mini-swe-agent trajectories as causal, multi-turn coding
workflows. Each assistant turn contains plain text (`THOUGHT` plus one bash
command), and the next prompt contains the original recorded command result.

During replay, workflow steps are serial, workflows may run concurrently, and
generated commands are recorded but not executed. Set `SWE_TOOL_MODE=none`:
this workload does not use OpenAI tool calls, schemas, or a tool-call parser.
The conversion command, data download procedure, preflight filtering protocol,
and full batch command are in [`swe_bench/README.md`](swe_bench/README.md).

Supported variants include:

- `baseline`
- `ngram_d<N>` or `ngram_bfs_d<N>`
- `ngram_prob_d<N>` or `prob_d<N>`
- `ngram_backmatch_d<N>` or `backmatch_d<N>`
- `ngram_prob_backmatch_d<N>` or `prob_backmatch_d<N>`
- variants without explicit `d<N>`, such as `prob_backmatch`, use
  `NGRAM_DRAFT_TOKENS` (default: `8`)

SPEED-Bench is evaluated with the official NVIDIA Model Optimizer runner and
is not vendored in this repository. See the
[official SpecDec-Bench example](https://github.com/NVIDIA/TensorRT-Model-Optimizer/tree/main/examples/specdec_bench).

## Example

```bash
BENCHMARKS="mt_bench spec_bench HumanEval" \
CONCURRENCIES="1" \
VARIANTS="baseline ngram_d8 prob_backmatch" \
MAX_TOTAL_TOKENS=32768 \
MAX_TOKENS=1024 \
TEMPERATURE=0 \
TOP_P=1.0 \
SEED=0 \
SGLANG_ENV=sglang-v059 \
BENCH_ENV=as \
SGLANG_DIR=/mnt/d/code/sglang \
AS_DIR=/mnt/d/code/AgentSociety \
bash SD_benchmark/run_benchmark_batch.sh
```

Example with different NGRAM draft lengths:

```bash
BENCHMARKS="mt_bench" \
CONCURRENCIES="1 4" \
VARIANTS="baseline ngram_d4 ngram_d8 ngram_prob_backmatch_d16" \
MAX_TOTAL_TOKENS=32768 \
MAX_TOKENS=1024 \
TEMPERATURE=0 \
TOP_P=1.0 \
SEED=0 \
SGLANG_ENV=sglang-v059 \
BENCH_ENV=as \
SGLANG_DIR=/mnt/d/code/sglang \
AS_DIR=/mnt/d/code/AgentSociety \
bash SD_benchmark/run_benchmark_batch.sh
```

Outputs are written under `SD_benchmark/outputs/`, which is intentionally
ignored by git.

## Notes

These tools are intended for serving-performance experiments. They do not run
official quality evaluation for MT-Bench, Spec-Bench, or HumanEval. In
particular, the HumanEval chat prompt may produce fenced or full-function code,
so it should not be treated as an official pass@1 evaluation without additional
post-processing.
