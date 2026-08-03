# SD Benchmark Workloads

This directory contains lightweight benchmark replay tools for evaluating
speculative decoding behavior with SGLang. The current focus is performance
workload comparison, not official benchmark scoring.

## Included Workloads

- `HumanEval/`: code-generation prompts converted to chat requests.
- `mt_bench/`: two-turn MT-Bench-style chat prompts.
- `spec_bench/`: multi-category Spec-Bench-style prompts, including writing,
  coding, math, summarization, translation, RAG, and related tasks.
- `swe_bench/`: trace-replay workload built from OpenHands SWE-bench coding
  trajectories.

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

## SWE-Bench OpenHands Trace Replay

`swe_bench/` is a **serving workload**, not an official SWE-Bench evaluation.
It measures the serving cost of agent-style, tool-using LLM requests under
baseline and speculative-decoding configurations. It does not execute tools,
apply patches, run tests, or report SWE-Bench solve rates.

### What Is Replayed

The source is an OpenHands trajectory produced by another model. The converter
extracts every assistant turn as one LLM request and stores its
`workflow_id`, `step_id`, full chat `messages`, recorded output, and source
tool-call metadata.

During replay:

1. Steps in the same `workflow_id` are submitted serially: a later step starts
   only after the preceding replay request returns.
2. Different workflows run concurrently up to `--concurrency`.
3. Each request uses its recorded message history. The model output generated
   during replay is recorded for inspection but is **not** executed and is not
   appended to later requests; later prompts retain the original recorded tool
   results.

This preserves the prompt sizes, tool-result history, and causal arrival
structure of the original agent workload while making the request stream fixed
enough for serving-performance comparison.

### Tools and Valid-Request Filtering

By default, replay sends the three observed OpenHands tools (`think`,
`execute_bash`, and `str_replace_editor`) through the OpenAI-compatible
`tools` field using
[`swe_bench/tool_definitions/tools_schema.json`](swe_bench/tool_definitions/tools_schema.json).
The batch driver starts SGLang with `--tool-call-parser qwen`; generated calls
are saved in `generated_tool_calls`, but never executed.

Some model/server configurations may fail to complete a tool-call JSON for a
small subset of requests. To make performance variants comparable, first run a
reference preflight over the full selected trace and export its failure list.
Then import that same list for every baseline and SD variant. Do not create a
separate list per variant.

```bash
# 1. Reference preflight: run all selected requests and record failures.
BENCHMARKS="swe_bench" \
SWE_TRACE_JSONL=/path/to/openhands_llm_calls.jsonl \
LIMIT=20 MAX_STEPS_PER_WORKFLOW=20 \
CONCURRENCIES="4" VARIANTS="baseline" MAX_TOKENS=512 \
SWE_FAILURE_LIST_OUTPUT=/path/to/failures_qwen_api.jsonl \
SGLANG_ENV=sglang-v059 BENCH_ENV=as \
SGLANG_DIR=/mnt/d/code/sglang AS_DIR=/mnt/d/code/AgentSociety \
bash SD_benchmark/run_benchmark_batch.sh

# 2. Compare variants on the identical valid-request subset.
BENCHMARKS="swe_bench" \
SWE_TRACE_JSONL=/path/to/openhands_llm_calls.jsonl \
LIMIT=20 MAX_STEPS_PER_WORKFLOW=20 \
CONCURRENCIES="1 4 16" VARIANTS="baseline ngram_d4 ngram_d8" MAX_TOKENS=512 \
SWE_SKIP_FAILURE_LIST=/path/to/failures_qwen_api.jsonl \
SGLANG_ENV=sglang-v059 BENCH_ENV=as \
SGLANG_DIR=/mnt/d/code/sglang AS_DIR=/mnt/d/code/AgentSociety \
bash SD_benchmark/run_benchmark_batch.sh
```

The output summary records `source_calls`, `skipped_failure_calls`, tool-call
completion counters, wall time, token throughput, latency percentiles, GPU
utilization, and SGLang speculative metrics. Report the retained-request
coverage with every filtered SWE result. See
[`swe_bench/README.md`](swe_bench/README.md) for trace conversion and runner
arguments.

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
