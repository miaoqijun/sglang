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
- `agentsociety/`: portable causal replay workload derived from an
  AgentSociety simulation record. It includes its raw LLM-call record and does
  not require an AgentSociety installation.

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
token counts, GPU utilization/memory, and paths to per-run logs. For an NGRAM
run, the authoritative speculative metrics are aggregated from raw counters
returned for each request: `spec_verify_ct`, `spec_num_correct_drafts`, and
`spec_num_proposed_drafts`. The resulting `spec_accept_length` includes the
normal target/bonus token, while `spec_draft_accept_length` counts only
accepted draft tokens; `spec_accept_rate` is accepted draft tokens divided by
proposed draft tokens. These are global ratios of summed raw counters, never
means of worker or decode-batch averages. The single-instance batch runner
also writes `perf_summary.csv/json` with baseline-relative wall-time and
throughput speedups.

## Scripts

`run_benchmark.py` runs one benchmark against an OpenAI-compatible chat server.
It also supports SGLang's teacher-forced trace replay: pass a prior successful
`turn_traces.jsonl` with `--teacher-forcing-trace` and the matching local
`--tokenizer`. The runner encodes each recorded output and sends it as
`custom_params.ngram_teacher_forcing_token_ids`; this requires the corresponding
teacher-forcing support in the SGLang checkout. `--server-urls URL0 URL1 ...`
is an optional client-side round-robin mode for directly addressing workers.
Normal benchmark runs are unchanged unless `--teacher-forcing-trace` is given.

`run_benchmark_batch.sh` starts SGLang, runs one or more benchmarks
sequentially, and restarts the server for each benchmark / variant /
concurrency setting.

`backfill_spec_metrics.py` can recover accepted-length and speculative counter
columns for existing batch results when the run directories already contain
`server_info.json` captured from `/server_info`.

## Multi-Instance Gateway Runs

`run_multi_instance_benchmark_batch.sh` starts two SGLang workers, a
round-robin `sglang_router`, and an optional shared NGRAM L2 history store.
Use an NGRAM variant ending in `_no_l2` to disable only cross-instance history.
Set `GPU_IDS="0 1"` to place the two workers on different GPUs. The older
`GPU_ID=0` form remains supported and defaults to `GPU_IDS="0 0"`. The
per-run `gpu.csv` samples each distinct selected GPU and `batch_results.csv`
records the worker GPU pair.
The portable AgentSociety counterpart is
`agentsociety/run_agentsociety_multi_instance_batch.sh`; it preserves the
recorded AgentSociety call dependencies while using the same worker/router/L2
deployment pattern.

The router's typed OpenAI request model otherwise drops SGLang's
`return_meta_info` extension. Apply the bundled patch once in the coauthor's
SGLang checkout and rebuild the Python gateway binding before collecting
accepted-length metrics:

```bash
cd /path/to/sglang

git apply --check --directory=sgl-model-gateway \
  /path/to/AgentSociety/SD_benchmark/patches/sgl_model_gateway_return_meta_info.patch
git apply --directory=sgl-model-gateway \
  /path/to/AgentSociety/SD_benchmark/patches/sgl_model_gateway_return_meta_info.patch

cd sgl-model-gateway/bindings/python
RUSTUP_TOOLCHAIN=stable maturin develop --features vendored-openssl
```

The benchmark runner sends the opt-in header only for NGRAM variants; the
patched router converts it to `return_meta_info: true` before forwarding to a
worker. A correct run has non-empty `spec_metric_turns`, `spec_verify_ct`,
`spec_accept_length`, and `spec_accept_rate` in both `summary.json` and
`batch_results.csv`. Do not use the older `accept_len_mean` or `/server_info`
fields for multi-instance comparisons.

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
