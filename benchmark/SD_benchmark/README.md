# SD Benchmark Workloads

This directory contains lightweight benchmark replay tools for evaluating
speculative decoding behavior with SGLang. The current focus is performance
workload comparison, not official benchmark scoring.

## Included Workloads

- `HumanEval/`: code-generation prompts converted to chat requests.
- `mt_bench/`: two-turn MT-Bench-style chat prompts.
- `spec_bench/`: multi-category Spec-Bench-style prompts, including writing,
  coding, math, summarization, translation, RAG, and related tasks.

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

## Scripts

`run_benchmark.py` runs one benchmark against an OpenAI-compatible chat server.

`run_benchmark_batch.sh` starts SGLang, runs one or more benchmarks
sequentially, and restarts the server for each benchmark / variant /
concurrency setting.

Supported variants include:

- `baseline`
- `ngram_d4`
- `ngram_d8`
- `prob_backmatch`

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

Outputs are written under `SD_benchmark/outputs/`, which is intentionally
ignored by git.

## Notes

These tools are intended for serving-performance experiments. They do not run
official quality evaluation for MT-Bench, Spec-Bench, or HumanEval. In
particular, the HumanEval chat prompt may produce fenced or full-function code,
so it should not be treated as an official pass@1 evaluation without additional
post-processing.
