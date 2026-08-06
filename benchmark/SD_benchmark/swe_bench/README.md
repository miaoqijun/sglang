# Mini-SWE-Agent Trace Replay

This directory turns mini-swe-agent trajectories into a fixed, workflow-style
serving workload for SGLang and speculative decoding experiments. It is not an
official SWE-bench correctness evaluation: the runner does not create a
workspace, execute generated commands, apply patches, or run tests.

## Data Source

The default workload is the public mini-swe-agent submission
`20250803_mini-v1.0.0_qwen2-5-coder-32b-instruct` under the SWE-bench
[experiments repository](https://github.com/SWE-bench/experiments). Submission
metadata is stored in Git; large trajectory artifacts are downloaded by the
repository's `analysis.download_logs` script from the public
`swe-bench-submissions` bucket.

```bash
git clone https://github.com/SWE-bench/experiments.git
cd experiments

conda create -n swe-trace python=3.11 -y
conda activate swe-trace
pip install boto3

python -m analysis.download_logs \
  evaluation/bash-only/20250803_mini-v1.0.0_qwen2-5-coder-32b-instruct \
  --only_trajs --test

python -m analysis.download_logs \
  evaluation/bash-only/20250803_mini-v1.0.0_qwen2-5-coder-32b-instruct \
  --only_trajs
```

The trajectories are downloaded to:

```text
evaluation/bash-only/20250803_mini-v1.0.0_qwen2-5-coder-32b-instruct/trajs/
```

No SWE-bench environment, Docker installation, or mini-swe-agent installation
is needed for conversion or replay.

## Bundled Small Trace

For a quick, reproducible smoke run, this directory includes
`mini_swe_qwen25_coder_32b_50workflows.zip`. It contains
`mini_swe_qwen25_coder_32b_50workflows.jsonl`: all recorded LLM calls from the
first 50 selected `Submitted` workflows, rather than a partial set of calls.

```bash
unzip -o SD_benchmark/swe_bench/mini_swe_qwen25_coder_32b_50workflows.zip \
  -d SD_benchmark/swe_bench
```

Use the full download and conversion procedure below only when regenerating the
trace or changing the selected source trajectories.

## Trace Format

mini-swe-agent uses a plain-text bash protocol rather than OpenAI function
calling:

~~~~text
system:    requires THOUGHT plus exactly one bash command
assistant: THOUGHT: ...
           ```bash
           command
           ```
user:      recorded command return code and output
~~~~

Each assistant turn becomes one LLM request. The next request already contains
the original command result as a later user message. During replay, newly
generated commands are recorded but never executed.

## Convert Trajectories

Only `Submitted` trajectories are retained; interrupted source runs such as
`APIError`, `RetryError`, and `LimitsExceeded` are excluded.

```bash
python SD_benchmark/swe_bench/extract_openhands_traces.py \
  /mnt/d/code/swe-bench/experiments/evaluation/bash-only/20250803_mini-v1.0.0_qwen2-5-coder-32b-instruct/trajs \
  --only-submitted \
  --output SD_benchmark/outputs/swe_bench/mini_swe_qwen25_coder_32b.jsonl
```

The JSONL rows include `call_id`, `workflow_id`, `step_id`, `messages`,
recorded `output`, prompt/output lengths, and source trajectory format/status.

## Replay Semantics

Requests within the same workflow are submitted serially. Different workflows
run concurrently up to `--concurrency`. Each request uses the recorded message
history, so the workload preserves original prompt growth and causal arrival
order while keeping later command results fixed.

Use `SWE_TOOL_MODE=none`: mini-swe-agent commands are ordinary generated text,
not OpenAI tool calls. Do not pass a tool schema or a tool-call parser.

```bash
BENCHMARKS="swe_bench" \
SWE_TRACE_JSONL=/mnt/d/code/AgentSociety/SD_benchmark/swe_bench/mini_swe_qwen25_coder_32b_50workflows.jsonl \
SWE_TOOL_MODE=none \
LIMIT=20 MAX_STEPS_PER_WORKFLOW=20 \
CONCURRENCIES="1 4 16" \
VARIANTS="baseline ngram_d4 ngram_d8" \
MAX_TOTAL_TOKENS=32768 MAX_TOKENS=1024 \
TEMPERATURE=0 TOP_P=1.0 SEED=0 \
SGLANG_ENV=sglang-v059 BENCH_ENV=as \
SGLANG_DIR=/mnt/d/code/sglang AS_DIR=/mnt/d/code/AgentSociety \
bash SD_benchmark/run_benchmark_batch.sh
```

`LIMIT` is the number of workflows, not the number of raw LLM calls.
`MAX_STEPS_PER_WORKFLOW` caps assistant turns within every selected workflow.

## Fixed Valid-Request Subset

If the target model returns request errors or reaches `max_tokens`, first run a
reference preflight over the full selected trace and write a failure list.
Reuse that exact list for baseline and every SD variant; do not create a
different list for each strategy.

```bash
# Preflight.
SWE_FAILURE_LIST_OUTPUT=/path/to/mini_swe_failures.jsonl \
VARIANTS="baseline" \
bash SD_benchmark/run_benchmark_batch.sh

# Matched comparison.
SWE_SKIP_FAILURE_LIST=/path/to/mini_swe_failures.jsonl \
VARIANTS="baseline ngram_d4 ngram_d8" \
bash SD_benchmark/run_benchmark_batch.sh
```

The output summary records request counts, skipped failures, wall time, token
throughput, latency percentiles, GPU utilization, and SGLang speculative
metrics. Report retained-request coverage with every filtered result.

The converter can also read other message-based agent traces, including legacy
OpenHands trajectories, but those may require an explicit tool-calling mode and
are not the default workload documented here.
