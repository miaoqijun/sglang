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
SGLANG_ENV=sglang-v0514 \
SGLANG_DIR=/mnt/d/code/sglang \
bash SD_benchmark/run_benchmark_batch.sh
```

`LIMIT` is the number of workflows, not the number of raw LLM calls.
`MAX_STEPS_PER_WORKFLOW` caps assistant turns within every selected workflow.

## Teacher-Forcing Reference

Build a full target-output reference from the bundled mini-swe-agent trace:

```bash
python SD_benchmark/swe_bench/build_teacher_forcing_trace.py \
  --trace-jsonl SD_benchmark/swe_bench/mini_swe_qwen25_coder_32b_50workflows.jsonl \
  --output SD_benchmark/swe_bench/mini_swe_teacher_forcing.jsonl
```

The reference contains the source `output` plus `workflow_id`, `step_id`, and
`call_id`. With no selection arguments, it contains all 50 bundled workflows
and all 1091 LLM calls. `--limit`, `--max-steps-per-workflow`, and
`--skip-failure-list` are available only when an intentionally matched subset
is needed. Pass this file to `run_swe_trace.py` with
`--teacher-forcing-trace` and the serving model's `--tokenizer`.

For a direct single-instance NGRAM server on `:19191`:

```bash
python SD_benchmark/swe_bench/run_swe_trace.py \
  --trace-jsonl SD_benchmark/swe_bench/mini_swe_qwen25_coder_32b_50workflows.jsonl \
  --server-url http://127.0.0.1:19191/v1 \
  --model Qwen2.5-14B-Instruct \
  --temperature 0 --top-p 1 --seed 0 \
  --max-tokens 1024 --concurrency 1 \
  --tool-mode none \
  --teacher-forcing-trace SD_benchmark/swe_bench/mini_swe_teacher_forcing.jsonl \
  --tokenizer /path/to/Qwen2.5-14B-Instruct \
  --collect-sglang-spec-metrics \
  --output-dir SD_benchmark/outputs/swe_bench/teacher_forcing_single_instance
```

Teacher forcing currently requires a direct worker URL. The multi-instance
gateway must additionally preserve `custom_params` before it can forward the
teacher token ids.

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
