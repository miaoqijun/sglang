# SWE-Bench Trajectory Traces

This folder contains utilities for using downloaded OpenHands SWE-bench
trajectories as a workflow problem-solving workload. The trajectories are not
vendored here; keep them in an external data directory and convert them to
JSONL traces before replay or offline analysis.

## Trajectory Data Source

Trajectory metadata comes from the SWE-bench
[experiments repository](https://github.com/SWE-bench/experiments/tree/main/evaluation/verified).
The repository contains verified-submission directories, while the large
`trajs/` and `logs/` artifacts are fetched separately by its download script
from the public `swe-bench-submissions` bucket.

The current trace was extracted from the OpenHands submission
`evaluation/verified/20250716_openhands_kimi_k2`, which contains one public
run over 500 SWE-bench Verified tasks. No OpenHands installation, Docker
environment, or SWE-bench execution is needed for trace extraction.

```bash
git clone https://github.com/SWE-bench/experiments.git
cd experiments

conda create -n swe-trace python=3.11 -y
conda activate swe-trace
pip install boto3

# Optional: list available OpenHands submissions.
find evaluation/verified -mindepth 1 -maxdepth 1 -type d -iname "*openhands*" | sort

# Verify that trajectories are available, then download them.
python -m analysis.download_logs \
  evaluation/verified/20250716_openhands_kimi_k2 \
  --only_trajs --test

python -m analysis.download_logs \
  evaluation/verified/20250716_openhands_kimi_k2 \
  --only_trajs
```

The download creates
`evaluation/verified/20250716_openhands_kimi_k2/trajs/`. This benchmark uses
those trajectories only as a recorded serving workload; it does not reproduce
the original agent execution or SWE-bench correctness evaluation.

The converter treats each assistant message as one recorded LLM call. All
messages before that assistant turn are formatted as the prompt context, and
the assistant message text plus tool calls are saved as the recorded output.

## Convert OpenHands Trajectories

Example:

```bash
python SD_benchmark/swe_bench/extract_openhands_traces.py \
  /mnt/d/code/swe-bench/experiments/evaluation/verified/20250716_openhands_kimi_k2/trajs \
  --output SD_benchmark/outputs/swe_bench/openhands_kimi_k2_llm_calls.jsonl
```

Each JSONL row contains:

- `call_id`
- `workflow_id`
- `step_id`
- `message_index`
- `prompt`
- `output`
- `messages`
- `tool_calls`
- `tool_names`
- `has_tool_call`
- `prompt_message_count`
- `prompt_char_length`
- `output_char_length`

`call_id` is a global numeric LLM-call id. `workflow_id` is a numeric
trajectory id assigned by sorted input file order. `step_id` is the assistant
turn id inside that workflow.

For assistant turns that call tools, `output` includes both the assistant text
and a serialized `<tool_calls>` block, because the tool call is part of the
model output for serving-workload purposes. The parsed tool calls are also kept
separately in `tool_calls`.

The normalized prompt message list is saved as `messages` by default so replay
can preserve original roles. Use `--no-messages` only when you intentionally
want a smaller text-only trace.

This is a serving-workload trace, not an official SWE-bench correctness runner.
It does not clone repositories, execute tools, apply patches, or run tests.

## Replay Converted Traces

Run against an already-running OpenAI-compatible server:

```bash
python SD_benchmark/swe_bench/run_swe_trace.py \
  --trace-jsonl SD_benchmark/outputs/swe_bench/openhands_kimi_k2_llm_calls.jsonl \
  --server-url http://127.0.0.1:1919/v1 \
  --model Qwen2.5-7B-Instruct-AWQ \
  --temperature 0 \
  --top-p 1.0 \
  --seed 0 \
  --max-tokens 1024 \
  --concurrency 1 \
  --output-dir SD_benchmark/outputs/swe_bench_replay/smoke
```

Or let the shared batch driver start and stop SGLang for each variant:

```bash
BENCHMARKS="swe_bench" \
SWE_TRACE_JSONL=/mnt/d/code/AgentSociety/SD_benchmark/outputs/swe_bench/openhands_kimi_k2_llm_calls.jsonl \
CONCURRENCIES="1 4" \
VARIANTS="baseline ngram_d4 ngram_d8 ngram_prob_backmatch_d8" \
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

The replay summary includes wall time, request throughput, token throughput,
latency p50/p90/p99, and `accept_len_mean` when the SGLang build exposes it
through `/server_info`. For speculative runs, the summary also includes true
SGLang `SpecMetrics` counters such as `spec_true_mean_accept_len`,
`spec_true_accept_rate`, and `spec_zero_accept_ratio`; these are not computed
from per-batch decode log averages.

`--limit` in the SWE runner means number of workflows/tasks, not number of raw
LLM-call rows. Requests inside the same workflow are sent sequentially: step 1
is issued only after step 0 returns. Different workflows may run concurrently
up to `--concurrency`.

Use `--max-steps-per-workflow N` to replay only the first `N` LLM calls from
each selected workflow while preserving causal order. This is useful for
avoiding very long late-trajectory requests that exceed the server context
limit.

### Reuse a fixed valid-request subset

Run a preflight once with `--failure-list-output`. The JSONL contains requests
that error, reach `max_tokens`, or fail to complete a source tool call. Pass
the same file with `--skip-failure-list` in later variants so they replay the
same filtered workload. Filtering removes only that request; later trace steps
continue to use their recorded histories.

```bash
python SD_benchmark/swe_bench/run_swe_trace.py ... \
  --failure-list-output SD_benchmark/outputs/swe_bench/failures_qwen_api.jsonl

python SD_benchmark/swe_bench/run_swe_trace.py ... \
  --skip-failure-list SD_benchmark/outputs/swe_bench/failures_qwen_api.jsonl
```

For output behavior, the runner uses the recorded `messages` field when present
instead of wrapping the whole formatted prompt as a single user message. This is
closer to the original OpenHands chat structure. If the converted JSONL was
created with an older converter and has no `messages`, the runner falls back to
the text `prompt` field.

The generated `output` field is the response from the model under test and is
saved for inspection only. It is not appended to later workflow requests. For
trace-based datastore construction or offline SD analysis, use `recorded_output`
from the original trajectory.

By default replay uses
`swe_bench/tool_definitions/tools_schema.json` as the OpenAI-compatible
`tools` request field and sends `tool_choice=required`. The batch driver starts
SGLang with `--tool-call-parser qwen` for this mode, so generated calls are
recorded in `generated_tool_calls` rather than only as unstructured text. The
runner still does not execute these calls; it uses the recorded tool results in
later prompts. Use `SWE_TOOL_MODE=plain` to append
`openhands_tools_plain_prompt.md` to the system message for an explicit
plain-text comparison, or `SWE_TOOL_MODE=none` for chat-only replay.
