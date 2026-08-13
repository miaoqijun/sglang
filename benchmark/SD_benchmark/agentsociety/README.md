# AgentSociety Replay Workload

This directory is a portable AgentSociety serving workload for speculative
decoding experiments. It includes the raw recorded LLM-call trace and a replay
runner; it does not import, start, or otherwise depend on an AgentSociety
simulation installation.

## Included Files

- `records/agentsociety_record.jsonl`: raw AgentSociety record from
  `records-local-4` (6 simulation steps, 1743 LLM requests).
- `run_agentsociety_replay.py`: causal replay runner and per-request trace
  writer. Requires only `openai`.
- `run_agentsociety_batch.sh`: single-SGLang-server batch launcher for
  baseline/NGRAM comparisons, including GPU sampling.
- `run_agentsociety_multi_instance_batch.sh`: two-worker gateway launcher for
  shared-L2 NGRAM experiments.

The record is the source workload. It preserves every call's `messages`,
request settings, agent ID, simulation step, phase, and agent-local sequence.
The runner rebuilds only the original LLM-call dependency topology:

```text
simulation step -> pre_dispatch -> main -> post_intercept
```

Calls from a given agent remain serial. Independent agent chains within a
phase run concurrently. The runner never executes AgentSociety state updates,
tools, environment actions, or a new simulation; each later request uses its
already-recorded prompt. Thus this is an LLM serving replay workload, not a
semantic re-execution of the simulation.

This matches the original legacy AgentSociety **LLM replay** semantics,
including its faithful/aggressive scheduling modes and latency boundary
(latency begins after acquiring the replay concurrency slot). It deliberately
does not claim to re-execute the full AgentSociety simulation semantics.

## Dependency

Install the sole runner dependency in the environment used to submit replay
requests:

```bash
python -m pip install openai
```

## One Replay Against an Existing Server

With a running OpenAI-compatible SGLang server:

```bash
python SD_benchmark/agentsociety/run_agentsociety_replay.py \
  --server-url http://127.0.0.1:1919/v1 \
  --model Qwen2.5-14B-Instruct \
  --api-key dummy \
  --mode faithful \
  --max-concurrency 16 \
  --output-dir SD_benchmark/outputs/agentsociety/smoke_c16
```

The direct runner preserves request parameters from the record. Add
`--temperature 0` or `--max-tokens 512` only when intentionally overriding
them. In contrast, both batch launchers default to `TEMPERATURE=0` because SD
performance comparisons need deterministic sampling. Set
`USE_RECORDED_TEMPERATURE=1` to recover record-level temperatures for a
semantic replay check.
To inspect only one template class without changing the stored trace, use:

```bash
--call-types plan_block.DETAILED_PLAN_PROMPT
```

The output directory contains `replay_trace.jsonl` with the newly generated
output, token usage, latency and raw SGLang speculative counters per call, plus
`summary.json`.

## Batch SGLang Experiment

The batch launcher starts a fresh server for every `(variant, concurrency)`
combination, runs the portable replay, then stops it. No NGRAM history leaks
between points.

```bash
SGLANG_ENV=sglang-v059 \
RUNNER_ENV=as \
SGLANG_DIR=/mnt/d/code/sglang \
MODEL_PATH=/mnt/d/code/Qwen2.5-7B-Instruct-AWQ \
SERVED_MODEL_NAME=Qwen2.5-7B-Instruct-AWQ \
VARIANTS="baseline ngram_d4 ngram_d8 ngram_prob_d8" \
CONCURRENCIES="1 4 16 32" \
MAX_TOTAL_TOKENS=32768 \
TEMPERATURE=0 \
bash SD_benchmark/agentsociety/run_agentsociety_batch.sh
```

`ngram_d<N>` uses BFS and `ngram_prob_d<N>` uses PROB. Set
`NGRAM_MATCH_TYPE` to override the variant-implied strategy; use
`NGRAM_MAX_TRIE_DEPTH`, `NGRAM_BFS_BREADTH`, `NGRAM_CAPACITY`, and
`NGRAM_EXTRA_ARGS` for branch-specific SGLang sweeps. Outputs go under
`SD_benchmark/outputs/agentsociety/<timestamp>/`, with `batch_results.csv`
aggregating only raw request-level speculative counters, never worker/decode-
batch averages.

Each run also writes `gpu.csv`; its `summary.json` and `batch_results.csv`
contain `gpu_util_avg`, `gpu_util_max`, and `gpu_mem_used_max_mb`. The
speculative fields remain global ratios reconstructed from raw request-level
counters, not worker or decode-batch means.

## Two-Worker Gateway Experiment

`run_agentsociety_multi_instance_batch.sh` uses the same causal replay runner,
but starts two SGLang workers and a round-robin `sglang_router`. NGRAM runs use
a fresh shared L2 namespace per point; append `_no_l2` to an NGRAM variant to
leave cross-instance history disabled while keeping local NGRAM enabled.

Before collecting speculative counters through the gateway, apply and build
the `return_meta_info` gateway patch described in the parent
[`README`](../README.md#multi-instance-gateway-runs). A correct NGRAM point
has non-empty `spec_verify_ct`, `spec_accept_length`, and
`spec_accept_rate` in its `summary.json`.

Run from the activated coauthor SGLang/gateway environment:

```bash
MODEL_PATH=$HOME/swq/models/Qwen2.5-14B-Instruct \
VARIANTS="baseline ngram_d8 ngram_d8_no_l2 ngram_prob_d8" \
CONCURRENCIES="1 4 16" \
GPU_IDS="0 0" \
bash SD_benchmark/agentsociety/run_agentsociety_multi_instance_batch.sh
```

`GPU_IDS` is a pair, one ID per worker. `GPU_IDS="0 1"` runs workers on two
different GPUs; `GPU_IDS="0 0"` is only a functional two-worker setup on one
GPU. Outputs go under `SD_benchmark/outputs/agentsociety/<timestamp>/`, just
like single-server replay runs; use the `deployment` field to distinguish the
two-worker gateway configuration.
