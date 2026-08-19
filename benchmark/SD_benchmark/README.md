# Retrieval-Based Speculative Decoding Benchmarks

This directory collects workloads for studying retrieval-based speculative
decoding (SD) across three distinct serving settings. The goal is to measure
serving behavior, especially NGRAM-style SD, rather than to reproduce the
official quality score of every source benchmark.

## Benchmark Scope

| Type | Benchmark | Workload | Form in this repository |
| --- | --- | --- | --- |
| Non-workflow | HumanEval (Chen et al., 2021) | 164 Python code-completion requests | Prompt replay |
| Non-workflow | MT-Bench (Zheng et al., 2023) | 80 two-turn conversations | Prompt replay |
| Non-workflow | Spec-Bench (Xia et al., 2024) | Six general LLM request categories, including chat, code, math, summarization, translation, and RAG | Prompt replay |
| Non-workflow | SPEED-Bench (Abramovich et al., 2026) | Multiple request types, input lengths, text styles, and entropy levels; includes synthetic throughput data | NVIDIA official runner |
| Workflow: problem solving | SWE-bench (Jimenez et al., 2024) | Coding-agent issue repair: inspect a repository, search files, modify code, run tests, and iteratively repair | Public mini-swe-agent trajectory replay |
| Workflow: simulation | AgentSociety (Zhang et al., 2025) | Multi-agent social simulation: agents make plans, reason about needs, choose activities and locations, then update the simulation across time steps | Recorded LLM-call trace replay |

The categories matter for retrieval-based SD. Non-workflow requests are
independent, so any reusable text must be found across unrelated requests or
from an external corpus. Problem-solving workflows have causal agent steps and
growing conversational context. Simulation workloads additionally have many
agents, simulation phases, and time-step dependencies; calls can be similar
within a phase while their submission pattern is constrained by the simulation.

## Runner Map

| Benchmark | One replay / debugging | Single-instance sweep | Two-instance sweep |
| --- | --- | --- | --- |
| HumanEval | `run_benchmark.py --benchmark HumanEval` | `run_benchmark_batch.sh` | `run_multi_instance_benchmark_batch.sh` |
| MT-Bench | `run_benchmark.py --benchmark mt_bench` | `run_benchmark_batch.sh` | `run_multi_instance_benchmark_batch.sh` |
| Spec-Bench | `run_benchmark.py --benchmark spec_bench` | `run_benchmark_batch.sh` | `run_multi_instance_benchmark_batch.sh` |
| SPEED-Bench | NVIDIA official `run.py` | Use the official runner directly | `speed_bench/run_speed_bench_batch.sh` |
| mini-swe-agent SWE-bench trace | `swe_bench/run_swe_trace.py` | `run_benchmark_batch.sh` | `run_multi_instance_benchmark_batch.sh` |
| AgentSociety record | `agentsociety/run_agentsociety_replay.py` | `agentsociety/run_agentsociety_batch.sh` | `agentsociety/run_agentsociety_multi_instance_batch.sh` |

Use the batch runners for comparisons: each point receives a fresh server
(or worker/router pair), so server-side NGRAM state cannot leak across
variants. Use the direct Python runners to inspect one workload point against
an already-running server.

## Included Workloads

### Non-workflow Prompt Replay

`HumanEval/`, `mt_bench/`, and `spec_bench/` convert their requests to OpenAI
chat messages and send them to an OpenAI-compatible SGLang server. They are
useful for conventional serving load and do not run the original benchmark
judges. In particular, HumanEval replay is not an official pass@1 result.

SPEED-Bench remains separate in [`speed_bench/`](speed_bench/README.md). Its
official runner controls request construction and reports its own performance
metrics, so this repository supplies setup and multi-instance integration
instead of a second implementation.

### SWE-bench Workflow Replay

[`swe_bench/`](swe_bench/README.md) uses trajectories from a public
mini-swe-agent SWE-bench submission. A workflow consists of multiple LLM calls:
the agent reasons, emits a bash command, and receives the recorded command
result in a later prompt. Calls within one workflow are therefore submitted in
order; independent workflows may run concurrently.

The replay does **not** create a repository workspace or execute new commands.
It preserves recorded prompts and causal timing while treating the trajectory
as a serving workload. This makes it suitable for SD performance comparisons,
not SWE-bench issue-resolution scoring.

### AgentSociety Simulation Replay

[`agentsociety/`](agentsociety/README.md) contains a portable record extracted
from an AgentSociety simulation. The replay reconstructs the recorded LLM-call
dependency topology across simulation steps and phases: calls from an agent
remain serial, while independent agent chains in the same phase may execute
concurrently.

It deliberately does not rerun simulation state updates, tools, environment
actions, or agent behavior. Later prompts come from the original record. Thus
it measures the serving behavior of an AgentSociety-style multi-agent
simulation workload without requiring an AgentSociety installation or claiming
to semantically re-execute the simulation.

## Running Experiments

### Single SGLang Instance

Use `run_benchmark_batch.sh` for the three prompt workloads and SWE-bench.
The script restarts SGLang for every `(benchmark, variant, concurrency)` point
so NGRAM history does not leak between points.

```bash
BENCHMARKS="mt_bench spec_bench HumanEval" \
CONCURRENCIES="1 4 16" \
VARIANTS="baseline ngram_d4 ngram_d8 ngram_prob_d8" \
MAX_TOTAL_TOKENS=32768 MAX_TOKENS=1024 \
TEMPERATURE=0 TOP_P=1.0 SEED=0 \
SGLANG_ENV=sglang-v0514 \
SGLANG_DIR=/path/to/sglang \
bash SD_benchmark/run_benchmark_batch.sh
```

For an AgentSociety record, use its dedicated launcher:

```bash
bash SD_benchmark/agentsociety/run_agentsociety_batch.sh
```

### Two SGLang Instances

`run_multi_instance_benchmark_batch.sh` runs the prompt workloads through two
workers and a round-robin `sglang_router`. `GPU_IDS="0 1"` places workers on
two GPUs; `GPU_IDS="0 0"` is useful only for a functional one-GPU smoke test.
Use an NGRAM variant ending in `_no_l2` to turn off only shared L2 history.

```bash
MODEL_PATH=/path/to/model \
BENCHMARKS="mt_bench" \
VARIANTS="baseline ngram_d8 ngram_d8_no_l2" \
CONCURRENCIES="1 4 16" \
GPU_IDS="0 1" \
bash SD_benchmark/run_multi_instance_benchmark_batch.sh
```

AgentSociety and SPEED-Bench each have their own multi-instance launchers;
see their linked READMEs above.

For gateway runs, apply
[`patches/sgl_model_gateway_return_meta_info.patch`](patches/sgl_model_gateway_return_meta_info.patch)
to the modified SGLang checkout and rebuild its Python gateway binding. The
patch preserves SGLang's opt-in per-request speculative metadata through the
gateway; without it, accepted-length metrics are unavailable.

## Reading Results

All artifacts are written under `SD_benchmark/outputs/`. A run contains a
per-request JSONL trace, `summary.json`, server logs, and a batch-level CSV.
Compare SD strategies using:

- `wall_time_s` and `completion_tokens_s`: end-to-end performance.
- `latency_p50_s`, `latency_p90_s`, `latency_p99_s`: request latency.
- `spec_draft_accept_length`: accepted draft tokens per verification.
- `spec_accept_length`: the same count plus SGLang's normal target bonus token.
- `spec_accept_rate`: accepted draft tokens divided by proposed draft tokens.

The `spec_*` metrics are global ratios reconstructed from summed raw
per-request SGLang counters, never averages of decode batches or workers.
`baseline` has no speculative counters. For an NGRAM run, non-empty
`spec_verify_ct`, `spec_draft_accept_length`, and `spec_accept_rate` are the
basic metric-integrity check.

## Current Status

Implemented and functionally tested:

- HumanEval, MT-Bench, and Spec-Bench prompt replay.
- Official SPEED-Bench integration through the NVIDIA runner.
- mini-swe-agent SWE-bench trajectory replay.
- AgentSociety causal LLM-call replay.
- Single-instance benchmark launchers and per-request performance traces.
- Two-instance functional tests with the SGLang gateway and optional shared L2
  NGRAM history.

Pending performance evaluation:

- Multi-GPU experiments with workers placed on separate GPUs.
- Large-scale comparisons across workloads, concurrency levels, and NGRAM
  configurations.

## Detailed Guides

- [SWE-bench workflow replay](swe_bench/README.md)
- [AgentSociety simulation replay](agentsociety/README.md)
- [Official SPEED-Bench integration](speed_bench/README.md)
- `run_benchmark_batch.sh` and `run_multi_instance_benchmark_batch.sh` for
  supported environment variables and variant parsing.

## References

- Chen et al. *Evaluating Large Language Models Trained on Code.* arXiv, 2021.
  Introduces HumanEval. [Paper](https://arxiv.org/abs/2107.03374)
- Zheng et al. *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.*
  NeurIPS Datasets and Benchmarks Track, 2023.
  [Proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/91f18a1287b398d378ef22505bf41832-Abstract-Datasets_and_Benchmarks.html)
- Xia et al. *Unlocking Efficiency in Large Language Model Inference: A
  Comprehensive Survey of Speculative Decoding.* Findings of ACL, 2024.
  Introduces Spec-Bench. [Proceedings](https://aclanthology.org/2024.findings-acl.456/)
- Abramovich et al. *SPEED-Bench: A Unified and Diverse Benchmark for
  Speculative Decoding.* arXiv, 2026. This is currently a preprint.
  [Paper](https://arxiv.org/abs/2604.09557)
- Jimenez et al. *SWE-bench: Can Language Models Resolve Real-World GitHub
  Issues?* ICLR, 2024.
  [Proceedings](https://proceedings.iclr.cc/paper_files/paper/2024/hash/edac78c3e300629acfe6cbe9ca88fb84-Abstract-Conference.html)
- Zhang et al. *A Parallelized Framework for Simulating Large-Scale LLM Agents
  with Realistic Environments and Interactions.* ACL Industry Track, 2025.
  This is the published AgentSociety framework reference used here.
  [Proceedings](https://aclanthology.org/2025.acl-industry.94/)

Generated artifacts are ignored by Git.
