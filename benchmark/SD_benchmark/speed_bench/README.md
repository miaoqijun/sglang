# Official SPEED-Bench With Multi-Instance SGLang

This directory adapts the official NVIDIA Model Optimizer
[SpecDec-Bench example](https://github.com/NVIDIA/Model-Optimizer/tree/main/examples/specdec_bench)
for a running multi-instance SGLang gateway. It does not replace the official
runner or reimplement SPEED-Bench.

The adapted path keeps the official `run.py`, `SPEEDBench` dataset class,
category-stratified request selection, tokenizer/chat-template construction,
causal multi-turn execution, `SimpleRunner`, `SpecBench` metric, and Timing
metric. Only the inference backend changes: `SGLANG_REMOTE` sends the official
runner's already-tokenized `input_ids` to the gateway's native `/generate`
endpoint.

## 1. Get the Official Runner

On the cluster or a Linux machine:

```bash
git clone --depth 1 https://github.com/NVIDIA/Model-Optimizer.git model-optimizer-specbench
cd model-optimizer-specbench/examples/specdec_bench

conda create -n speedbench python=3.10 -y
conda activate speedbench
python -m pip install -r requirements.txt
python -m pip install sglang transformers torch
```

The official repository and its dataset preparation procedure are the source of
truth. Do not commit its prepared data or generated outputs into AgentSociety.

## 2. Prepare Official SPEED-Bench Data

The first successful small configuration is usually `throughput_1k`:

```bash
cd /path/to/model-optimizer-specbench/examples/specdec_bench
python prepare_data.py --dataset speed --config throughput_1k
```

Expected output:

```text
Saved to data/speed/throughput_1k/test.parquet
```

Other official configurations are prepared the same way. `throughput_32k`
contains long prompt workloads and needs a correspondingly large worker context
and KV-cache budget.

## 3. Apply the Original In-Process SGLang Compatibility Fixes

These two edits are for the official `--engine SGLANG` in-process backend. They
are not used by `SGLANG_REMOTE`, but retain the previously working single
instance command and avoid the two incompatibilities encountered with the
coauthor SGLang branch.

From the official `examples/specdec_bench` directory:

```bash
sed -i \
  -e 's/speculative_algorithm = "LOOKAHEAD"/speculative_algorithm = "NGRAM"/' \
  -e 's/"cuda_graph_max_bs": max_concurrent_requests,/"cuda_graph_max_bs_decode": max_concurrent_requests,/' \
  specdec_bench/models/sglang.py
```

The first edit maps the official label `NGRAM` to the algorithm name accepted
by the coauthor branch. The second replaces an older SGLang engine argument.

## 4. Install the Explicit Remote Adapter

Copy or sync this `SD_benchmark/speed_bench/` directory to the machine holding
the official checkout. Then run the installer explicitly:

```bash
python /path/to/AgentSociety/SD_benchmark/speed_bench/install_official_runner_adapter.py \
  /path/to/model-optimizer-specbench/examples/specdec_bench \
  --check

python /path/to/AgentSociety/SD_benchmark/speed_bench/install_official_runner_adapter.py \
  /path/to/model-optimizer-specbench/examples/specdec_bench
```

It changes only the official checkout passed on the command line:

1. Copies `sglang_remote.py` to `specdec_bench/models/`.
2. Adds `SGLANGRemoteModel` to `specdec_bench/models/__init__.py`.
3. Adds `SGLANG_REMOTE`, its `context_length` mapping, `--server_url`, and
   the corresponding model-constructor argument to `run.py`.

The adapter sends `input_ids`, `sampling_params`, and `stream=true` to
`<server_url>/generate`. The router must expose the native SGLang `/generate`
endpoint; the coauthor `sglang_router` does. It reads `SGLANG_API_KEY` for the
gateway bearer token. The batch launcher sets this from its `API_KEY` variable.
After updating AgentSociety, rerun this installer so the external official
checkout receives the updated adapter.

## 5. Batch Experiments: Start Services and Run SPEED-Bench Automatically

For comparisons, use the batch launcher rather than starting a persistent
gateway manually. Each `(SPEED config, variant, concurrency)` point gets fresh
workers, router, L2 namespace and output directory; the launcher stops all
processes before moving to the next point. It still invokes the official
`run.py` through `SGLANG_REMOTE`.

Run it from the activated coauthor-SGLang environment after applying the
adapter in step 4:

```bash
cd /path/to/AgentSociety
conda activate sglang-v0514

OFFICIAL_RUNNER_DIR="$HOME/swq/test/SD_benchmark/speedbench-official" \
MODEL_PATH="$HOME/swq/models/Qwen2.5-14B-Instruct" \
SPEED_CONFIGS="throughput_1k" \
VARIANTS="baseline ngram_d4 ngram_d8 ngram_prob_d8" \
CONCURRENCIES="1 4 16" \
NUM_REQUESTS=20 \
OUTPUT_LENGTH=1024 \
GPU_IDS="0 1" \
bash SD_benchmark/speed_bench/run_speed_bench_batch.sh
```

By default, workers and the official runner use the active environment's
`python`. If the official runner lives in a separate `speedbench` environment,
keep the active environment as `sglang-v0514` and point only the runner at the
other interpreter:

```bash
RUNNER_PYTHON_BIN="$HOME/miniconda3/envs/speedbench/bin/python" \
bash SD_benchmark/speed_bench/run_speed_bench_batch.sh
```

`ngram_d4` / `ngram_d8` select BFS with the respective draft length;
`ngram_prob_d8` selects PROB. Add `_no_l2` to preserve local NGRAM history
but disable only the cross-worker L2 history. The following environment
variables tune the remaining NGRAM parameters for the complete sweep:

```bash
NGRAM_MATCH_TYPE=PROB \
NGRAM_BFS_BREADTH=1 \
NGRAM_MAX_TRIE_DEPTH=18 \
NGRAM_CAPACITY=500000 \
L2_ENABLED=1 \
L2_MMAP_CAPACITY=65536 \
bash SD_benchmark/speed_bench/run_speed_bench_batch.sh
```

Leave `NGRAM_MATCH_TYPE` unset to follow each variant name. Results are under
`SD_benchmark/outputs/speed_bench_multi_instance/<timestamp>/`; the root
`batch_results.csv` collects wall time, official Output TPS, official Overall
Average AL, and paths to the full official output and worker/router logs.
Each point stores L2 history in its own output directory and gets a timestamped
namespace, so no L2 history is reused across variants or separate batch runs.

For a new NGRAM option that only exists in a particular coauthor SGLang branch,
pass its worker CLI flags unchanged via `NGRAM_EXTRA_ARGS`. For example:

```bash
NGRAM_EXTRA_ARGS="--some-coauthor-ngram-option value" \
bash SD_benchmark/speed_bench/run_speed_bench_batch.sh
```

Use this only for flags that need no shell quoting inside their value. For a
path containing spaces, add a small explicit launcher option instead.

## What the Metrics Mean

The output remains the official runner's `SpecBench` and Timing output:

- `Average AL` and `Acceptance Length Histogram` are based on native SGLang
  streaming output segment lengths, as in the official in-process adapter.
- `Output TPS`, E2E, TTFT, and generation-step time now include the local
  router and HTTP transport. This is intentional for a multi-instance serving
  experiment.
- Compare variants within the same remote setup. Do not directly compare their
  TPS with a single-process, in-process `sgl.Engine` result.

The saved `responses.jsonl` and configuration remain produced by the official
runner.
