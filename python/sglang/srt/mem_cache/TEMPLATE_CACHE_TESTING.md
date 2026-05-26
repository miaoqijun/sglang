# Template-Aware Chunk Cache — Testing Instructions

Migrate this branch to a working environment with GPU + the sglang dev
toolchain, then run the steps below to verify correctness end-to-end.

## 0. Files changed / added (sanity check)

```
# New
python/sglang/srt/managers/prompt_template_registry.py
python/sglang/srt/mem_cache/template_chunk_cache.py
test/registered/template_cache/test_template_chunk_cache.py
benchmark/generative_agents/templates.py
benchmark/generative_agents/bench_sglang_template.py

# Modified
python/sglang/srt/managers/io_struct.py           # +template fields, +RegisterPromptTemplateReq{Input,Output}
python/sglang/srt/managers/tokenizer_manager.py   # registry, expand_template path, register_prompt_template
python/sglang/srt/managers/scheduler.py           # TemplateAwareChunkCache branch, Req(...) template kwargs
python/sglang/srt/managers/schedule_batch.py      # Req fields: template_id / segment_boundaries / segment_kinds
python/sglang/srt/server_args.py                  # --enable-template-chunk-cache + mutex check
python/sglang/srt/entrypoints/http_server.py      # POST /register_prompt_template
```

## 1. Unit test (no GPU required)

```bash
cd <repo>
python -m pytest test/registered/template_cache/test_template_chunk_cache.py -v
```

Expected output: 8 tests pass. Covers
1. first request misses → 2 chunks inserted (fixed + var)
2. identical second request: full prefix hit
3. same fixed, different var: only fixed hits
4. different template_id: no sharing
5. fall-through (no template_id): ChunkCache-like, no chunks created
6. LRU: oldest leaf evicted first
7. lock_ref blocks eviction, dec_lock_ref unblocks
8. page_size > 1 + shared boundary page: page survives until both chunks evicted

## 2. End-to-end smoke test (single GPU)

In one terminal — start the server with template cache enabled:

```bash
python -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3-8B-Instruct \
    --port 30000 \
    --enable-template-chunk-cache \
    --disable-cuda-graph        # speeds up first iteration
```

Wait until you see `The server is fired up and ready to roll!`.

In another terminal — register a template and exercise it twice:

```bash
# (a) Register
curl -s -X POST localhost:30000/register_prompt_template -H "Content-Type: application/json" -d '{
  "template_id": "demo",
  "segments": [
    {"kind": "fixed", "text": "Greet a user named "},
    {"kind": "var",   "var_name": "name"},
    {"kind": "fixed", "text": " in one short sentence."}
  ]
}' | jq .

# Expected: {"success": true, "template_id": "demo", ...}

# (b) First call — miss on the fixed segment, then insert.
curl -s -X POST localhost:30000/generate -H "Content-Type: application/json" -d '{
  "template_id": "demo",
  "template_vars": {"name": "Alice"},
  "sampling_params": {"max_new_tokens": 20, "temperature": 0.0}
}' | jq '.text, .meta_info'

# (c) Second call — same name. Should hit BOTH segments.
curl -s -X POST localhost:30000/generate -H "Content-Type: application/json" -d '{
  "template_id": "demo",
  "template_vars": {"name": "Alice"},
  "sampling_params": {"max_new_tokens": 20, "temperature": 0.0}
}' | jq '.meta_info | {prompt_tokens, cached_tokens}'

# (d) Third call — different name. Should hit only the fixed segment.
curl -s -X POST localhost:30000/generate -H "Content-Type: application/json" -d '{
  "template_id": "demo",
  "template_vars": {"name": "Bob"},
  "sampling_params": {"max_new_tokens": 20, "temperature": 0.0}
}' | jq '.meta_info | {prompt_tokens, cached_tokens}'
```

What to check in `meta_info`:
- Call (c): `cached_tokens` ≈ prompt length (full prefix hit).
- Call (d): `cached_tokens` ≈ length of the *fixed* segment only.

Inspect more detail via:

```bash
curl -s localhost:30000/get_server_info | jq '.scheduler[0].tree_cache'
```

## 3. Mutex validation

```bash
python -m sglang.launch_server --model-path <m> --enable-template-chunk-cache --disable-radix-cache
# Expect: ValueError --enable-template-chunk-cache is mutually exclusive with: --disable-radix-cache

python -m sglang.launch_server --model-path <m> --enable-template-chunk-cache --enable-hierarchical-cache
# Expect: ValueError --enable-template-chunk-cache is mutually exclusive with: --enable-hierarchical-cache
```

## 4. Fall-through (request without `template_id`)

With `--enable-template-chunk-cache` enabled, sending a normal `/generate`
request that omits `template_id` should still work (no caching, but no
crash). Verify:

```bash
curl -s -X POST localhost:30000/generate -d '{
  "text": "Tell me a joke.",
  "sampling_params": {"max_new_tokens": 30}
}' | jq .text
```

## 5. End-to-end benchmark (compare RadixCache vs template cache)

```bash
# Baseline: plain RadixCache
python -m sglang.launch_server --model-path <model> --port 30000
python benchmark/generative_agents/bench_sglang.py --num-events 50

# Restart server with template cache
python -m sglang.launch_server --model-path <model> --port 30000 \
    --enable-template-chunk-cache
python benchmark/generative_agents/bench_sglang_template.py \
    --base-url http://127.0.0.1:30000 --num-events 50
```

Compare:
- Total latency reported by each script
- `cached_tokens` aggregated from `/get_server_info`
- KV-pool occupancy (look for `available_size` in `/get_server_info`); under
  the template-aware path the fixed-segment share of the prompt should
  remain resident across calls

## 6. Page-size variant

Re-run the benchmark with `--page-size 16` on both servers to confirm
page-refcount accounting (D1) does not over-free shared boundary pages.

```bash
python -m sglang.launch_server --model-path <model> --port 30000 \
    --enable-template-chunk-cache --page-size 16
```

If `evict()` ever asserts on a double-free or the allocator reports
`available_size > total_size`, that's the regression to look for.

## 7. Known caveats (expected, not bugs)

- Per-segment tokenize+concat can differ from a single tokenize() over the
  fully assembled prompt by 1–2 tokens around variable boundaries (BPE
  merges across the boundary). Generation quality is unaffected; only token
  counts may drift slightly. If a downstream check requires exact parity,
  use offset_mapping in a follow-up.
- The cache stores variable segments too; with high cardinality vars these
  chunks rarely hit. The plan deliberately defers smarter eviction
  (segment_kind-weighted LRU) until after this prototype is benchmarked.
