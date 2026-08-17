# NGRAM Service

This directory contains a standalone CPU service around SGLang's existing C++
`NgramCorpus`. It does not implement another Trie and does not launch a model.

The version-1 data plane deliberately exposes the same three operations used by
the local corpus:

- `batch_get(req_ids, batch_tokens, total_lens)`
- `batch_put(batch_tokens, wait_for_visibility=False)`
- `erase_match_state(req_ids)`

`batch_put(..., wait_for_visibility=True)` drains the C++ corpus' global insert
queue before replying. With concurrent clients, this can wait for other pending
updates in addition to the caller's batch.

Each TCP connection receives a private numeric state namespace. Closing the
connection erases every incremental match cursor that it created. Corpus tokens
remain in the shared Trie.

Run the service from the repository root with SGLang available in the active
environment:

```bash
PYTHONPATH=python:. python -m ngram_service --host 127.0.0.1 --port 31291
```

Point a single-rank SGLang instance at it with the `NGRAM_SERVICE` speculative
backend:

```bash
sglang serve \
  --model-path <model> \
  --speculative-algorithm NGRAM_SERVICE \
  --speculative-ngram-service-address tcp://127.0.0.1:31291
```

The first implementation requires `--tp-size 1`. The service timeout defaults
to five seconds and is configurable with
`--speculative-ngram-service-timeout-s`.

The tests exercise both a minimal mock client and SGLang's production client;
they do not load a model or use a GPU:

```bash
PYTHONPATH=python:. python -m pytest -q ngram_service/tests/test_service.py
```
