# Rust NGRAM service

Minimal Tokio service for the existing NGRAM binary protocol. The hot path
calls SGLang's C++ NGRAM corpus directly through `cpp/ngram_bridge.cpp` and does
not depend on Python, HTTP, or JSON.

Build from the SGLang repository root:

```bash
cargo build --release --manifest-path ngram_service_rust/Cargo.toml
```

Run a service compatible with `NgramServiceClient`:

```bash
ngram_service_rust/target/release/ngram-service-rust \
  --host 127.0.0.1 --port 31291 \
  --capacity 10000000 --max-trie-depth 18 \
  --min-bfs-breadth 1 --max-bfs-breadth 10 \
  --draft-token-num 4 --match-type BFS
```
