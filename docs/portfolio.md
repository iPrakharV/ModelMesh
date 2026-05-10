# Portfolio copy

## Pinned repo title

ModelMesh

## GitHub repo description

Trainable ML inference mesh with MPS training, load-aware routing, caching, failure replay, benchmarks, Docker, and a live dashboard.

## LinkedIn Featured description

Built a compact ML inference mesh with Apple Silicon training, model workers, cache-aware gateway routing, failure replay, Docker, benchmarks, and a live dashboard for p95 latency and worker health.

## Resume bullets

- Built a FastAPI-based ML inference mesh with a gateway, model workers, cache-aware prediction serving, retry handling, and live worker health metrics.
- Trained and exported a small PyTorch text classifier on Apple Silicon MPS, recording training time, throughput, evaluation metrics, and a model card.
- Added benchmark and replay tooling to compare cache behavior, round-robin routing, load-aware routing, slow-worker handling, and failure recovery.

## LinkedIn post draft

I built ModelMesh, a small ML inference system for testing what happens around a model after training.

It has a FastAPI gateway, multiple model workers, caching, load-aware routing, worker failure controls, Docker setup, benchmark scripts, and a live dashboard. The model is intentionally small, but it trains locally on Apple Silicon MPS and exports as a JSON artifact the workers can load without PyTorch at inference time.

The most useful part was measuring system behavior: cache on vs off, round-robin vs load-aware routing, slow workers, flaky workers, and recovery. It is not a production ML platform, but it is a solid lab for the pieces that usually sit around an ML model in real systems.

Repo: ModelMesh
