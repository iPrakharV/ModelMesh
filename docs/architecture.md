# Architecture notes

ModelMesh is an ML systems project. The model is intentionally small so the repo can focus on serving behavior: routing, caching, retries, observability, and benchmark evidence.

```mermaid
flowchart LR
    client["Client"] --> gateway["FastAPI gateway"]
    gateway --> cache["Redis or memory cache"]
    gateway --> worker_a["Worker A"]
    gateway --> worker_b["Worker B"]
    worker_a --> model_a["Text model"]
    worker_b --> model_b["Text model"]
    gateway --> metrics["Metrics and dashboard"]
```

## Request path

1. The client sends text to `POST /predict`.
2. The gateway checks the prediction cache using the input text as the cache identity.
3. On a miss, the gateway asks the worker pool for candidates.
4. The selected worker returns a prediction with label, score, model version, and worker id.
5. The gateway records latency, cache hits, worker errors, and worker health.

## Routing

The gateway supports two routing modes:

- `round_robin` for a simple baseline.
- `load_aware` for routing based on failures, in-flight requests, and recent latency.

The benchmark runner uses both modes against the same slow-worker scenario so the difference is measurable.

## Training

`training/train_text_model.py` trains the compact text model with PyTorch. On Apple Silicon it uses `mps`, and it exits instead of silently falling back to CPU unless `--allow-cpu` is passed.

The trained artifact is plain JSON so the worker can load it without depending on PyTorch at inference time.

The latest checked-in run trained on `mps` in under a second. The run file records device, epochs, examples/sec, train accuracy, and test accuracy so the performance claim is reproducible.

## Live failure demo

Workers expose a small local control endpoint for demo runs:

- `GET /config` returns delay and failure settings.
- `POST /control` updates artificial delay and failure rate.

The gateway wraps those controls at `POST /workers/{index}/control`, and the dashboard exposes inputs for delay and fail rate. `bench/replay_traffic.py` uses the same endpoint to move the system through baseline, slow-worker, flaky-worker, and recovery stages.
