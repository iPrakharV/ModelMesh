# ModelMesh

A small inference gateway for testing how ML services behave when traffic, caching, and worker failures get involved.

The first version runs a FastAPI gateway in front of a few model workers. The gateway accepts prediction requests, checks a cache, routes to a worker, retries on failure, and keeps basic latency and error metrics. Worker routing tracks recent latency, failures, and in-flight requests.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

uvicorn worker.app.main:app --port 8011
WORKER_URLS=http://localhost:8011 uvicorn gateway.app.main:app --port 8000
```

Try it:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text":"the model service is fast and stable"}'
```

Run a quick local benchmark:

```bash
python bench/run_bench.py --requests 100 --concurrency 10
```

Open the local dashboard:

```text
http://localhost:8000/dashboard
```

## Docker

```bash
docker compose up --build
```

That starts Redis, one gateway, and two workers.

## Benchmarks

The scenario runner starts local gateway and worker processes, sends load, then writes JSON and markdown results.

```bash
python bench/run_scenarios.py
```

Latest local run:

| Scenario | RPS | p50 ms | p95 ms | Cache hits | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| cache enabled | 211.37 | 5.54 | 312.52 | 71 | 0 |
| cache disabled | 24.92 | 318.85 | 326.38 | 0 | 0 |
| slow worker, round-robin | 25.68 | 67.3 | 1033.24 | 0 | 0 |
| slow worker, load-aware | 128.51 | 4.94 | 425.03 | 0 | 0 |
| dead worker recovery | 698.97 | 11.15 | 28.1 | 0 | 0 |

Full output is in `bench/results/latest.md` and `bench/results/latest.json`.

## Training

The worker can load a small trained text model artifact from `worker/app/artifacts/tiny_text_model.json`.

On Apple Silicon, the training script uses PyTorch MPS. It refuses to run on CPU by default so a slow accidental CPU run fails fast.

```bash
pip install -r requirements-train.txt
python training/train_text_model.py
```

If it is using the Mac GPU, the output includes `'device': 'mps'`.

## What is in here now

- FastAPI gateway with `/predict`, `/health`, `/workers`, and `/metrics`
- Worker service with a tiny text classifier and optional trained artifact
- Redis-backed cache when Redis is available, with an in-memory fallback
- Retry on failed worker calls and basic load-aware routing
- Scenario benchmark runner with saved results
- Local dashboard for gateway and worker state
- Unit tests for routing, caching, metrics, and the model

This is still intentionally small. The useful part is the system behavior around the model, not the model itself.
