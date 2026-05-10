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

Run live traffic against the dashboard:

```bash
python bench/replay_traffic.py --seconds-per-stage 8 --rps 18
```

## Docker

```bash
docker compose up --build
```

That starts Redis, one gateway, and two workers.

For a production-style local run:

```bash
cp .env.example .env
docker compose -f docker-compose.prod.yml up --build
python scripts/smoke_check.py
```

Deployment notes are in `docs/deployment.md`.

## Benchmarks

The scenario runner starts local gateway and worker processes, sends load, then writes JSON and markdown results.

```bash
python bench/run_scenarios.py
```

Latest local run:

| Scenario | RPS | p50 ms | p95 ms | Cache hits | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| cache enabled | 728.95 | 5.01 | 52.04 | 72 | 0 |
| cache disabled | 185.7 | 41.1 | 53.09 | 0 | 0 |
| slow worker, round-robin | 187.36 | 57.79 | 111.91 | 0 | 0 |
| slow worker, load-aware | 688.48 | 5.21 | 111.9 | 0 | 0 |
| dead worker recovery | 707.98 | 11.37 | 28.93 | 0 | 0 |

Full output is in `bench/results/latest.md` and `bench/results/latest.json`.

The full project report is in `docs/report.md`.

## Training

The worker can load a small trained text model artifact from `worker/app/artifacts/tiny_text_model.json`. The current model trains on generated service-status examples, so the point is the local training and serving path, not benchmark-grade NLP accuracy.

On Apple Silicon, the training script uses PyTorch MPS. It refuses to run on CPU by default so a slow accidental CPU run fails fast.

```bash
pip install -r requirements-train.txt
python training/train_text_model.py
```

If it is using the Mac GPU, the output includes `'device': 'mps'`.

Latest training run on an M3 Pro:

| Metric | Value |
| --- | ---: |
| Device | mps |
| Epochs | 320 |
| Train examples | 3,276 |
| Test examples | 820 |
| Train time | 0.4351 s |
| Examples/sec | 2,409,512.37 |
| Train accuracy | 1.0 |
| Test accuracy | 1.0 |

The full run output is in `training/runs/latest.md` and `training/runs/latest.json`.

Model evaluation and limitations are documented in `docs/model_card.md`.

## What is in here now

- FastAPI gateway with `/predict`, `/health`, `/workers`, and `/metrics`
- Worker service with a tiny text classifier and optional trained artifact
- Redis-backed cache when Redis is available, with an in-memory fallback
- Retry on failed worker calls and basic load-aware routing
- Scenario benchmark runner with saved results
- Local dashboard for gateway and worker state
- Worker delay/failure controls for live demos
- Unit tests for routing, caching, metrics, and the model

This is still intentionally small. The useful part is the system behavior around the model, not the model itself.
