# ModelMesh

A small inference gateway for testing how ML services behave when traffic, caching, and worker failures get involved.

The first version runs a FastAPI gateway in front of a few model workers. The gateway accepts prediction requests, checks a cache, routes to a worker, retries on failure, and keeps basic latency and error metrics.

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

## Docker

```bash
docker compose up --build
```

That starts Redis, one gateway, and two workers.

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
- Retry on failed worker calls
- Simple benchmark script
- Unit tests for routing, caching, metrics, and the model

This is still intentionally small. The useful part is the system behavior around the model, not the model itself.
