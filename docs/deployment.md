# Deployment notes

ModelMesh is still easiest to demo locally because the failure controls and traffic replay are designed for a closed lab. The deployable part is the same gateway and worker stack packaged with Docker.

## Local production compose

```bash
cp .env.example .env
docker compose -f docker-compose.prod.yml up --build
python scripts/smoke_check.py
```

Open:

```text
http://localhost:8000/dashboard
```

## Hosting shape

For a hosted demo, deploy the services as a private internal mesh:

| Service | Public | Notes |
| --- | --- | --- |
| gateway | yes | Exposes `/predict`, `/dashboard`, `/metrics`, and worker control proxy |
| worker-a | no | Internal service on port 9000 |
| worker-b | no | Internal service on port 9000 |
| redis | no | Internal cache only |

Required gateway env:

```text
WORKER_URLS=http://worker-a:9000,http://worker-b:9000
REDIS_URL=redis://redis:6379/0
REQUEST_TIMEOUT_SECONDS=1.5
ROUTER_STRATEGY=load_aware
CACHE_ENABLED=true
```

## What not to overclaim

This is not a production ML platform yet. It is a compact inference mesh that demonstrates training, serving, caching, routing, live worker controls, failure recovery, and benchmark evidence.

The next real hosting step is choosing a provider and wiring its private service networking. The Docker setup here is intentionally provider-neutral so the app can go to Fly.io, Render, Railway, a VM, or a Kubernetes cluster without changing the Python code.
