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
| gateway | yes | Exposes `/predict`, `/dashboard`, `/metrics`, and `/workers` |
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
GATEWAY_CONTROLS_ENABLED=false
```

## What not to overclaim

This is not a production ML platform yet. It is a compact inference mesh that demonstrates training, serving, caching, routing, live worker controls, failure recovery, and benchmark evidence.

The next real hosting step is applying the Render Blueprint and validating the gateway URL with
`scripts/smoke_check.py`.

## Render Blueprint

`render.yaml` defines a Render Blueprint for:

- one public gateway web service
- two private worker services
- one Render Key Value cache

The gateway reads `WORKER_A_HOSTPORT` and `WORKER_B_HOSTPORT`, which Render fills from each private worker service. That avoids hardcoding internal hostnames.

The Blueprint also sets `GATEWAY_CONTROLS_ENABLED=false`, so the public dashboard can show worker state without exposing delay or failure controls.

Important: the private worker services use Render's `starter` plan in the Blueprint because private services do not run on the free plan. Review the cost in Render before applying the Blueprint.
