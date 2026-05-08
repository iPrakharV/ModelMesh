# Demo script

This is the local demo flow I use to show the system moving.

Terminal 1:

```bash
docker compose up --build
```

Open:

```text
http://localhost:8000/dashboard
```

Terminal 2:

```bash
python bench/replay_traffic.py --seconds-per-stage 8 --rps 18
```

What to watch:

- Baseline traffic warms up the gateway.
- The slow-worker stage adds delay to worker 1 through the gateway control endpoint.
- The flaky-worker stage sets worker 0 to a 45 percent failure rate.
- Recovery resets both workers.

The dashboard should show request count, p95 latency, worker failure counts, delay, and fail rate moving as the replay runs.

The replay writes `bench/results/replay-latest.json` for later inspection.
