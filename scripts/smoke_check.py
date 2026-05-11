from __future__ import annotations

import argparse
import json
import time
from urllib import request


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--timeout", type=float, default=12.0)
    return parser.parse_args()


def get_json(url: str, timeout: float = 2.0) -> dict:
    with request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_text(url: str, timeout: float = 2.0) -> str:
    with request.urlopen(url, timeout=timeout) as response:
        return response.read().decode("utf-8")


def post_json(url: str, payload: dict, timeout: float = 2.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url,
        data=data,
        headers={"content-type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_for_gateway(base_url: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            health = get_json(f"{base_url}/health")
            if health.get("status") == "ok":
                return health
        except Exception as exc:
            last_error = exc
            time.sleep(0.25)
    raise RuntimeError(f"gateway did not become healthy: {last_error}")


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    health = wait_for_gateway(base_url, args.timeout)
    prediction = post_json(
        f"{base_url}/predict",
        {"text": "fast stable reliable service", "request_id": "smoke-check"},
    )
    metrics = get_json(f"{base_url}/metrics")
    workers = get_json(f"{base_url}/workers")
    dashboard = get_text(f"{base_url}/dashboard")

    if prediction["prediction"]["label"] != "healthy":
        raise RuntimeError(f"unexpected prediction: {prediction}")
    if "requests" not in metrics or "p95_latency_ms" not in metrics:
        raise RuntimeError(f"unexpected metrics payload: {metrics}")
    if not workers:
        raise RuntimeError("gateway returned no workers")
    if "ModelMesh" not in dashboard or "Gateway metrics" not in dashboard:
        raise RuntimeError("dashboard did not return the expected HTML")

    print(
        json.dumps(
            {
                "health": health,
                "prediction": prediction,
                "metrics": metrics,
                "worker_count": len(workers),
                "dashboard": "ok",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
