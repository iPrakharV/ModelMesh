from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / "bench" / "results" / "replay-latest.json"
TEXTS = [
    "fast stable reliable service",
    "cache hit keeps service fast",
    "great latency stable route",
    "slow broken worker timeout",
    "worker crash caused retry",
    "request completed healthy worker",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--seconds-per-stage", type=float, default=8.0)
    parser.add_argument("--rps", type=float, default=18.0)
    parser.add_argument("--output", type=Path, default=RESULT_PATH)
    return parser.parse_args()


async def control_worker(client: httpx.AsyncClient, base_url: str, index: int, *, delay_ms: int, fail_rate: float) -> bool:
    try:
        response = await client.post(
            f"{base_url}/workers/{index}/control",
            json={"delay_ms": delay_ms, "fail_rate": fail_rate},
            timeout=1.5,
        )
        return response.status_code == 200
    except httpx.HTTPError:
        return False


async def reset_workers(client: httpx.AsyncClient, base_url: str) -> None:
    try:
        workers = (await client.get(f"{base_url}/workers", timeout=1.5)).json()
    except httpx.HTTPError:
        return

    for index, _ in enumerate(workers):
        await control_worker(client, base_url, index, delay_ms=0, fail_rate=0)


async def snapshot(client: httpx.AsyncClient, base_url: str, stage: str) -> dict[str, Any]:
    metrics = (await client.get(f"{base_url}/metrics")).json()
    workers = (await client.get(f"{base_url}/workers")).json()
    return {
        "stage": stage,
        "metrics": metrics,
        "workers": workers,
    }


async def send_stage(client: httpx.AsyncClient, base_url: str, stage: str, seconds: float, rps: float) -> dict[str, Any]:
    deadline = time.perf_counter() + seconds
    interval = 1 / rps
    sent = 0
    ok = 0
    started = time.perf_counter()

    while time.perf_counter() < deadline:
        text = TEXTS[sent % len(TEXTS)]
        if stage in {"slow-worker", "flaky-worker"}:
            text = f"{text} {stage} live {sent}"

        try:
            response = await client.post(
                f"{base_url}/predict",
                json={"text": text, "request_id": f"{stage}-{sent}"},
                timeout=3.0,
            )
            ok += int(response.status_code == 200)
        except httpx.HTTPError:
            pass

        sent += 1
        elapsed = time.perf_counter() - started
        sleep_for = max(0.0, sent * interval - elapsed)
        await asyncio.sleep(sleep_for)

    snap = await snapshot(client, base_url, stage)
    return {
        "stage": stage,
        "sent": sent,
        "ok": ok,
        **snap,
    }


async def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")
    stages: list[dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=5.0) as client:
        await reset_workers(client, base_url)
        stages.append(await send_stage(client, base_url, "baseline", args.seconds_per_stage, args.rps))

        await control_worker(client, base_url, 1, delay_ms=350, fail_rate=0)
        stages.append(await send_stage(client, base_url, "slow-worker", args.seconds_per_stage, args.rps))

        await reset_workers(client, base_url)
        await control_worker(client, base_url, 0, delay_ms=0, fail_rate=0.45)
        stages.append(await send_stage(client, base_url, "flaky-worker", args.seconds_per_stage, args.rps))

        await reset_workers(client, base_url)
        stages.append(await send_stage(client, base_url, "recovery", args.seconds_per_stage, args.rps))

    payload = {
        "base_url": base_url,
        "seconds_per_stage": args.seconds_per_stage,
        "target_rps": args.rps,
        "stages": stages,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
