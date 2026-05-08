from __future__ import annotations

import argparse
import asyncio
import statistics
import time

import httpx


SAMPLES = [
    "fast stable model service",
    "worker timeout and broken response",
    "great reliable inference path",
    "slow crash error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000/predict")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    return parser.parse_args()


async def call(client: httpx.AsyncClient, url: str, index: int) -> tuple[float, bool]:
    started = time.perf_counter()
    response = await client.post(url, json={"text": SAMPLES[index % len(SAMPLES)], "request_id": str(index)})
    latency = (time.perf_counter() - started) * 1000
    return latency, response.status_code == 200


async def main() -> None:
    args = parse_args()
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=5) as client:
        async def limited(index: int) -> tuple[float, bool]:
            async with semaphore:
                return await call(client, args.url, index)

        started = time.perf_counter()
        results = await asyncio.gather(*(limited(index) for index in range(args.requests)))
        total_ms = (time.perf_counter() - started) * 1000

    latencies = [latency for latency, ok in results if ok]
    failures = sum(1 for _, ok in results if not ok)
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0)

    print({
        "requests": args.requests,
        "successes": len(latencies),
        "failures": failures,
        "total_ms": round(total_ms, 2),
        "requests_per_second": round(args.requests / (total_ms / 1000), 2),
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(p95, 2),
    })


if __name__ == "__main__":
    asyncio.run(main())
