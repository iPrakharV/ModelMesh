from __future__ import annotations

import argparse
import asyncio
import json
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
    parser.add_argument("--unique-requests", action="store_true")
    return parser.parse_args()


async def call(
    client: httpx.AsyncClient,
    url: str,
    index: int,
    *,
    unique_requests: bool,
) -> tuple[float, bool, bool]:
    started = time.perf_counter()
    text = SAMPLES[index % len(SAMPLES)]
    if unique_requests:
        text = f"{text} {index}"
    response = await client.post(url, json={"text": text, "request_id": str(index)})
    latency = (time.perf_counter() - started) * 1000
    cached = False
    if response.status_code == 200:
        cached = bool(response.json().get("cached"))
    return latency, response.status_code == 200, cached


async def main() -> None:
    args = parse_args()
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=5) as client:
        async def limited(index: int) -> tuple[float, bool, bool]:
            async with semaphore:
                return await call(client, args.url, index, unique_requests=args.unique_requests)

        started = time.perf_counter()
        results = await asyncio.gather(*(limited(index) for index in range(args.requests)))
        total_ms = (time.perf_counter() - started) * 1000

    latencies = [latency for latency, ok, _ in results if ok]
    failures = sum(1 for _, ok, _ in results if not ok)
    cache_hits = sum(1 for _, ok, cached in results if ok and cached)
    p95 = (
        statistics.quantiles(latencies, n=20)[18]
        if len(latencies) >= 20
        else max(latencies, default=0)
    )

    print(
        json.dumps(
            {
                "requests": args.requests,
                "successes": len(latencies),
                "failures": failures,
                "cache_hits": cache_hits,
                "cache_hit_rate": round(cache_hits / len(latencies), 4)
                if latencies
                else 0,
                "total_ms": round(total_ms, 2),
                "requests_per_second": round(args.requests / (total_ms / 1000), 2),
                "p50_latency_ms": round(statistics.median(latencies), 2)
                if latencies
                else 0,
                "p95_latency_ms": round(p95, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
