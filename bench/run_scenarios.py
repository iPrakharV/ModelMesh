from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import httpx


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "bench" / "results"
BASE_PORT = 8300

SAMPLES = [
    "fast stable model service",
    "worker timeout and broken response",
    "great reliable inference path",
    "slow crash error",
]


@dataclass(frozen=True)
class WorkerConfig:
    name: str
    port: int
    delay_ms: int = 0
    fail_rate: float = 0.0


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    workers: list[WorkerConfig]
    router_strategy: str = "load_aware"
    cache_enabled: bool = True
    requests: int = 80
    concurrency: int = 10
    unique_inputs: bool = False


SCENARIOS = [
    Scenario(
        name="cache_enabled",
        description="One delayed worker, four repeated inputs, cache enabled.",
        workers=[WorkerConfig("worker-a", BASE_PORT + 11, delay_ms=35)],
        cache_enabled=True,
        requests=80,
        concurrency=8,
    ),
    Scenario(
        name="cache_disabled",
        description="Same traffic as cache_enabled, but cache disabled.",
        workers=[WorkerConfig("worker-a", BASE_PORT + 11, delay_ms=35)],
        cache_enabled=False,
        requests=80,
        concurrency=8,
    ),
    Scenario(
        name="slow_worker_round_robin",
        description="One fast worker and one slow worker with round-robin routing.",
        workers=[
            WorkerConfig("worker-fast", BASE_PORT + 11),
            WorkerConfig("worker-slow", BASE_PORT + 12, delay_ms=95),
        ],
        router_strategy="round_robin",
        cache_enabled=False,
        requests=80,
        concurrency=12,
        unique_inputs=True,
    ),
    Scenario(
        name="slow_worker_load_aware",
        description="Same workers as slow_worker_round_robin, using load-aware routing.",
        workers=[
            WorkerConfig("worker-fast", BASE_PORT + 11),
            WorkerConfig("worker-slow", BASE_PORT + 12, delay_ms=95),
        ],
        router_strategy="load_aware",
        cache_enabled=False,
        requests=80,
        concurrency=12,
        unique_inputs=True,
    ),
    Scenario(
        name="dead_worker_recovery",
        description="One worker always fails, one worker stays healthy.",
        workers=[
            WorkerConfig("worker-dead", BASE_PORT + 11, fail_rate=1.0),
            WorkerConfig("worker-healthy", BASE_PORT + 12),
        ],
        router_strategy="load_aware",
        cache_enabled=False,
        requests=60,
        concurrency=10,
        unique_inputs=True,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-json", type=Path, default=RESULTS_DIR / "latest.json")
    parser.add_argument("--output-md", type=Path, default=RESULTS_DIR / "latest.md")
    return parser.parse_args()


def service_env(extra: dict[str, str]) -> dict[str, str]:
    env = os.environ.copy()
    env.update(extra)
    env["PYTHONPATH"] = str(ROOT)
    return env


def start_process(args: list[str], env: dict[str, str], log_path: Path) -> subprocess.Popen:
    log_file = log_path.open("w")
    return subprocess.Popen(
        args,
        cwd=ROOT,
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
    )


async def wait_for_health(url: str, timeout_seconds: float = 8.0) -> None:
    deadline = time.time() + timeout_seconds
    async with httpx.AsyncClient(timeout=1.0) as client:
        while time.time() < deadline:
            try:
                response = await client.get(url)
                if response.status_code == 200:
                    return
            except httpx.HTTPError:
                await asyncio.sleep(0.1)
    raise RuntimeError(f"service did not become healthy: {url}")


def stop_processes(processes: list[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
    for process in processes:
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()


async def call(client: httpx.AsyncClient, url: str, index: int, unique_inputs: bool) -> dict[str, Any]:
    text = SAMPLES[index % len(SAMPLES)]
    if unique_inputs:
        text = f"{text} sample {index}"
    started = time.perf_counter()
    response = await client.post(url, json={"text": text, "request_id": str(index)})
    latency_ms = (time.perf_counter() - started) * 1000
    body: dict[str, Any] = {}
    if response.headers.get("content-type", "").startswith("application/json"):
        body = response.json()
    return {
        "ok": response.status_code == 200,
        "status_code": response.status_code,
        "latency_ms": latency_ms,
        "cached": bool(body.get("cached")),
        "worker_url": body.get("worker_url"),
    }


async def run_load(url: str, scenario: Scenario) -> dict[str, Any]:
    semaphore = asyncio.Semaphore(scenario.concurrency)
    async with httpx.AsyncClient(timeout=5.0) as client:
        async def limited(index: int) -> dict[str, Any]:
            async with semaphore:
                return await call(client, url, index, scenario.unique_inputs)

        started = time.perf_counter()
        rows = await asyncio.gather(*(limited(index) for index in range(scenario.requests)))
        total_ms = (time.perf_counter() - started) * 1000

        gateway_metrics = (await client.get(url.replace("/predict", "/metrics"))).json()
        worker_snapshot = (await client.get(url.replace("/predict", "/workers"))).json()

    latencies = [row["latency_ms"] for row in rows if row["ok"]]
    failures = sum(1 for row in rows if not row["ok"])
    cache_hits = sum(1 for row in rows if row["ok"] and row["cached"])
    p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else max(latencies, default=0)

    return {
        "requests": scenario.requests,
        "concurrency": scenario.concurrency,
        "successes": len(latencies),
        "failures": failures,
        "cache_hits": cache_hits,
        "cache_hit_rate": round(cache_hits / len(latencies), 4) if latencies else 0,
        "total_ms": round(total_ms, 2),
        "requests_per_second": round(scenario.requests / (total_ms / 1000), 2),
        "p50_latency_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "p95_latency_ms": round(p95, 2),
        "gateway_metrics": gateway_metrics,
        "workers": worker_snapshot,
    }


async def run_scenario(scenario: Scenario, log_dir: Path) -> dict[str, Any]:
    processes: list[subprocess.Popen] = []
    gateway_port = BASE_PORT
    worker_urls = []

    try:
        for worker in scenario.workers:
            worker_urls.append(f"http://localhost:{worker.port}")
            processes.append(start_process(
                [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "worker.app.main:app",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(worker.port),
                ],
                service_env({
                    "WORKER_ID": worker.name,
                    "SIMULATE_DELAY_MS": str(worker.delay_ms),
                    "FAIL_RATE": str(worker.fail_rate),
                }),
                log_dir / f"{scenario.name}-{worker.name}.log",
            ))

        for url in worker_urls:
            await wait_for_health(f"{url}/health")

        processes.append(start_process(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "gateway.app.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(gateway_port),
            ],
            service_env({
                "WORKER_URLS": ",".join(worker_urls),
                "ROUTER_STRATEGY": scenario.router_strategy,
                "CACHE_ENABLED": "true" if scenario.cache_enabled else "false",
                "REQUEST_TIMEOUT_SECONDS": "1.0",
            }),
            log_dir / f"{scenario.name}-gateway.log",
        ))
        await wait_for_health(f"http://localhost:{gateway_port}/health")

        result = await run_load(f"http://localhost:{gateway_port}/predict", scenario)
        return {
            "name": scenario.name,
            "description": scenario.description,
            "router_strategy": scenario.router_strategy,
            "cache_enabled": scenario.cache_enabled,
            **result,
        }
    finally:
        stop_processes(processes)


def write_markdown(path: Path, results: list[dict[str, Any]]) -> None:
    lines = [
        "# Benchmark results",
        "",
        "Generated with `python bench/run_scenarios.py` on the local machine.",
        "",
        "| Scenario | Router | Cache | Requests | RPS | p50 ms | p95 ms | Cache hits | Failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in results:
        lines.append(
            f"| {row['name']} | {row['router_strategy']} | {row['cache_enabled']} | "
            f"{row['requests']} | {row['requests_per_second']} | {row['p50_latency_ms']} | "
            f"{row['p95_latency_ms']} | {row['cache_hits']} | {row['failures']} |"
        )

    lines.extend([
        "",
        "Notes:",
        "",
        "- Cache scenarios repeat four text inputs to show cache behavior.",
        "- Slow-worker scenarios use one normal worker and one worker with 95 ms artificial delay.",
        "- Dead-worker recovery uses one worker with a 100 percent simulated failure rate.",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


async def main() -> None:
    args = parse_args()
    with TemporaryDirectory(prefix="modelmesh-bench-") as directory:
        log_dir = Path(directory)
        results = []
        for scenario in SCENARIOS:
            print(f"running {scenario.name}")
            results.append(await run_scenario(scenario, log_dir))

    payload = {
        "generated_by": "python bench/run_scenarios.py",
        "results": results,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n")
    write_markdown(args.output_md, results)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
