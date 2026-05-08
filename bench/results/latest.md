# Benchmark results

Generated with `python bench/run_scenarios.py` on the local machine.

| Scenario | Router | Cache | Requests | RPS | p50 ms | p95 ms | Cache hits | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache_enabled | load_aware | True | 80 | 211.37 | 5.54 | 312.52 | 71 | 0 |
| cache_disabled | load_aware | False | 80 | 24.92 | 318.85 | 326.38 | 0 | 0 |
| slow_worker_round_robin | round_robin | False | 80 | 25.68 | 67.3 | 1033.24 | 0 | 0 |
| slow_worker_load_aware | load_aware | False | 80 | 128.51 | 4.94 | 425.03 | 0 | 0 |
| dead_worker_recovery | load_aware | False | 60 | 698.97 | 11.15 | 28.1 | 0 | 0 |

Notes:

- Cache scenarios repeat four text inputs to show cache behavior.
- Slow-worker scenarios use one normal worker and one worker with 95 ms artificial delay.
- Dead-worker recovery uses one worker with a 100 percent simulated failure rate.
