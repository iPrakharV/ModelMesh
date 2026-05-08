# Benchmark results

Generated with `python bench/run_scenarios.py` on the local machine.

| Scenario | Router | Cache | Requests | RPS | p50 ms | p95 ms | Cache hits | Failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cache_enabled | load_aware | True | 80 | 728.95 | 5.01 | 52.04 | 72 | 0 |
| cache_disabled | load_aware | False | 80 | 185.7 | 41.1 | 53.09 | 0 | 0 |
| slow_worker_round_robin | round_robin | False | 80 | 187.36 | 57.79 | 111.91 | 0 | 0 |
| slow_worker_load_aware | load_aware | False | 80 | 688.48 | 5.21 | 111.9 | 0 | 0 |
| dead_worker_recovery | load_aware | False | 60 | 707.98 | 11.37 | 28.93 | 0 | 0 |

Notes:

- Cache scenarios repeat four text inputs to show cache behavior.
- Slow-worker scenarios use one normal worker and one worker with 95 ms artificial delay.
- Dead-worker recovery uses one worker with a 100 percent simulated failure rate.
