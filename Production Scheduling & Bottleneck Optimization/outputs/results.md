# Results Summary — Job-Shop Scheduling Optimization

**Instance:** ft10 (Fisher & Thompson, 1963) — 10 jobs x 10 machines, 100 operations
*(This is a famous benchmark: its optimal makespan of 930 remained unproven for 26 years until Carlier & Pinson solved it in 1989.)*

## Makespan: Optimizer vs. Heuristic Baseline

| Method | Makespan | Solve Time | Notes |
|---|---|---|---|
| SPT dispatching heuristic (baseline) | **1429** | 0.0005s | Giffler-Thompson construction, SPT priority rule |
| CP-SAT (OR-Tools, exact) | **930** | 19.99s | Status: OPTIMAL |

**Result: 34.9% reduction in makespan** (499 time units saved) by replacing a greedy dispatching rule with an exact CP-SAT optimization model.

## Bottleneck Machine Analysis (on the optimal schedule)

- **Bottleneck machine: Machine 3** — 67.8% utilization (631/930 time units busy, 299 idle)
- Average machine utilization across all 10 machines: 54.9%
- Critical path length: 17 operations, spanning machines [0, 4, 5, 7, 8]

### Utilization by machine (optimal schedule)
| Machine | Utilization | Idle Time |
|---|---|---|
| M3 | 67.8% | 299 |
| M2 | 59.8% | 374 |
| M1 | 58.9% | 382 |
| M4 | 57.4% | 396 |
| M8 | 57.1% | 399 |
| M7 | 53.7% | 431 |
| M0 | 53.0% | 437 |
| M6 | 52.8% | 439 |
| M5 | 44.7% | 514 |
| M9 | 44.1% | 520 |

## Suggested resume bullet

> Built a production scheduling optimizer in Python using Google OR-Tools CP-SAT, reducing
> makespan by **35%** (1429 → 930 time units, proven optimal)
> versus a Shortest-Processing-Time dispatching heuristic baseline on a 10-job/10-machine
> benchmark; identified the primary bottleneck resource (Machine 3, running at
> 68% utilization) via automated critical-path and machine-utilization analysis.
