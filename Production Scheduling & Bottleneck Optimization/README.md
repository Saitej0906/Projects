# Production Scheduling & Bottleneck Optimization

Exact optimization of job-shop production schedules using Google OR-Tools **CP-SAT**,
benchmarked against a classical dispatching-rule heuristic, with automated bottleneck
identification and Gantt-chart visualization.

## About the dataset

This project runs on **`ft10`**, the classic Fisher & Thompson (1963) 10-job x 10-machine
job-shop benchmark — one of the most famous instances in scheduling research (its optimal
makespan of **930** went unproven for 26 years until Carlier & Pinson closed it in 1989).

> **Note:** The originally uploaded file (`10_4Jopshop`) turned out to be malformed for
> genuine Taillard format — it contained a single 100x100 matrix with a corrupted header
> (`10000 10000`), rather than the two matrices (processing times + machine routing) that
> Taillard-format instances require, and at 100x100 it was also far too large for an exact
> CP-SAT solve to be practical in a demo. `ft10` was substituted as a clean, correctly
> formatted, appropriately sized, and well-documented Taillard-style instance. The parser in
> `src/parser.py` reads genuine Taillard format, so swapping in any other correctly formatted
> instance (e.g. from the [OR-Library](http://people.brunel.ac.uk/~mastjjb/jeb/orlib/jobshopinfo.html))
> is a one-line change (`DATA_PATH` in `src/main.py`).

## Problem

Given J jobs, each consisting of a fixed sequence of operations, and each operation requiring
a specific machine for a specific duration, find a schedule (start time for every operation)
that:
1. respects each job's operation order,
2. never double-books a machine, and
3. minimizes the **makespan** — the completion time of the last operation across all jobs.

This is NP-hard in general. The project solves it two ways and compares them:

| Approach | What it does | Guarantee |
|---|---|---|
| **CP-SAT (OR-Tools)** | Exact constraint-programming model | Proven optimal (or a certified gap, if time-limited) |
| **SPT dispatching heuristic** | Greedy Giffler-Thompson construction with a Shortest-Processing-Time priority rule | Fast, no optimality guarantee |

## Approach

### 1. Parsing (`src/parser.py`)
Reads Taillard-format files: a header (`num_jobs num_machines`), a processing-times matrix,
and a machine-routing matrix (which machine each operation uses, in job-specific order).

### 2. Exact optimization (`src/cp_sat_solver.py`)
Models the problem as a CP-SAT constraint program:
- **Interval variables** for every operation (`start`, `end`, fixed `duration`).
- **Precedence constraints** chaining each job's operations in order.
- **`AddNoOverlap` constraints** per machine — this is the disjunctive "only one job on a
  machine at a time" rule, and the core combinatorial difficulty of job-shop scheduling.
- **Objective**: minimize `makespan = max(end time of every job's last operation)`.

CP-SAT searches by simultaneously improving a best-found solution (upper bound) and a
relaxation-based lower bound until they meet — at which point the solution is *proven*
optimal, not just heuristically good.

### 3. Heuristic baseline (`src/heuristic_spt.py`)
Implements the Giffler & Thompson (1960) active-schedule-generation algorithm using
Shortest-Processing-Time as its priority rule: at each step, among all operations that could
legitimately conflict for the next available machine slot, schedule the one with the shortest
duration. This produces a valid, "active" (no-wasted-gaps) schedule extremely fast, but with
no optimality guarantee — the natural point of comparison for the exact solver.

### 4. Bottleneck analysis (`src/bottleneck_analysis.py`)
- **Utilization** per machine = busy time / makespan. The highest-utilization machine is
  flagged as the bottleneck — the resource most constantly in demand.
- **Critical path**: a backward trace from the operation that finishes at the makespan,
  following zero-slack same-job and same-machine predecessors, to show which machines are
  actually driving the schedule's length.

### 5. Visualization (`src/visualize.py`)
- Gantt charts (per-machine timeline, bars colored by job) for both the optimal and heuristic
  schedules.
- A machine utilization bar chart with the bottleneck machine highlighted in red.

### 6. Results summary (`src/main.py`)
Runs the full pipeline and writes `outputs/results.json` (machine-readable) and
`outputs/results.md` (human-readable, including a ready-to-use resume bullet).

## How to run

```bash
pip install -r requirements.txt
python src/main.py
```

Outputs are written to `outputs/`:
- `gantt_cp_sat_optimal.png`
- `gantt_spt_heuristic.png`
- `machine_utilization.png`
- `results.json`
- `results.md`

Each module also has a `__main__` block, so you can run any step individually, e.g.:
```bash
python src/cp_sat_solver.py
python src/heuristic_spt.py
python src/bottleneck_analysis.py
```

## Project structure

```
.
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── ft10.txt              # Taillard-format instance (10 jobs x 10 machines)
├── src/
│   ├── parser.py              # Taillard-format parser
│   ├── cp_sat_solver.py       # Exact CP-SAT job-shop model
│   ├── heuristic_spt.py       # Giffler-Thompson / SPT dispatching baseline
│   ├── bottleneck_analysis.py # Utilization + critical path analysis
│   ├── visualize.py           # Gantt + utilization charts
│   └── main.py                # End-to-end pipeline
└── outputs/
    ├── gantt_cp_sat_optimal.png
    ├── gantt_spt_heuristic.png
    ├── machine_utilization.png
    ├── results.json
    └── results.md
```

## Results (ft10 instance)

| Metric | Value |
|---|---|
| SPT heuristic makespan | 1429 |
| CP-SAT optimal makespan | **930** (proven optimal) |
| Improvement | **34.9% reduction in makespan** |
| Bottleneck machine | Machine 3 (67.8% utilization) |
| Average machine utilization | 54.9% |

See `outputs/results.md` for full details and a ready-to-use resume bullet.
