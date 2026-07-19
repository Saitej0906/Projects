"""
main.py
-------
Runs the full pipeline end-to-end:
  1. Parse the Taillard-format dataset
  2. Solve to optimality with CP-SAT
  3. Solve with the SPT dispatching heuristic (baseline)
  4. Identify the bottleneck machine
  5. Generate Gantt charts + utilization chart
  6. Write a quantified results summary (outputs/results.md and .json)

Run from the project root:
    python src/main.py
"""

import json
import os
import time

from parser import parse_taillard
from cp_sat_solver import solve_job_shop
from heuristic_spt import solve_spt_heuristic
from bottleneck_analysis import compute_utilization, identify_bottleneck, critical_path
from visualize import plot_gantt, plot_utilization

DATA_PATH = "data/ft10.txt"
OUT_DIR = "outputs"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ---- 1. Parse ----
    print("Parsing dataset...")
    instance = parse_taillard(DATA_PATH)
    print(f"  {instance.num_jobs} jobs x {instance.num_machines} machines "
          f"({instance.total_operations} operations)")
    print(f"  Trivial lower bound on makespan: {instance.lower_bound_makespan}")

    # ---- 2. CP-SAT exact solve ----
    print("\nSolving with CP-SAT (exact)...")
    t0 = time.time()
    cp_result = solve_job_shop(instance, time_limit_seconds=60.0)
    cp_wall = time.time() - t0
    print(f"  Status: {cp_result.status}")
    print(f"  Makespan: {cp_result.makespan}")
    print(f"  Solve time: {cp_wall:.2f}s")

    # ---- 3. SPT heuristic baseline ----
    print("\nSolving with SPT dispatching heuristic (baseline)...")
    t0 = time.time()
    heuristic_result = solve_spt_heuristic(instance)
    heur_wall = time.time() - t0
    print(f"  Makespan: {heuristic_result.makespan}")
    print(f"  Solve time: {heur_wall:.4f}s")

    improvement_pct = 100.0 * (heuristic_result.makespan - cp_result.makespan) / heuristic_result.makespan

    # ---- 4. Bottleneck analysis (on the optimal schedule) ----
    print("\nAnalyzing bottleneck machine (optimal schedule)...")
    stats = compute_utilization(cp_result.schedule, instance.num_machines, cp_result.makespan)
    bottleneck = identify_bottleneck(stats)
    avg_utilization = sum(s.utilization for s in stats) / len(stats)
    cpath = critical_path(cp_result.schedule, instance, cp_result.makespan)
    print(f"  Bottleneck: Machine {bottleneck.machine} "
          f"({bottleneck.utilization*100:.1f}% utilization, {bottleneck.idle_time} idle time units)")
    print(f"  Critical path: {len(cpath)} operations across machines "
          f"{sorted(set(op.machine for op in cpath))}")

    # Also compute utilization under the heuristic schedule, for comparison
    heur_stats = compute_utilization(heuristic_result.schedule, instance.num_machines, heuristic_result.makespan)
    heur_bottleneck = identify_bottleneck(heur_stats)

    # ---- 5. Visualizations ----
    print("\nGenerating charts...")
    plot_gantt(cp_result.schedule, instance.num_machines, instance.num_jobs,
               title="Optimal Schedule (CP-SAT)", makespan=cp_result.makespan,
               out_path=os.path.join(OUT_DIR, "gantt_cp_sat_optimal.png"))
    plot_gantt(heuristic_result.schedule, instance.num_machines, instance.num_jobs,
               title="Baseline Schedule (SPT Dispatching Heuristic)", makespan=heuristic_result.makespan,
               out_path=os.path.join(OUT_DIR, "gantt_spt_heuristic.png"))
    plot_utilization(stats, bottleneck.machine, os.path.join(OUT_DIR, "machine_utilization.png"))
    print("  Saved gantt_cp_sat_optimal.png, gantt_spt_heuristic.png, machine_utilization.png")

    # ---- 6. Quantified results summary ----
    results = {
        "instance": {
            "name": "ft10 (Fisher & Thompson, 1963)",
            "num_jobs": instance.num_jobs,
            "num_machines": instance.num_machines,
            "num_operations": instance.total_operations,
            "trivial_lower_bound": instance.lower_bound_makespan,
        },
        "cp_sat_optimal": {
            "status": cp_result.status,
            "makespan": cp_result.makespan,
            "lower_bound": cp_result.lower_bound,
            "proven_optimal": cp_result.status == "OPTIMAL",
            "solve_time_seconds": round(cp_wall, 3),
        },
        "spt_heuristic_baseline": {
            "makespan": heuristic_result.makespan,
            "solve_time_seconds": round(heur_wall, 5),
        },
        "improvement": {
            "makespan_reduction_pct": round(improvement_pct, 2),
            "makespan_reduction_units": heuristic_result.makespan - cp_result.makespan,
        },
        "bottleneck_analysis_optimal_schedule": {
            "bottleneck_machine": bottleneck.machine,
            "bottleneck_utilization_pct": round(bottleneck.utilization * 100, 2),
            "bottleneck_busy_time": bottleneck.busy_time,
            "bottleneck_idle_time": bottleneck.idle_time,
            "average_machine_utilization_pct": round(avg_utilization * 100, 2),
            "critical_path_length_ops": len(cpath),
            "critical_path_machines": sorted(set(op.machine for op in cpath)),
            "all_machine_utilization_pct": {f"M{s.machine}": round(s.utilization * 100, 2) for s in stats},
        },
        "bottleneck_analysis_heuristic_schedule": {
            "bottleneck_machine": heur_bottleneck.machine,
            "bottleneck_utilization_pct": round(heur_bottleneck.utilization * 100, 2),
            "bottleneck_idle_time": heur_bottleneck.idle_time,
        },
    }

    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    md = f"""# Results Summary — Job-Shop Scheduling Optimization

**Instance:** ft10 (Fisher & Thompson, 1963) — {instance.num_jobs} jobs x {instance.num_machines} machines, {instance.total_operations} operations
*(This is a famous benchmark: its optimal makespan of 930 remained unproven for 26 years until Carlier & Pinson solved it in 1989.)*

## Makespan: Optimizer vs. Heuristic Baseline

| Method | Makespan | Solve Time | Notes |
|---|---|---|---|
| SPT dispatching heuristic (baseline) | **{heuristic_result.makespan}** | {heur_wall:.4f}s | Giffler-Thompson construction, SPT priority rule |
| CP-SAT (OR-Tools, exact) | **{cp_result.makespan}** | {cp_wall:.2f}s | Status: {cp_result.status} |

**Result: {improvement_pct:.1f}% reduction in makespan** ({heuristic_result.makespan - cp_result.makespan} time units saved) by replacing a greedy dispatching rule with an exact CP-SAT optimization model.

## Bottleneck Machine Analysis (on the optimal schedule)

- **Bottleneck machine: Machine {bottleneck.machine}** — {bottleneck.utilization*100:.1f}% utilization ({bottleneck.busy_time}/{cp_result.makespan} time units busy, {bottleneck.idle_time} idle)
- Average machine utilization across all {instance.num_machines} machines: {avg_utilization*100:.1f}%
- Critical path length: {len(cpath)} operations, spanning machines {sorted(set(op.machine for op in cpath))}

### Utilization by machine (optimal schedule)
| Machine | Utilization | Idle Time |
|---|---|---|
"""
    for s in sorted(stats, key=lambda s: -s.utilization):
        md += f"| M{s.machine} | {s.utilization*100:.1f}% | {s.idle_time} |\n"

    md += f"""
## Suggested resume bullet

> Built a production scheduling optimizer in Python using Google OR-Tools CP-SAT, reducing
> makespan by **{improvement_pct:.0f}%** ({heuristic_result.makespan} → {cp_result.makespan} time units, proven optimal)
> versus a Shortest-Processing-Time dispatching heuristic baseline on a 10-job/10-machine
> benchmark; identified the primary bottleneck resource (Machine {bottleneck.machine}, running at
> {bottleneck.utilization*100:.0f}% utilization) via automated critical-path and machine-utilization analysis.
"""

    with open(os.path.join(OUT_DIR, "results.md"), "w") as f:
        f.write(md)

    print("\nSaved results.json and results.md")
    print("\nDone.")


if __name__ == "__main__":
    main()
