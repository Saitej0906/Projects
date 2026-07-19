"""
visualize.py
------------
Renders:
  1. A Gantt chart of a schedule (one row per machine, bars colored by job).
  2. A machine utilization bar chart, with the bottleneck machine highlighted.
"""

from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from cp_sat_solver import ScheduledOp
from bottleneck_analysis import MachineStats


def plot_gantt(schedule: List[ScheduledOp], num_machines: int, num_jobs: int,
               title: str, makespan: int, out_path: str):
    fig, ax = plt.subplots(figsize=(14, 0.55 * num_machines + 2))

    cmap = plt.get_cmap("tab20" if num_jobs > 10 else "tab10")
    job_colors = {j: cmap(j % cmap.N) for j in range(num_jobs)}

    for op in schedule:
        ax.barh(
            y=op.machine, width=op.duration, left=op.start, height=0.7,
            color=job_colors[op.job], edgecolor="black", linewidth=0.5,
        )
        if op.duration > makespan * 0.015:  # only label bars wide enough to read
            ax.text(op.start + op.duration / 2, op.machine, f"J{op.job}",
                     ha="center", va="center", fontsize=7, color="white", fontweight="bold")

    ax.set_yticks(range(num_machines))
    ax.set_yticklabels([f"Machine {m}" for m in range(num_machines)])
    ax.set_xlabel("Time")
    ax.set_title(f"{title}\nMakespan = {makespan}")
    ax.set_xlim(0, makespan * 1.02)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    handles = [mpatches.Patch(color=job_colors[j], label=f"Job {j}") for j in range(num_jobs)]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, ncol=1, title="Jobs")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_utilization(stats: List[MachineStats], bottleneck_machine: int, out_path: str):
    stats_sorted = sorted(stats, key=lambda s: s.machine)
    machines = [f"M{s.machine}" for s in stats_sorted]
    utils = [s.utilization * 100 for s in stats_sorted]
    colors = ["#d62728" if s.machine == bottleneck_machine else "#1f77b4" for s in stats_sorted]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(machines, utils, color=colors, edgecolor="black", linewidth=0.5)

    for bar, s in zip(bars, stats_sorted):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{s.utilization*100:.0f}%", ha="center", fontsize=8)

    ax.set_ylabel("Utilization (%)")
    ax.set_title("Machine Utilization (Optimal Schedule)\nRed = Bottleneck Machine")
    ax.set_ylim(0, 100)
    ax.axhline(y=sum(utils) / len(utils), color="gray", linestyle="--", linewidth=1,
               label=f"Average = {sum(utils)/len(utils):.1f}%")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
