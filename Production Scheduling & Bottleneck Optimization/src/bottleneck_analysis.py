"""
bottleneck_analysis.py
-----------------------
Identifies the bottleneck machine from a solved schedule.

WHAT "BOTTLENECK" MEANS HERE
=============================
Given a schedule with a fixed makespan M, every machine is busy for some
total amount of time (the sum of its operations' durations) and idle for
the rest (M minus that busy time). We define:

    utilization(machine) = busy_time(machine) / makespan

The machine with the HIGHEST utilization is doing the least "waiting
around" -- it is the resource most constantly in demand, and therefore the
most likely candidate for a bottleneck: the machine you'd invest in
(add capacity to, speed up, or de-prioritize maintenance downtime on) if
you wanted to reduce the makespan further.

We also report the CRITICAL MACHINE SET: machines that appear on the
critical path -- the chain of operations from time 0 to the makespan where
NO operation has any slack (delaying it would delay the entire schedule).
We compute this via a backward pass from whichever operation finishes
exactly at the makespan.
"""

from dataclasses import dataclass
from typing import List

from cp_sat_solver import ScheduledOp
from parser import JobShopInstance


@dataclass
class MachineStats:
    machine: int
    busy_time: int
    idle_time: int
    utilization: float
    num_operations: int


def compute_utilization(schedule: List[ScheduledOp], num_machines: int, makespan: int) -> List[MachineStats]:
    busy = [0] * num_machines
    count = [0] * num_machines
    for op in schedule:
        busy[op.machine] += op.duration
        count[op.machine] += 1

    stats = []
    for m in range(num_machines):
        idle = makespan - busy[m]
        util = busy[m] / makespan if makespan > 0 else 0.0
        stats.append(MachineStats(machine=m, busy_time=busy[m], idle_time=idle,
                                   utilization=util, num_operations=count[m]))
    return stats


def identify_bottleneck(stats: List[MachineStats]) -> MachineStats:
    return max(stats, key=lambda s: s.utilization)


def critical_path(schedule: List[ScheduledOp], instance: JobShopInstance, makespan: int) -> List[ScheduledOp]:
    """
    Backward-trace the chain of zero-slack operations that determines the
    makespan. Starting from whichever operation ends exactly at `makespan`,
    repeatedly step back to whichever "cause" (same-job predecessor, or
    same-machine predecessor) ends exactly when this operation starts.
    """
    by_job_op = {(s.job, s.op_index): s for s in schedule}
    by_machine = {}
    for s in schedule:
        by_machine.setdefault(s.machine, []).append(s)
    for m in by_machine:
        by_machine[m].sort(key=lambda s: s.start)

    # Find an operation ending at the makespan
    current = next(s for s in schedule if s.end == makespan)
    path = [current]

    while True:
        candidates = []

        # Same-job predecessor
        if current.op_index > 0:
            pred = by_job_op[(current.job, current.op_index - 1)]
            if pred.end == current.start:
                candidates.append(pred)

        # Same-machine predecessor (the op immediately before this one
        # on its machine, if it ends exactly when this one starts)
        machine_ops = by_machine[current.machine]
        idx = machine_ops.index(current)
        if idx > 0:
            pred = machine_ops[idx - 1]
            if pred.end == current.start:
                candidates.append(pred)

        if not candidates:
            break
        current = candidates[0]
        path.append(current)

    path.reverse()
    return path


if __name__ == "__main__":
    from parser import parse_taillard
    from cp_sat_solver import solve_job_shop

    inst = parse_taillard("data/ft10.txt")
    result = solve_job_shop(inst, time_limit_seconds=30.0)
    stats = compute_utilization(result.schedule, inst.num_machines, result.makespan)
    bottleneck = identify_bottleneck(stats)

    print("Machine utilization:")
    for s in sorted(stats, key=lambda s: -s.utilization):
        print(f"  M{s.machine}: {s.utilization*100:5.1f}% busy, "
              f"{s.idle_time:4d} idle, {s.num_operations} ops")
    print(f"\nBottleneck machine: M{bottleneck.machine} "
          f"({bottleneck.utilization*100:.1f}% utilization, {bottleneck.idle_time} idle)")

    cp = critical_path(result.schedule, inst, result.makespan)
    print(f"\nCritical path length: {len(cp)} operations")
    print("Machines on critical path (in order):", [op.machine for op in cp])
