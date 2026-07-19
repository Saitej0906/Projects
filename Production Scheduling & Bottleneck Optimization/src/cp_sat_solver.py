"""
cp_sat_solver.py
-----------------
Exact job-shop scheduling solver using Google OR-Tools CP-SAT.

THE OPTIMIZATION MODEL, EXPLAINED
==================================
Job-shop scheduling is: given J jobs, each a fixed SEQUENCE of operations,
and each operation requiring a specific machine for a specific duration,
find start times for every operation that

  (a) respect the order of operations within each job (job 3's op 2 can't
      start before job 3's op 1 finishes),
  (b) never double-book a machine (a machine can only do one operation at
      a time), and
  (c) minimize the makespan -- the time the LAST operation across all jobs
      finishes.

This is NP-hard (it's a generalization of the traveling salesman problem's
cousin, and ft10 itself was open for 26 years). CP-SAT solves it with
constraint programming rather than exhaustively trying schedules:

1. DECISION VARIABLES
   For every operation (job j, position k) we create an "interval variable":
   a triple (start, end, duration) where duration is fixed by the data and
   end = start + duration is enforced automatically by OR-Tools.

2. PRECEDENCE CONSTRAINTS (within a job)
   For consecutive operations in the same job, we force:
       start(op_k+1) >= end(op_k)
   This encodes "you can't start welding a part before it's been cut."

3. NO-OVERLAP CONSTRAINTS (per machine)
   We group all operations that need the same machine and add a single
   "NoOverlap" constraint over their interval variables. CP-SAT internally
   handles this with disjunctive reasoning (for any two operations on the
   same machine, EITHER one runs fully before the other, OR vice versa) --
   this is exactly what makes job-shop scheduling combinatorially explosive,
   and exactly what CP-SAT's specialized propagators are built to prune
   efficiently (much better than a generic MILP formulation could).

4. OBJECTIVE
   makespan = max(end time of every job's FINAL operation)
   We minimize this. CP-SAT proves optimality by simultaneously tightening
   an upper bound (best schedule found) and a lower bound (based on LP
   relaxations / cumulative reasoning) until they meet.

The result: a schedule that is PROVABLY optimal (not just "good"), along
with a formal optimality gap if the solver is stopped early.
"""

import collections
from dataclasses import dataclass
from typing import List

from ortools.sat.python import cp_model

from parser import JobShopInstance


@dataclass
class ScheduledOp:
    job: int
    op_index: int
    machine: int
    start: int
    end: int
    duration: int


@dataclass
class SolveResult:
    status: str
    makespan: int
    schedule: List[ScheduledOp]
    solve_time_seconds: float
    lower_bound: int


def solve_job_shop(instance: JobShopInstance, time_limit_seconds: float = 30.0) -> SolveResult:
    model = cp_model.CpModel()

    # Horizon: a safe upper bound on the makespan = sum of ALL processing
    # times (i.e. "run everything back-to-back with no parallelism at all").
    horizon = sum(d for ops in instance.jobs for _, d in ops)

    # task_type groups the three pieces OR-Tools needs per operation.
    task_type = collections.namedtuple("task_type", "start end interval")
    all_tasks = {}                                   # (job, op_index) -> task_type
    machine_to_intervals = collections.defaultdict(list)  # machine -> [interval vars]

    # ---- 1. Create interval variables for every operation ----
    for j, ops in enumerate(instance.jobs):
        for k, (machine, duration) in enumerate(ops):
            suffix = f"_{j}_{k}"
            start_var = model.NewIntVar(0, horizon, "start" + suffix)
            end_var = model.NewIntVar(0, horizon, "end" + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var, "interval" + suffix)
            all_tasks[(j, k)] = task_type(start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # ---- 2. Precedence constraints within each job ----
    for j, ops in enumerate(instance.jobs):
        for k in range(len(ops) - 1):
            model.Add(all_tasks[(j, k + 1)].start >= all_tasks[(j, k)].end)

    # ---- 3. No-overlap constraints per machine ----
    for machine, intervals in machine_to_intervals.items():
        model.AddNoOverlap(intervals)

    # ---- 4. Objective: minimize makespan ----
    makespan = model.NewIntVar(0, horizon, "makespan")
    last_ops = [all_tasks[(j, len(ops) - 1)].end for j, ops in enumerate(instance.jobs)]
    model.AddMaxEquality(makespan, last_ops)
    model.Minimize(makespan)

    # ---- Solve ----
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 8  # parallel search
    status = solver.Solve(model)

    status_name = solver.StatusName(status)
    schedule = []
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        for j, ops in enumerate(instance.jobs):
            for k, (machine, duration) in enumerate(ops):
                t = all_tasks[(j, k)]
                schedule.append(ScheduledOp(
                    job=j, op_index=k, machine=machine,
                    start=solver.Value(t.start), end=solver.Value(t.end),
                    duration=duration,
                ))
        schedule.sort(key=lambda s: (s.machine, s.start))

    return SolveResult(
        status=status_name,
        makespan=solver.Value(makespan) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else -1,
        schedule=schedule,
        solve_time_seconds=solver.WallTime(),
        lower_bound=int(solver.BestObjectiveBound()) if status in (cp_model.OPTIMAL, cp_model.FEASIBLE) else -1,
    )


if __name__ == "__main__":
    from parser import parse_taillard
    inst = parse_taillard("data/ft10.txt")
    result = solve_job_shop(inst, time_limit_seconds=30.0)
    print(f"Status: {result.status}")
    print(f"Makespan: {result.makespan}")
    print(f"Lower bound: {result.lower_bound}")
    print(f"Solve time: {result.solve_time_seconds:.2f}s")
