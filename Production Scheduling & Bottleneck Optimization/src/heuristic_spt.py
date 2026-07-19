"""
heuristic_spt.py
-----------------
Baseline: a Shortest-Processing-Time (SPT) dispatching heuristic, built with
the Giffler & Thompson (1960) active-schedule-generation algorithm.

WHY NOT JUST "GREEDILY SCHEDULE JOBS IN ORDER"?
================================================
A naive heuristic (e.g., "always schedule whichever job is next in job ID
order") can produce schedules with pointless idle time -- gaps where a
machine sits empty even though a ready operation could have filled it.
Giffler & Thompson guarantees an ACTIVE schedule: no operation could be
moved earlier without delaying another, which is the right way to give a
dispatching rule (like SPT) a fair fight against the exact optimizer.

THE ALGORITHM, STEP BY STEP
============================
Maintain a "conflict set" G: for every job, its next not-yet-scheduled
operation (one operation per unfinished job).

Repeat until every operation is scheduled:
  1. For each operation in G, compute the earliest possible completion
     time if it were scheduled next: 
         earliest_start = max(job_ready_time, machine_ready_time)
         earliest_completion = earliest_start + duration
  2. Find the operation o* in G with the SMALLEST earliest_completion time,
     and note its machine m*. This identifies the true bottleneck decision:
     "machine m* will finish its next job-eligible task at time c* at the
     very latest, no matter what we do."
  3. Among all operations in G that also need machine m* AND whose
     earliest_start is < c* (i.e., they COULD have run before c* and are
     therefore in genuine conflict for that machine), pick ONE using the
     priority rule -- here, Shortest Processing Time (smallest duration).
  4. Schedule the chosen operation at its earliest_start. Update that
     machine's ready time and that job's ready time.
  5. Remove the scheduled operation from G; if its job has a next
     operation, add that to G.

This produces a complete, active schedule using only a greedy, local
priority rule -- no lookahead, no backtracking -- which is exactly what
makes dispatching heuristics fast (O(n^2)-ish) but generally suboptimal.
Comparing its makespan against CP-SAT's proven optimum is the "before vs
after" story for the project.
"""

from dataclasses import dataclass
from typing import List

from parser import JobShopInstance
from cp_sat_solver import ScheduledOp


def solve_spt_heuristic(instance: JobShopInstance) -> "HeuristicResult":
    num_jobs = instance.num_jobs

    # next_op_index[j] = index of job j's next operation to schedule
    next_op_index = [0] * num_jobs
    job_ready_time = [0] * num_jobs
    machine_ready_time = [0] * instance.num_machines
    ops_remaining = [len(ops) for ops in instance.jobs]

    schedule: List[ScheduledOp] = []
    total_ops = instance.total_operations

    while len(schedule) < total_ops:
        # Build conflict set G: current candidate operation for every
        # job that still has operations left.
        candidates = []  # (job, op_index, machine, duration, earliest_start, earliest_completion)
        for j in range(num_jobs):
            if ops_remaining[j] == 0:
                continue
            k = next_op_index[j]
            machine, duration = instance.jobs[j][k]
            earliest_start = max(job_ready_time[j], machine_ready_time[machine])
            earliest_completion = earliest_start + duration
            candidates.append((j, k, machine, duration, earliest_start, earliest_completion))

        # Step 2: find the operation with minimum earliest completion time
        j_star, k_star, m_star, d_star, es_star, ec_star = min(candidates, key=lambda c: c[5])

        # Step 3: among operations needing machine m_star that could
        # conflict with it (earliest_start < ec_star), pick shortest duration
        conflict_set = [c for c in candidates if c[2] == m_star and c[4] < ec_star]
        chosen = min(conflict_set, key=lambda c: c[3])  # SPT tie-break rule
        j, k, machine, duration, earliest_start, _ = chosen

        # Step 4: commit the chosen operation to that earliest start
        start = earliest_start
        end = start + duration
        schedule.append(ScheduledOp(job=j, op_index=k, machine=machine,
                                     start=start, end=end, duration=duration))
        job_ready_time[j] = end
        machine_ready_time[machine] = end
        next_op_index[j] += 1
        ops_remaining[j] -= 1

    schedule.sort(key=lambda s: (s.machine, s.start))
    makespan = max(s.end for s in schedule)
    return HeuristicResult(makespan=makespan, schedule=schedule)


@dataclass
class HeuristicResult:
    makespan: int
    schedule: List[ScheduledOp]


if __name__ == "__main__":
    from parser import parse_taillard
    inst = parse_taillard("data/ft10.txt")
    result = solve_spt_heuristic(inst)
    print(f"SPT heuristic makespan: {result.makespan}")
