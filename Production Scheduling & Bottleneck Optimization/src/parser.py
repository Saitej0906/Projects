"""
parser.py
---------
Parses job-shop scheduling instances in Taillard benchmark format.

TAILLARD FORMAT
===============
Line 1:            <num_jobs> <num_machines>
Next num_jobs lines:  processing times matrix.
                      Row j = the processing time of each operation of job j,
                      listed IN THE ORDER the operations are performed.
Next num_jobs lines:  machine routing matrix.
                      Row j = the machine number (1-indexed in the file) for
                      each operation of job j, in the SAME order as the times
                      matrix above.

So operation k of job j has:
    duration = times[j][k]
    machine  = machines[j][k] - 1   (converted to 0-indexed)

This matters because job-shop (unlike flow-shop) allows every job to visit
machines in its own arbitrary order -- that routing matrix is what makes it
a genuine job-shop instance rather than a simplified flow-shop.
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class JobShopInstance:
    num_jobs: int
    num_machines: int
    # jobs[j] = list of (machine_id, duration) tuples, in operation order
    jobs: List[List[Tuple[int, int]]]

    @property
    def total_operations(self) -> int:
        return sum(len(ops) for ops in self.jobs)

    @property
    def lower_bound_makespan(self) -> int:
        """
        A trivial but useful lower bound: the makespan can never be less
        than the busiest machine's total workload, nor less than the
        longest single job's total processing time.
        """
        machine_load = [0] * self.num_machines
        for ops in self.jobs:
            for m, d in ops:
                machine_load[m] += d
        job_load = [sum(d for _, d in ops) for ops in self.jobs]
        return max(max(machine_load), max(job_load))


def parse_taillard(path: str) -> JobShopInstance:
    with open(path, "r") as f:
        tokens = f.read().split()

    idx = 0
    num_jobs = int(tokens[idx]); idx += 1
    num_machines = int(tokens[idx]); idx += 1

    times = []
    for _ in range(num_jobs):
        row = [int(tokens[idx + k]) for k in range(num_machines)]
        idx += num_machines
        times.append(row)

    machines = []
    for _ in range(num_jobs):
        row = [int(tokens[idx + k]) for k in range(num_machines)]
        idx += num_machines
        machines.append(row)

    jobs = []
    for j in range(num_jobs):
        ops = [(machines[j][k] - 1, times[j][k]) for k in range(num_machines)]
        jobs.append(ops)

    return JobShopInstance(num_jobs=num_jobs, num_machines=num_machines, jobs=jobs)


if __name__ == "__main__":
    inst = parse_taillard("data/ft10.txt")
    print(f"Parsed instance: {inst.num_jobs} jobs x {inst.num_machines} machines")
    print(f"Total operations: {inst.total_operations}")
    print(f"Trivial makespan lower bound: {inst.lower_bound_makespan}")
    print(f"Job 0 operations (machine, duration): {inst.jobs[0]}")
