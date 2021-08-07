# lancio tramite
# mpirun -np N python taskAssignDisropt.py

from auction_algo import AuctionAlgorithm
import time, sys, math
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent

import argparse

start_time = time.time()

parser = argparse.ArgumentParser(description='Distributed task assignment algorithm.')
parser.add_argument('-v', '--verbose', required=False, nargs='?', const=1, type=int)
args = parser.parse_args()

comm = MPI.COMM_WORLD
num_agents = comm.Get_size()
local_rank = comm.Get_rank()

agent_ids = range(num_agents)

neighbors = []
if local_rank == 0:
    neighbors = [local_rank + 1]
elif local_rank == num_agents - 1:
    neighbors = [local_rank - 1]
else:
    neighbors = [local_rank - 1, local_rank + 1]

tasks = list(range(num_agents)) # J rappresentato con range(num_agents), 0..4 invece di 1..5

runs = 0

def lprint(*arg):
    print(local_rank, "#{}".format(runs), *arg)

def incr_runs():
    global runs
    runs = runs + 1

## Run
        
np.random.seed(math.floor(local_rank + time.time()))

if args.verbose:
    lprint("Neighbors:", neighbors)
agent = Agent(in_neighbors=neighbors,
            out_neighbors=neighbors)
bids = np.random.random(len(tasks))
task_agent = AuctionAlgorithm(agent.id, bids, agent, tasks, agent_ids)

if args.verbose:
    lprint("Bids for agent {}: {}".format(task_agent.id, task_agent.bids))

task_agent.run(incr_runs)


print("{}: Done in {:.2f}s, {} iterations".format(local_rank, time.time() - start_time, runs), file=sys.stderr)

if args.verbose:
    print("\n###################")
    lprint("Results in {} runs:".format(runs))
    lprint("Assigned tasks:")
    lprint(task_agent.assigned_tasks)
    lprint("-------------------")
    lprint("Max bids:")
    lprint(task_agent.max_bids[task_agent.id])
    lprint("-------------------")
    lprint("Bids:")
    lprint(task_agent.bids)
    lprint("Done!")
    print("###################\n")
else:
    print("{}:{}".format(task_agent.id, " ".join(map(lambda x: str(int(x)), task_agent.assigned_tasks))))
