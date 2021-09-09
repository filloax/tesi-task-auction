# lancio tramite
# mpirun -np N python taskAssignDisropt.py

from bundle_algo import BundleAlgorithm, DistanceScoreFunction
import time, sys, math
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent

import argparse
from task_positions import load_positions

start_time = time.time()

parser = argparse.ArgumentParser(description='Distributed task assignment algorithm.')
parser.add_argument('-v', '--verbose', required=False, nargs='?', const=1, type=int)
parser.add_argument('--agent-pos-path', type=str, default="./agent-positions.csv")
parser.add_argument('--task-pos-path', type=str, default="./task-positions.csv")
parser.add_argument('-L', '--agent-tasks', help="Max tasks per agent", required=True, type=int)
args = parser.parse_args()

comm = MPI.COMM_WORLD
num_agents = comm.Get_size()
local_rank = comm.Get_rank()

agent_pos_path = args.agent_pos_path
task_pos_path = args.task_pos_path
max_agent_tasks = args.agent_tasks
verbose = args.verbose

agent_ids = range(num_agents)

neighbors = []
if local_rank == 0:
    neighbors = [local_rank + 1]
elif local_rank == num_agents - 1:
    neighbors = [local_rank - 1]
else:
    neighbors = [local_rank - 1, local_rank + 1]

runs = 0

def main(): 
    np.random.seed(math.floor(local_rank + time.time()))

    if verbose:
        lprint("Neighbors:", neighbors)
    agent = Agent(in_neighbors=neighbors,
                out_neighbors=neighbors)

    (agent_positions, task_positions) = load_positions(agent_pos_path, task_pos_path)

    if len(agent_positions) != num_agents:
        raise ValueError("Number of agent positions wrong: {}, should be {}".format(len(agent_positions), num_agents))

    tasks = list(range(len(task_positions)))

    score_fun = DistanceScoreFunction(agent.id, agent_positions[local_rank], task_positions, 0.9)
    task_agent = BundleAlgorithm(agent.id, agent, score_fun, tasks, max_agent_tasks, agent_ids, verbose)

    task_agent.run(incr_runs)

    lprint("Done in {:.2f}s, {} iterations".format(time.time() - start_time, runs), file=sys.stderr)

    if verbose:
        result = ""
        result += "\n###################" + '\n'
        result += str("Results in {} runs:".format(runs)) + '\n'
        result += str(task_agent.winning_bids) + '\n'
        result += str("-------------------") + '\n'
        result += str("Winning agents:") + '\n'
        result += str(task_agent.winning_agents) + '\n'
        result += str("-------------------") + '\n'
        result += str("Task paths:") + '\n'
        result += str("{}: {}".format(agent.id, task_agent.task_path)) + '\n'
        result += str("-------------------") + '\n'
        result += str("Assigned tasks:") + '\n'
        result += str(task_agent.assigned_tasks) + '\n'
        result += str("###################") + '\n'

        formattedresult = ""
        for line in result.split("\n"):
            formattedresult = formattedresult + "{} | {}: ".format(local_rank, runs) + line + "\n"

        print(formattedresult)
    else:
        lprint(" ".join(map(lambda x: str(int(x)), task_agent.assigned_tasks)))

def linear_dist(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2) ** 2))

def lprint(*arg, file=sys.stdout):
    print("{} | {}: ".format(local_rank, runs), *arg, file=file)

def incr_runs():
    global runs
    runs = runs + 1

## Run
main()