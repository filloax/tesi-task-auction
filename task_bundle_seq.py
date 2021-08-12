import sys
from bundle_algo import BundleAlgorithm, TimeScoreFunction
from gen_task_bundle_positions import generate_positions, write_positions
import random
from disropt.functions import Variable
from disropt.problems import Problem
import numpy as np
import time
import argparse

from numpy.lib.function_base import average, select

parser = argparse.ArgumentParser(description='Distributed task assignment algorithm (CBBA).')
parser.add_argument('-n', '--agents', help="Amount of agents", required=True, type=int)
parser.add_argument('-t', '--tasks', help="Amount of tasks", required=True, type=int)
parser.add_argument('-L', '--agent-tasks', help="Max tasks per agent", required=True, type=int)
parser.add_argument('-v', '--verbose', default=False, action='store_true')
parser.add_argument('--test-only', default=False, action='store_true')
parser.add_argument('--test-runs', default=1, type=int)
args = parser.parse_args()

N = args.agents
num_tasks = args.tasks
max_agent_tasks = args.agent_tasks
verbose = args.verbose
test_only = args.test_only
runs = args.test_runs

agent_ids = list(range(N))
tasks = list(range(num_tasks))
# Analogo a
# [0, 1, 0, 0, 0],
# [1, 0, 1, 0, 0],
# [0, 1, 0, 1, 0],
# [0, 0, 1, 0, 1],
# [0, 0, 0, 1, 0]
links = np.roll(np.pad(np.eye(N), 1), 1, 0)[1:-1,1:-1] + np.roll(np.pad(np.eye(N), 1), -1, 0)[1:-1,1:-1]

links = links + np.eye(N) # Commentare per mettere g_ii = 0

def run(run = 0):
    # Diverso da CBAA: simula il caso specifico delle posizioni per dare un senso
    # alla generazione di un percorso, quindi non si limita a semplici bid casuali
    
    (agent_positions, task_positions) = generate_positions(N, num_tasks)
    write_positions(agent_positions, task_positions)

    def calc_time_fun(agent_id, task, path: list) -> float:
        if task in path:
            task_id = path.index(task)
            out = 0

            for i in range(task_id + 1):
                if i == 0:
                    out += linear_dist(agent_positions[agent_id], task_positions[path[i]])
                else:
                    out += linear_dist(task_positions[path[i - 1]], task_positions[path[i]])

            return out
        else:
            raise ValueError("Task not present in specified path!")

    # Inizializza dati degli agenti
    agents = [BundleAlgorithm(id, None, TimeScoreFunction(id, [0.9 for task in tasks], calc_time_fun), tasks, max_agent_tasks, agent_ids, verbose) for id in agent_ids]

    try:
        done = False
        i = 1
        start_time = time.time()
        while not done:
            order = list(range(N))
            random.shuffle(order)

            done = True
            for id in order:
                agents[id].construct_phase(i)

                agents[id].changed_ids = []
                # ricevi dati da altri agenti (invio è passivo)
                for other_id in agent_ids:
                    if other_id != id and links[id][other_id] == 1:
                        # time.sleep(random.random() * 0.01) # simula invio dati a momenti diversi
                        rec_time = time.time() - start_time
                        agents[id].winning_agents[other_id] = agents[other_id].winning_agents[other_id]
                        agents[id].winning_bids[other_id] = agents[other_id].winning_bids[other_id]
                        agents[id].message_times[other_id] = agents[other_id].message_times[other_id]
                        agents[id].changed_ids.append(other_id)
                        agents[id].message_times[id][other_id] = rec_time

                agents[id].conflict_resolve_phase(i)
                
                this_done = agents[id].check_done(i)

                done = done and this_done

            if verbose:
                print("Iter", i)
                print(np.array([agent.winning_bids[agent.id] for agent in agents]))
                print("-------------------")
                print("Winning agents:")
                print(np.array([agent.winning_agents[agent.id] for agent in agents]))
                print("-------------------")
                print("Task paths:")
                print("\n".join(["{}: {}".format(agent.id, agent.task_path) for agent in agents]))
                print("-------------------")
                print("Assigned tasks:")
                print(np.array([agent.assigned_tasks for agent in agents]))
                print("###################")

            i = i + 1

            # time.sleep(0.5)
            # input()

    except KeyboardInterrupt:
        print("Keyboard interrupt, early end...")

    # sol = np.array([agent.assigned_tasks for agent in agents])

    if not test_only:
        print("\n\nFinal version after", i, "iterations:")
        print("###################")
        print("Starting from:")
        print("Agent positions: \n{}".format({ id: agent_positions[id] for id in agent_ids }), file=sys.stderr)
        print("Task positions: \n{}".format({ id: task_positions[id] for id in tasks }), file=sys.stderr)
        print("###################")
        print(np.array([agent.winning_bids[agent.id] for agent in agents]))
        print("-------------------")
        print("Winning agents:")
        print(np.array([agent.winning_agents[agent.id] for agent in agents]))
        print("-------------------")
        print("Task paths:")
        print("\n".join(["{}: {}".format(agent.id, agent.task_path) for agent in agents]))
        print("-------------------")
        print("Assigned tasks:")
        print(np.array([agent.assigned_tasks for agent in agents]))
        print("###################")

def linear_dist(pos1, pos2):
    return np.sqrt(np.sum((pos1 - pos2) ** 2))

if __name__ == "__main__":
    if test_only:
        print("run,c * x (disropt),c * x (own), diff, diff %")

    results = []
    times = []

    for run_num in range(runs):
        pre_time = time.time()
        ret = run(run_num)
        times.append(time.time() - pre_time)
        results.append(ret)

    # print("---------------------")
    # print("Average time taken:", average(times))
    # print("Average pct diff: {}%".format(round(average([result[4] for result in results]), 2)))
    # print("No. of exact results: {}/{}".format(len(list(filter(lambda result: result[3] == 0, results))), runs))
    # print("Pct of exact results: {}%".format(round(100 * len(list(filter(lambda result: result[3] == 0, results))) / runs, 2)))