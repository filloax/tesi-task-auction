import argparse
import math
import time

from disropt.functions import Variable
from disropt.problems.problem import Problem
import numpy as np
from numpy import matlib
from numpy.lib.function_base import average
from task_positions import generate_positions, linear_dist, write_positions
from task_assign_seq import run as run_cbaa
from task_bundle_seq import run as run_cbba


parser = argparse.ArgumentParser(description='Test CBBA and CBAA.')
parser.add_argument('-N', '--num-agents', help="Amount of agents and tasks", required=True, type=int)
parser.add_argument('-r', '--runs', default=1, type=int, help="Amount of test runs, if greater than 1 will disable X output and only check c * X.")
parser.add_argument('-v', '--verbose', default=False, action='store_true')

def main(num_agents, runs, verbose = False):
    results = []

    num_tasks = num_agents

    print("Running with {} agents and {} tasks:\n".format(num_agents, num_tasks))

    for i in range(runs):
        if verbose:
            print("\n\n\n\n########################\n##### Run #{} #####\n########################\n\n\n\n".format(i))

        (agent_positions, task_positions) = generate_positions(num_agents, num_tasks, 
            test_tasks_same_distance=True, 
            test_same_distance=True, 
            test_same_positions=True
            )
        if runs == 1:
            write_positions(agent_positions, task_positions)

        if verbose:
            print("Agent positions: \n{}".format({ id: agent_positions[id] for id in range(num_agents) }))
            print("Task positions: \n{}".format({ id: task_positions[id] for id in range(num_tasks) }))
            print("\n\n\n\n########################\n##### Running CBAA #####\n########################\n\n\n\n")
        time_cbaa = time.time()
        (sol_cbaa, iterations_cbaa) = run_cbaa(num_agents, verbose, agent_positions=agent_positions, task_positions=task_positions, silent=not verbose, return_iterations=True)
        time_cbaa = time.time() - time_cbaa
        if verbose:
            print("\n\n\n\n########################\n##### Running CBBA #####\n########################\n\n\n\n")
        time_cbba = time.time()
        (sol_cbba, iterations_cbba) = run_cbba(num_agents, num_tasks, 1, verbose, agent_positions=agent_positions, task_positions=task_positions, silent=not verbose, return_iterations=True)
        time_cbba = time.time() - time_cbba


        x = Variable(num_agents * num_agents)

        c = np.array([[linear_dist(agent_positions[agent], task_positions[task]) for task in range(num_tasks)] for agent in range(num_agents)])
        c_line = -np.reshape(c, (num_agents * num_tasks, 1))

        obj_function = c_line @ x

        # I vincoli sono descritti nell'articolo di fonte
        sel_tasks = matlib.repmat(np.eye(num_tasks), 1, num_agents)
        sel_agents = np.array([ np.roll([1] * num_tasks + [0] * num_tasks * (num_agents - 1), y * num_tasks) for y in range(num_agents) ])

        constraints = [
            # Non assegnati più agent allo stesso task
            sel_tasks.T @ x <= np.ones((num_tasks, 1)), 
            # Non assegnati più di max_agent_tasks task allo stesso agent
            sel_agents.T @ x <= np.ones((num_agents, 1)), 
            # Non assegnati più o meno task del possibile in totale
            np.ones((num_agents * num_agents, 1)) @ x == num_agents,
            # X appartiene a 0 o 1
            x >= np.zeros((num_agents * num_agents, 1)),
            x <= np.ones((num_agents * num_agents, 1)),
        ]

        problem = Problem(obj_function, constraints)
        check_sol_line = problem.solve()
        check_sol = np.reshape(check_sol_line, (num_agents, num_tasks))

        cx_dis = float(-c_line.T @ check_sol_line)
        cx_cbaa = float(-c_line.T @ np.reshape(sol_cbaa, (num_agents * num_tasks, 1)))
        cx_cbba = float(-c_line.T @ np.reshape(sol_cbba, (num_agents * num_tasks, 1)))

        if runs == 1:
            print("Optimized tasks:\n{}".format(check_sol))
            print("CBAA assigned tasks:\n{}".format(sol_cbaa))
            print("CBBA assigned tasks:\n{}".format(sol_cbba))

            print("Optimized\tc * x: {}".format(cx_dis))
            print("CBAA\tc * x: {}\tdiff: {}%".format(cx_cbaa, round(100 * (cx_dis- cx_cbaa) / cx_dis, 2)))
            print("CBBA\tc * x: {}\tdiff: {}%".format(cx_cbba, round(100 * (cx_dis- cx_cbba) / cx_dis, 2)))
        else:
            results.append((cx_dis, cx_cbaa, cx_cbba, time_cbaa, time_cbba, iterations_cbaa, iterations_cbba))

    if runs > 1:
        cx_dis_avg = average([results[run][0] for run in range(runs)])
        cx_cbaa_avg = average([results[run][1] for run in range(runs)])
        cx_cbba_avg = average([results[run][2] for run in range(runs)])
        time_cbaa_avg = average([results[run][3] for run in range(runs)])
        time_cbba_avg = average([results[run][4] for run in range(runs)])
        iterations_cbaa_avg = average([results[run][5] for run in range(runs)])
        iterations_cbba_avg = average([results[run][6] for run in range(runs)])
        print("Results over {} runs:".format(runs))
        print("Avg. Optimized\tc * x: {}".format(round(cx_dis_avg, 2)))
        print("Avg. CBAA\tc * x: {}\tdiff: {}%\ttime: {}ms\titerations: {}".format(
            round(cx_cbaa_avg, 2), 
            round(100 * (cx_dis_avg- cx_cbaa_avg) / cx_dis_avg, 2), 
            math.floor(time_cbaa_avg * 1000),
            round(iterations_cbaa_avg, 2)),
            )
        print("Avg. CBBA\tc * x: {}\tdiff: {}%\ttime: {}ms\titerations: {}".format(
            round(cx_cbba_avg, 2), 
            round(100 * (cx_dis_avg- cx_cbba_avg) / cx_dis_avg, 2), 
            math.floor(time_cbba_avg * 1000),
            round(iterations_cbba_avg, 2)),
            )


if __name__ == '__main__':
    args = parser.parse_args()

    main(args.num_agents, args.runs, args.verbose)