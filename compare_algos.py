import argparse
import math
import time
from threading import Thread
import sys

from disropt.functions import Variable
from disropt.problems.problem import Problem
import numpy as np
from numpy import matlib
from numpy.lib.function_base import average
from task_positions import generate_positions, linear_dist, write_positions
from task_assign_seq import TesterCBAA
from task_bundle_seq import TesterCBBA


parser = argparse.ArgumentParser(description='Test CBBA and CBAA.')
parser.add_argument('-N', '--num-agents', help="Amount of agents and tasks. Will be ignored if using positional arguments.", required=True, type=int)
parser.add_argument('-r', '--runs', default=1, type=int, help="Amount of test runs, if greater than 1 will disable X output and only check c * X.")
parser.add_argument('-v', '--verbose', default=False, action='store_true')
parser.add_argument('-p', '--iter-progress', default=False, action='store_true', help="Print progress during each run of the algorithms")
parser.add_argument('-T', '--prog-update-time', default=0.5, type=float, help="Time interval to update each run's progress bars")
parser.add_argument(dest='multi_agents', nargs='*', help="If you want to run more tests for agent numbers, use positional arguments.", type=int)

# Caratteri speciali
MOVE_TO_PREV_LINE_START = '\033[F'
# move_up_one_line = '\033[A'

class TestThread(Thread):
    def __init__(self, tester: (TesterCBBA or TesterCBAA), num_agents, agent_positions, task_positions, verbose = False):
        super().__init__()

        self.tester = tester
        self.num_agents = num_agents
        self.agent_positions = agent_positions
        self.task_positions = task_positions
        self.verbose = verbose
        self.result = None

    def run(self):
        time_before = time.time()
        ret = self.tester.run(
            num_agents = self.num_agents, 
            verbose = self.verbose, 
            agent_positions  = self.agent_positions,
            task_positions = self.task_positions,
            silent = not self.verbose,
            return_iterations = True,
            log_file = "log.txt",
            )
        tim = time.time() - time_before
        self.result = (ret[0], ret[1], tim)

def do_test(num_agents: int, runs: int, verbose = False, print_iter_progress = False, prog_update_time = 0.5):
    results = []

    num_tasks = num_agents

    print("Running with {} agents and {} tasks:\n".format(num_agents, num_tasks))

    for i in range(runs):
        if i == 0:
            printProgressBar(0, runs * 2, prefix="Runs", suffix="Complete (Doing CBAA)")

        if verbose:
            print("\n\n\n\n########################\n##### Run #{} #####\n########################\n\n\n\n".format(i))

        (agent_positions, task_positions) = generate_positions(num_agents, num_tasks, 
            test_tasks_same_distance=True, 
            test_same_distance=True, 
            test_same_positions=True
            )
        if runs == 1:
            write_positions(agent_positions, task_positions)

        #CBAA
        if verbose:
            print("Agent positions: \n{}".format({ id: agent_positions[id] for id in range(num_agents) }))
            print("Task positions: \n{}".format({ id: task_positions[id] for id in range(num_tasks) }))
            print("\n\n\n\n########################\n##### Running CBAA #####\n########################\n\n\n\n")

        tester_cbaa = TestThread(TesterCBAA(), num_agents, agent_positions, task_positions, verbose)
        if print_iter_progress:
            tester_cbaa.start()

            (done_agents, iterations, win_bids_progress) = tester_cbaa.tester.get_done_status()
            printIterationsProgress("CBAA", iterations, win_bids_progress, num_agents, True)
            while tester_cbaa.is_alive():
                time.sleep(prog_update_time)
                (done_agents, iterations, win_bids_progress) = tester_cbaa.tester.get_done_status()
                printIterationsProgress("CBAA", iterations, win_bids_progress, num_agents)
            printIterationsProgress("CBAA", iterations, win_bids_progress, num_agents, go_back_only=True)
        else:
            tester_cbaa.run() #stesso thread

        (sol_cbaa, iterations_cbaa, time_cbaa) = tester_cbaa.result
            
        printProgressBar(i * 2 + 1, runs * 2, prefix="Runs", suffix="Complete (Doing CBBA)")

        # CBBA
        if verbose:
            print("\n\n\n\n########################\n##### Running CBBA #####\n########################\n\n\n\n")

        tester_cbba = TestThread(TesterCBBA(), num_agents, agent_positions, task_positions, verbose)
        if print_iter_progress:
            tester_cbba.start()

            (done_agents, iterations, win_bids_progress) = tester_cbba.tester.get_done_status()
            printIterationsProgress("CBBA", iterations, win_bids_progress, num_agents, True)
            while tester_cbba.is_alive():
                time.sleep(prog_update_time)
                (done_agents, iterations, win_bids_progress) = tester_cbba.tester.get_done_status()
                printIterationsProgress("CBBA", iterations, win_bids_progress, num_agents)
            printIterationsProgress("CBBA", iterations, win_bids_progress, num_agents, go_back_only=True)
        else:
            tester_cbba.run() #stesso thread

        (sol_cbba, iterations_cbba, time_cbba) = tester_cbba.result
        if i == runs - 1:
            printProgressBar(i * 2 + 2, runs * 2, prefix="Runs", suffix="Complete")
        else:
            printProgressBar(i * 2 + 2, runs * 2, prefix="Runs", suffix="Complete (Doing CBAA)")


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

# Fonte: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd, flush=True)
    # Print New Line on Complete
    if iteration == total: 
        print()

def printIterationsProgress(suffix, iterations, win_bids_list, num_agents, start = False, go_back_only = False):
    required_const_bids = num_agents * 2 + 1
    if not start:
        print(MOVE_TO_PREV_LINE_START * (num_agents + 2))
    else:
        print()

    id_len = len(str(num_agents))
    id_str = "{:" + str(id_len) + "d}"

    if not go_back_only:
        print("Iteration #{:5d} | {}                   ".format(iterations, suffix))
        for i in range(num_agents):
            if len(win_bids_list) > i:
                pct_done = round(min(win_bids_list[i] * 100 / required_const_bids, 100), 2)
                # if pct_done > 50:
                print((id_str + ": {:>6.2f}% at {}/{} completion        ").format(i, pct_done, win_bids_list[i], required_const_bids))
                # else:
                #     print("{}: still unstable                 ".format(i))
            else:
                print((id_str + ":                                         ").format(i, len(win_bids_list)))
    else:
        for i in range(num_agents + 1):
            print("                                                       ")
        print(MOVE_TO_PREV_LINE_START * (num_agents + 3))

def main(agent_nums_to_test: list, runs: int, verbose = False, print_iter_progress = False, prog_update_time = 0.5):
    if len(agent_nums_to_test) == 0:
        raise ValueError("Needs at least one agent number to test with")
    elif len(agent_nums_to_test) == 1:
        do_test(agent_nums_to_test[0], runs, verbose, print_iter_progress, prog_update_time)
    else:
        pass

if __name__ == '__main__':
    args = parser.parse_args()

    multi_agents = args.multi_agents
    if len(multi_agents) == 0:
        multi_agents = [args.num_agents]

    main(multi_agents, args.runs, args.verbose, args.iter_progress, args.prog_update_time)