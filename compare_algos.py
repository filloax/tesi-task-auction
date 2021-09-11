import argparse
from itertools import count
import math
import time
from threading import Thread
import os
from typing import Iterator
from utils import bids_to_string, sol_to_string

from disropt.functions import Variable
from disropt.problems.problem import Problem
import numpy as np
from numpy import matlib
from numpy.lib.function_base import average
from task_positions import generate_positions, linear_dist, write_positions
from task_assign_seq import TesterCBAA
from task_bundle_seq import TesterCBBA
from task_seq_tester import TaskTester


parser = argparse.ArgumentParser(description='Test CBBA and CBAA.')
parser.add_argument('-N', '--num-agents', help="Amount of agents and tasks. Will be ignored if using positional arguments.", required=True, type=int)
parser.add_argument('-r', '--runs', default=1, type=int, help="Amount of test runs, if greater than 1 will disable X output and only check c * X.")
parser.add_argument('--test-same-score', default=False, action='store_true', help="If set, there will always be tasks with the same position as an " + 
    "agent, the same position of another task, at the same distance of another task, etc to test if bid tie-breaking works")
parser.add_argument('-v', '--verbose', default=False, action='store_true')
parser.add_argument('-p', '--iter-progress', default=False, action='store_true', help="Print progress during each run of the algorithms")
parser.add_argument('-T', '--prog-update-time', default=0.5, type=float, help="Time interval to update each run's progress bars")
parser.add_argument(dest='multi_agents', nargs='*', help="If you want to run more tests for agent numbers, use positional arguments.", type=int)

# Caratteri speciali
MOVE_TO_PREV_LINE_START = '\033[F'
# move_up_one_line = '\033[A'

class TestThread(Thread):
    def __init__(self, tester: TaskTester, num_agents, agent_positions, task_positions, verbose = False):
        super().__init__()

        self.tester = tester
        self.num_agents = num_agents
        self.agent_positions = agent_positions
        self.task_positions = task_positions
        self.verbose = verbose
        self.result = None

    def run(self):
        if os.path.exists("log.txt"):
            os.remove("log.txt")

        with open("log.txt", "w") as logfile:
            print("Agents:", file=logfile)
            for i in range(len(self.agent_positions)):
                print("{}: [{}]".format(i, ' '.join(str(x) for x in self.agent_positions[i])), file=logfile)

            print("Tasks:", file=logfile)
            for i in range(len(self.task_positions)):
                print("{}: [{}]".format(i, ' '.join(str(x) for x in self.task_positions[i])), file=logfile)

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

def do_test(num_agents: int, runs: int, verbose = False, print_iter_progress = False, prog_update_time = 0.5,
    test_same_score = True):
    results = []

    num_tasks = num_agents

    np.set_printoptions(precision = 3, linewidth = 999)

    print("Running with {} agents and {} tasks:\n".format(num_agents, num_tasks))

    for i in range(runs):
        if i == 0:
            printProgressBar(0, runs * 2, prefix="Runs", suffix="Complete (Doing CBAA)")

        if verbose:
            print("\n\n\n\n########################\n##### Run #{} #####\n########################\n\n\n\n".format(i))

        (agent_positions, task_positions) = generate_positions(num_agents, num_tasks, 
            test_tasks_same_distance=test_same_score, 
            test_same_distance=test_same_score, 
            test_same_positions=test_same_score,
            test_far_agent = False,
            )
        if runs == 1:
            write_positions(agent_positions, task_positions)

        #CBAA
        if verbose:
            print("Agent positions: \n{}".format(agent_positions))
            print("Task positions: \n{}".format(task_positions))
            print("\n\n\n\n########################\n##### Running CBAA #####\n########################\n\n\n\n")

        tester_cbaa = TestThread(TesterCBAA(), num_agents, agent_positions, task_positions, verbose)
        if print_iter_progress:
            tester_cbaa.start()

            (done_agents, iterations, win_bids_progress, last_solution) = tester_cbaa.tester.get_cur_status()
            printIterationsProgress("CBAA", iterations, win_bids_progress, num_agents, last_solution, True)
            while tester_cbaa.is_alive():
                time.sleep(prog_update_time)
                (done_agents, iterations, win_bids_progress, last_solution) = tester_cbaa.tester.get_cur_status()
                printIterationsProgress("CBAA", iterations, win_bids_progress, num_agents, last_solution)
            printIterationsProgress("CBAA", iterations, win_bids_progress, num_agents, last_solution, go_back_only=True)
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

            (done_agents, iterations, win_bids_progress, last_solution) = tester_cbba.tester.get_cur_status()
            printIterationsProgress("CBBA", iterations, win_bids_progress, num_agents, last_solution, True)
            while tester_cbba.is_alive():
                time.sleep(prog_update_time)
                (done_agents, iterations, win_bids_progress, last_solution) = tester_cbba.tester.get_cur_status()
                printIterationsProgress("CBBA", iterations, win_bids_progress, num_agents, last_solution)
            printIterationsProgress("CBBA", iterations, win_bids_progress, num_agents, last_solution, go_back_only=True)
        else:
            tester_cbba.run() #stesso thread

        (sol_cbba, iterations_cbba, time_cbba) = tester_cbba.result
        if i == runs - 1:
            printProgressBar(i * 2 + 2, runs * 2, prefix="Runs", suffix="Complete")
        else:
            printProgressBar(i * 2 + 2, runs * 2, prefix="Runs", suffix="Complete (Doing CBAA)")


        x = Variable(num_agents * num_agents)

        c = np.array([[0.9 ** linear_dist(agent_positions[agent], task_positions[task]) for task in range(num_tasks)] for agent in range(num_agents)])
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
            print("Centralized bids:\n{}".format(bids_to_string(c)))
            print("CBAA bids:\n{}".format(bids_to_string(np.array([agent.bids for agent in tester_cbaa.tester.agents]))))
            print("CBBA bids:\n{}".format(bids_to_string(np.array([agent.bids for agent in tester_cbba.tester.agents]))))

            print("Optimized tasks:\n{}".format(sol_to_string(sol=check_sol)))
            print("CBAA assigned tasks:\n{}".format(sol_to_string(sol=sol_cbaa)))
            print("CBBA assigned tasks:\n{}".format(sol_to_string(sol=sol_cbba)))

            if tester_cbaa.tester.has_conflicts(sol_cbaa):
                print("WARNING: CBAA solution has conflicts!")
            if tester_cbba.tester.has_conflicts(sol_cbba):
                print("WARNING: CBBA solution has conflicts!")

            print("Optimized\tc * x: {}".format(cx_dis))
            print("CBAA\tc * x: {}\tdiff: {}%".format(cx_cbaa, round(100 * (cx_dis- cx_cbaa) / cx_dis, 2)))
            print("CBBA\tc * x: {}\tdiff: {}%".format(cx_cbba, round(100 * (cx_dis- cx_cbba) / cx_dis, 2)))
        else:
            results.append((cx_dis, cx_cbaa, cx_cbba, time_cbaa, time_cbba, iterations_cbaa, iterations_cbba, 
                tester_cbaa.tester.has_conflicts(sol_cbaa), tester_cbba.tester.has_conflicts(sol_cbba)))

    if runs > 1:
        cx_dis_avg = average([results[run][0] for run in range(runs)])
        cx_cbaa_avg = average([results[run][1] for run in range(runs)])
        cx_cbba_avg = average([results[run][2] for run in range(runs)])
        time_cbaa_avg = average([results[run][3] for run in range(runs)])
        time_cbba_avg = average([results[run][4] for run in range(runs)])
        iterations_cbaa_avg = average([results[run][5] for run in range(runs)])
        iterations_cbba_avg = average([results[run][6] for run in range(runs)])
        conflicts_cbaa = len(list(filter(None, [results[run][7] for run in range(runs)])))
        conflicts_cbba = len(list(filter(None, [results[run][8] for run in range(runs)])))
        print(MOVE_TO_PREV_LINE_START + "Results over {} runs:".format(runs) + " " * 70)
        print("Avg. Optimized\tc * x: {}".format(round(cx_dis_avg, 2)))
        print("Avg. CBAA\tc * x: {}\tdiff: {}%\ttime: {}ms\titerations: {}".format(
            round(cx_cbaa_avg, 2), 
            round(100 * (cx_dis_avg - cx_cbaa_avg) / cx_dis_avg, 2), 
            math.floor(time_cbaa_avg * 1000),
            round(iterations_cbaa_avg, 2)),
            )
        print("Avg. CBBA\tc * x: {}\tdiff: {}%\ttime: {}ms\titerations: {}".format(
            round(cx_cbba_avg, 2), 
            round(100 * (cx_dis_avg - cx_cbba_avg) / cx_dis_avg, 2), 
            math.floor(time_cbba_avg * 1000),
            round(iterations_cbba_avg, 2)),
            )
        if conflicts_cbaa > 0:
            print(f'WARNING: CBAA had {conflicts_cbaa} conflicts!')
        if conflicts_cbba > 0:
            print(f'WARNING: CBAA had {conflicts_cbba} conflicts!')

# Fonte: https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
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

def printIterationsProgress(suffix, iterations, win_bids_list, num_agents, solution: np, start = False, go_back_only = False):
    required_const_bids = num_agents * 2 + 1

    id_len = len(str(num_agents))
    id_str = "{:" + str(id_len) + "d}"
    str_1_len = id_len + 31
    sol_lines = sol_to_string(sol=solution).split("\n") if solution is not None and solution.size > 0 else ["" for i in range(num_agents)]
    # sol_line_len = max(len(line) for line in sol_lines) if solution is not None and solution.size > 0 else 3 + 2 * num_agents
    sol_line_len = 3 + 2 * num_agents

    expected_line_len = str_1_len + sol_line_len
    print_solution = True
    if expected_line_len > 76:
        print_solution = False

    if not start:
        print(MOVE_TO_PREV_LINE_START * (num_agents + 2), flush=True)
    else:
        print(flush=True)

    if not go_back_only:
        print("Iteration #{:5d} | {}                   ".format(iterations, suffix), flush=True)
        for i in range(num_agents):
            base_str = ""
            if len(win_bids_list) > i:
                pct_done = round(min(win_bids_list[i] * 100 / required_const_bids, 100), 2)
                win_bids = win_bids_list[i]
                if win_bids > required_const_bids:
                    win_bids = str(required_const_bids) + "+"
                base_str = (id_str + ": {:>6.2f}% at {}/{} completion").format(i, pct_done, win_bids, required_const_bids)
            else:
                base_str = (id_str + ":").format(i, len(win_bids_list))

            out = base_str.ljust(str_1_len)
            if print_solution:
                out += sol_lines[i].ljust(sol_line_len)
            elif solution is not None and solution.size > 0:
                if np.any(solution[i] == 1):
                    out += (" | selected: " + "{:" + str(id_len) + "d}").format(next(j for j in range(len(solution[i])) if solution[i][j] == 1))
                else:
                    out += " | selected: -"
            print(out, flush=True)
    else:
        for i in range(num_agents + 1):
            print(' ' * (sol_line_len + str_1_len), flush=True)
        print(MOVE_TO_PREV_LINE_START * (num_agents + 3), flush=True)

def main(agent_nums_to_test: list, runs: int, verbose = False, print_iter_progress = False, prog_update_time = 0.5, test_same_score = True):
    if len(agent_nums_to_test) == 0:
        raise ValueError("Needs at least one agent number to test with")
    elif len(agent_nums_to_test) == 1:
        do_test(agent_nums_to_test[0], runs, verbose, print_iter_progress, prog_update_time, test_same_score)
    else:
        pass

if __name__ == '__main__':
    args = parser.parse_args()

    multi_agents = args.multi_agents
    if len(multi_agents) == 0:
        multi_agents = [args.num_agents]

    main(multi_agents, args.runs, args.verbose, args.iter_progress, args.prog_update_time, args.test_same_score)