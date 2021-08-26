from ast import Num
import itertools
import math
import sys
from utils import bids_to_string, sol_to_string
from task_seq_tester import TaskTester
from bundle_algo import BundleAlgorithm, TimeScoreFunction
from task_positions import gen_distance_calc_time_fun, generate_positions, load_positions, write_positions
import random
from disropt.functions import Variable
from disropt.problems import Problem
import numpy as np
import numpy.matlib as matlib
import time
import argparse

from numpy.lib.function_base import average, select

parser = argparse.ArgumentParser(description='Distributed task assignment algorithm (CBBA).')
parser.add_argument('-N', '--agents', help="Amount of agents", required=True, type=int)
parser.add_argument('-t', '--tasks', help="Amount of tasks", required=True, type=int)
parser.add_argument('-L', '--agent-tasks', help="Max tasks per agent", required=True, type=int)
parser.add_argument('--load-pos', help="Use last generated positions instead of creating new", default=False, action='store_true')
parser.add_argument('-v', '--verbose', default=False, action='store_true')
parser.add_argument('--log', default='', type=str, help='Log file path to print verbose output to (doesn\'t need -v)')
parser.add_argument('--test-mode', default='', type=str, help="conflict: check conflicts, test: check efficiency")
parser.add_argument('--test-runs', help="Do a run for each combination of N, t, L up to the amount specified", default=False, action='store_true')
parser.add_argument('-r', '--runs', default=1, type=int)

class TesterCBBA(TaskTester):
    def __init__(self):
        super().__init__()

    def run(self, num_agents, num_tasks = -1, max_agent_tasks = 1, verbose = False, test_mode = False, run = 0, agent_positions = None, task_positions = None, 
    silent = False, return_iterations = False, log_file=''):
        if num_tasks < 0:
            num_tasks = num_agents
            
        self.log_file = log_file

        agent_ids = list(range(num_agents))
        tasks = list(range(num_tasks))
        # Analogo a
        # [0, 1, 0, 0, 0],
        # [1, 0, 1, 0, 0],
        # [0, 1, 0, 1, 0],
        # [0, 0, 1, 0, 1],
        # [0, 0, 0, 1, 0]
        links = np.roll(np.pad(np.eye(num_agents), 1), 1, 0)[1:-1,1:-1] + np.roll(np.pad(np.eye(num_agents), 1), -1, 0)[1:-1,1:-1]

        links = links + np.eye(num_agents) # Commentare per mettere g_ii = 0
        
        # Diverso da CBAA: simula il caso specifico delle posizioni per dare un senso
        # alla generazione di un percorso, quindi non si limita a semplici bid casuali
        
        if agent_positions is None or task_positions is None:
            (agent_positions, task_positions) = generate_positions(num_agents, num_tasks)
            if not silent:
                write_positions(agent_positions, task_positions)

        # Inizializza dati degli agenti
        agents = [BundleAlgorithm(id, 
            agent = None, 
            score_function = TimeScoreFunction(id, [0.9 for task in tasks], gen_distance_calc_time_fun(agent_positions, task_positions)), 
            tasks = tasks, 
            max_agent_tasks = max_agent_tasks, 
            agent_ids = agent_ids, 
            verbose = verbose,
            log_file = log_file,
            ) for id in agent_ids]

        self.agents = agents # for external access

        self.iterations = 0
        self.const_winning_bids = [0 for agent in agents]
        force_print = False

        try:
            self.done_agents = 0

            start_time = time.time()
            while self.done_agents < num_agents:
                self.iterations = self.iterations + 1

                order = list(range(num_agents))
                random.shuffle(order)

                for id in order:
                    agents[id].construct_phase(self.iterations)

                    agents[id].changed_ids = []
                    # ricevi dati da altri agenti (invio è passivo)
                    for other_id in agent_ids:
                        if (self.iterations > 1 or order.index(other_id) < order.index(id)) and other_id != id and links[id][other_id] == 1:
                            # time.sleep(random.random() * 0.01) # simula invio dati a momenti diversi
                            rec_time = time.time() - start_time
                            agents[id].winning_agents[other_id] = agents[other_id].winning_agents[other_id]
                            agents[id].winning_bids[other_id] = agents[other_id].winning_bids[other_id]
                            agents[id].message_times[other_id] = agents[other_id].message_times[other_id]
                            agents[id].changed_ids.append(other_id)
                            agents[id].message_times[id][other_id] = rec_time

                random.shuffle(order)
                for id in order:
                    agents[id].conflict_resolve_phase(self.iterations)
                    
                    if agents[id].check_done(self.iterations):
                        self.done_agents += 1

                    self.const_winning_bids[id] = agents[id].win_bids_equal_cnt

                sol = np.array([agent.assigned_tasks for agent in agents])
                self.last_solution = sol

                self.log("Iter", self.iterations, do_console=verbose)
                self.log("Bids:", do_console=verbose)
                self.log(bids_to_string([agent.bids for agent in agents]), do_console=verbose)
                self.log("Winning bids:", do_console=verbose)
                self.log(bids_to_string([agent.winning_bids[agent.id] for agent in agents]), do_console=verbose)
                self.log("-------------------", do_console=verbose)
                self.log("Winning agents:", do_console=verbose)
                self.log(np.array([agent.winning_agents[agent.id] for agent in agents]), do_console=verbose)
                self.log("-------------------", do_console=verbose)
                self.log("Task paths:", do_console=verbose)
                self.log("\n".join(["{}: {}".format(agent.id, agent.task_path) for agent in agents]), do_console=verbose)
                self.log("-------------------", do_console=verbose)
                self.log("Assigned tasks:", do_console=verbose)
                self.log(sol_to_string(sol=sol), do_console=verbose)
                self.log("###################", do_console=verbose)

                # time.sleep(0.5)
                # input()

        except KeyboardInterrupt or BrokenPipeError:
            print("CBBA: Keyboard interrupt, early end...")
            print("CBBA: Keyboard interrupt, early end...", file=sys.stderr)
            force_print = True
        finally:
            if force_print or not silent:
                sol = np.array([agent.assigned_tasks for agent in agents])

                self.log("\n\nFinal version after", self.iterations, "iterations:")
                self.log("###################")
                self.log("Starting from:")
                self.log("Agent positions: \n{}".format(agent_positions))
                self.log("Task positions: \n{}".format(task_positions))
                self.log("L: {}".format(max_agent_tasks))
                self.log("###################")
                self.log("Winning bids:")
                self.log(bids_to_string([agent.winning_bids[agent.id] for agent in agents]))
                self.log("-------------------")
                self.log("Winning agents:")
                self.log(np.array([agent.winning_agents[agent.id] for agent in agents]))
                self.log("-------------------")
                self.log("Task paths:")
                self.log("\n".join(["{}: {}".format(agent.id, agent.task_path) for agent in agents]))
                self.log("-------------------")
                self.log("Assigned tasks:")
                self.log(sol_to_string(agents))

                if self.has_conflicts(sol):
                    self.log("!!WARNING!!: HAS TASK ASSIGNMENT CONFLICTS!")

                self.log("###################")


        sol = np.array([agent.assigned_tasks for agent in agents])
        self.last_solution = sol

        if test_mode:
            self.log("\nControllo efficienza tramite Disropt", do_console=not silent)

            x = Variable(num_agents * num_tasks)

            # Calcolo centralizzato di c_{ij}
            # Per ogni agente i, le offerte poste su ogni task sono il massimo guadagno possibile che può 
            # ottenere da quel task, controllando ogni combinazione possibile di path
            c = np.zeros((num_agents, num_tasks))

            self.log("\nCalcolo c [N: {} / t: {}]:".format(num_agents, num_tasks), do_console=not silent)

            for i in agent_ids:
                score_fun = TimeScoreFunction(i, [0.9 for task in tasks], gen_distance_calc_time_fun(agent_positions, task_positions))
                for j in tasks:
                    tasks_except_this = tasks.copy()
                    tasks_except_this.remove(j)
                    # Iteratore per ogni percorso possibile che non includa questo task
                    all_paths_iter = itertools.chain(*(itertools.permutations(tasks_except_this, l) for l in range(max_agent_tasks)))
                    task_gain = max(get_task_gain(j, list(task_path), score_fun) for task_path in all_paths_iter)
                    c[i, j] = task_gain
                self.log(bids_to_string(c[i]), do_console=not silent)

            # # iterations bid rappresentano la funzione di costo del problema di ottimizzazione
            # NEGATO visto che nel nostro caso bisogna massimizzare iterations valori, mentre
            # Problem di Disropt trova iterations minimi
            bids_line = - np.reshape(c, (num_agents * num_tasks, 1))

            # print("Bids line:", bids_line)

            obj_function = bids_line @ x

            # iterations vincoli sono descritti nell'articolo di fonte
            sel_tasks = matlib.repmat(np.eye(num_tasks), 1, num_agents)
            sel_agents = np.array([ np.roll([1] * num_tasks + [0] * num_tasks * (num_agents - 1), y * num_tasks) for y in range(num_agents) ])

            constraints = [
                # Non assegnati più agent allo stesso task
                sel_tasks.T @ x <= np.ones((num_tasks, 1)), 
                # Non assegnati più di max_agent_tasks task allo stesso agent
                sel_agents.T @ x <= np.ones((num_agents, 1)) * max_agent_tasks, 
                # Non assegnati più o meno task del possibile in totale
                np.ones((num_agents * num_tasks, 1)) @ x == num_tasks,
                # X appartiene a 0 o 1
                x >= np.zeros((num_agents * num_tasks, 1)),
                x <= np.ones((num_agents * num_tasks, 1)),
            ]

            problem = Problem(obj_function, constraints)
            check_sol_line = problem.solve()
            check_sol = np.reshape(check_sol_line, (num_agents, num_tasks))

            self.log("X (Disropt):", do_console=not silent)
            self.log(sol_to_string(sol=check_sol), do_console=not silent)

            sol_line = np.reshape(sol, (num_agents * num_tasks, 1))

            cx = float(-bids_line.T @ sol_line)
            cx_check = float(-bids_line.T @ check_sol_line)
            pct_diff = (cx_check - cx) * 100 / cx_check

            self.log("c*x (CBBA): {}".format(cx), do_console=not silent)
            self.log("c*x (Disropt): {}".format(cx_check), do_console=not silent)
            self.log("Diff: {}%".format(pct_diff), do_console=not silent)

            return (sol, self.iterations, cx, cx_check, pct_diff)

        if return_iterations:
            return (sol, self.iterations)
        else:
            return sol

# Usato per il calcolo centralizzato di c_{ij} per il controllo
# con Disropt
def get_task_gain(task, task_path, score_function):
    if task in task_path:
        return 0
    else:
        start_score = score_function.eval(task_path)
        if len(task_path) > 0:
            return max(score_function.eval(insert_in_list(task_path, n, task)) for n in range(len(task_path))) - start_score
        else:
            return score_function.eval([task]) - start_score

def insert_in_list(lst, pos, val):
    return lst[:pos + 1] + [val] + lst[pos + 1:]

if __name__ == "__main__":
    args = parser.parse_args()

    num_agents = args.agents
    num_tasks = args.tasks
    max_agent_tasks = args.agent_tasks

    if max_agent_tasks < math.ceil(num_tasks / num_agents):
        print("L non sufficiente! Con questi N e t deve essere almeno {}".format(math.ceil(num_tasks / num_agents)))
        exit(-1)

    verbose = args.verbose
    test_mode = args.test_mode
    runs = args.runs
    do_test_runs = args.test_runs
    log_file = args.log
    silent = False

    if test_mode == 'test' and (runs > 1 or do_test_runs):
        if do_test_runs:
            print("run;agent num;task num;L_t;c * x (disropt);c * x (own);diff %;time (ms)")
        else:
            print("run;c * x (disropt);c * x (own);diff %;time (ms)")
        silent = True

    results = []
    times = []

    silent = silent or test_mode == 'conflict'

    def do_run(run_id, N, t, L):
        agent_positions = None
        task_positions = None
        if args.load_pos:
            (agent_positions, task_positions) = load_positions()
            if not silent:
                print("Loaded positions")

        tester = TesterCBBA()
        pre_time = time.time()
        ret = tester.run(N, t, L, verbose, test_mode == 'test', run_id, agent_positions=agent_positions, task_positions=task_positions, silent=silent, log_file=log_file)
        tim = time.time() - pre_time
        times.append(tim)
        results.append(ret)

        if test_mode == 'conflict':
            if tester.has_conflicts(ret):
                print("FOUND CONFLICTS!")
                print("Run: {}".format(run_id))
                print("num_agents: {} num_tasks: {} L: {}".format(N, t, L))
                print("sol:\n{}".format(sol_to_string(sol=ret)))
                return True
        elif test_mode == 'test' and runs > 1 or do_test_runs:
            print("{};{};{};{};{}".format(run_id, round(ret[3], 2), round(ret[2], 2), round(ret[4], 2), math.floor(tim * 1000)), flush=True)

        return False

    if do_test_runs:
        for N in range(3, num_agents + 1):
            for t in range(3, num_tasks + 1):
                for L in range(math.ceil(t / N), max(max_agent_tasks, math.ceil(t / N)) + 1):
                    for run_num in range(runs):
                        do_run("{};{};{};{}".format(run_num, N, t, L), N, t, L)
    else:
        for run_num in range(runs):
            do_run(run_num, num_agents, num_tasks, max_agent_tasks)

    if test_mode == 'test' and (runs > 1 or do_test_runs):
        avg_cx = round(average(list(ret[2] for ret in results)), 2)
        avg_cx_check = round(average(list(ret[3] for ret in results)), 2)
        avg_diff = round(average(list(ret[4] for ret in results)), 2)
        avg_time = math.floor(average(times) * 1000)

        if do_test_runs:
            print("{0};{0};{0};{0};{1};{2};{3};{4}".format("avg", avg_cx_check, avg_cx, avg_diff,avg_time))
        else:
            print("{};{};{};{};{}".format("avg", avg_cx_check, avg_cx, avg_diff, avg_time))

    if test_mode == 'conflict':
        print("No conflicts found in {} runs".format(runs))

    # print("---------------------")
    # print("Average time taken:", average(times))
    # print("Average pct diff: {}%".format(round(average([result[4] for result in results]), 2)))
    # print("No. of exact results: {}/{}".format(len(list(filter(lambda result: result[3] == 0, results))), runs))
    # print("Pct of exact results: {}%".format(round(100 * len(list(filter(lambda result: result[3] == 0, results))) / runs, 2)))