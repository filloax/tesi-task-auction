import sys
from bundle_algo import BundleAlgorithm, TimeScoreFunction
from task_positions import gen_distance_calc_time_fun, generate_positions, write_positions
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
parser.add_argument('-v', '--verbose', default=False, action='store_true')
parser.add_argument('--test-mode', default=False, action='store_true')
parser.add_argument('--test-runs', default=1, type=int)

class TesterCBBA:
    def __init__(self):
        # Non impostati nel costruttore, pensato per poter essere usato più volte con
        # dati diversi; servono principalmente a ottenerli dall'esterno nel multithreading
        self.done_agents = 0
        self.iterations = 0
        self.const_winning_bids = []
        self.log_file = ''

    def get_done_status(self):
        return (self.done_agents, self.iterations, self.const_winning_bids)

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

                self.log("Iter", self.iterations, do_console=verbose)
                self.log("Winning bids:", do_console=verbose)
                self.log(np.round(np.array([agent.winning_bids[agent.id] for agent in agents]), 2), do_console=verbose)
                self.log("-------------------", do_console=verbose)
                self.log("Winning agents:", do_console=verbose)
                self.log(np.array([agent.winning_agents[agent.id] for agent in agents]), do_console=verbose)
                self.log("-------------------", do_console=verbose)
                self.log("Task paths:", do_console=verbose)
                self.log("\n".join(["{}: {}".format(agent.id, agent.task_path) for agent in agents]), do_console=verbose)
                self.log("-------------------", do_console=verbose)
                self.log("Assigned tasks:", do_console=verbose)
                self.log(str(np.array([agent.assigned_tasks for agent in agents])).replace("0.", "_ "), do_console=verbose)
                self.log("###################", do_console=verbose)

                # time.sleep(0.5)
                # input()

        except KeyboardInterrupt or BrokenPipeError:
            print("CBBA: Keyboard interrupt, early end...")
            print("CBBA: Keyboard interrupt, early end...", file=sys.stderr)
            force_print = True
        finally:
            if not test_mode and (force_print or not silent):
                self.log("\n\nFinal version after", self.iterations, "iterations:")
                self.log("###################")
                self.log("Starting from:")
                self.log("Agent positions: \n{}".format(agent_positions[id] for id in agent_ids))
                self.log("Task positions: \n{}".format(task_positions[id] for id in tasks))
                self.log("###################")
                self.log("Winning bids:")
                self.log(np.round(np.array([agent.winning_bids[agent.id] for agent in agents]), 2))
                self.log("-------------------")
                self.log("Winning agents:")
                self.log(np.array([agent.winning_agents[agent.id] for agent in agents]))
                self.log("-------------------")
                self.log("Task paths:")
                self.log("\n".join(["{}: {}".format(agent.id, agent.task_path) for agent in agents]))
                self.log("-------------------")
                self.log("Assigned tasks:")
                self.log(str(np.array([agent.assigned_tasks for agent in agents])).replace("0.", "_ "))
                self.log("###################")

        sol = np.array([agent.assigned_tasks for agent in agents])

        if test_mode:
            x = Variable(num_agents * num_tasks)

            #TODO
            c = np.ones((num_agents, num_tasks))

            # iterations bid rappresentano la funzione di costo del problema di ottimizzazione
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
                np.ones((num_agents * num_agents, 1)) @ x == num_agents,
                # X appartiene a 0 o 1
                x >= np.zeros((num_agents * num_agents, 1)),
                x <= np.ones((num_agents * num_agents, 1)),
            ]

            problem = Problem(obj_function, constraints)
            check_sol_line = problem.solve()

        if return_iterations:
            return (sol, self.iterations)
        else:
            return sol

    def log(self, *values, do_console=True):
        if do_console:
            print(*values)
        if self.log_file != '':
            with open(self.log_file, 'a') as f:
                print(*values, file=f)


if __name__ == "__main__":
    args = parser.parse_args()

    num_agents = args.agents
    num_tasks = args.tasks
    max_agent_tasks = args.agent_tasks
    verbose = args.verbose
    test_mode = args.test_mode
    runs = args.test_runs

    if test_mode:
        print("run,c * x (disropt),c * x (own), diff, diff %")

    results = []
    times = []

    for run_num in range(runs):
        tester = TesterCBBA()
        pre_time = time.time()
        ret = tester.run(num_agents, num_tasks, max_agent_tasks, verbose, test_mode, run_num)
        times.append(time.time() - pre_time)
        results.append(ret)

    # print("---------------------")
    # print("Average time taken:", average(times))
    # print("Average pct diff: {}%".format(round(average([result[4] for result in results]), 2)))
    # print("No. of exact results: {}/{}".format(len(list(filter(lambda result: result[3] == 0, results))), runs))
    # print("Pct of exact results: {}%".format(round(100 * len(list(filter(lambda result: result[3] == 0, results))) / runs, 2)))