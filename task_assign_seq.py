import random
from utils import bids_to_string, sol_to_string
from task_seq_tester import TaskTester
from disropt.functions import Variable
from disropt.problems import Problem
import numpy as np
import numpy.matlib as matlib
import sys, os, time
import argparse
from numpy.lib.function_base import average
from auction_algo import AuctionAlgorithm
from task_positions import generate_positions, linear_dist, write_positions

parser = argparse.ArgumentParser(description='Distributed task assignment algorithm (CBAA).')
parser.add_argument('-N', '--agents', help="Amount of agents and tasks", required=True, type=int)
parser.add_argument('-v', '--verbose', default=False, action='store_true')
parser.add_argument('--test-mode', default=False, action='store_true')
parser.add_argument('--test-runs', default=1, type=int)

class TesterCBAA(TaskTester):

    def __init__(self):
        super().__init__()

    def run(self, num_agents, verbose = False, test_mode = False, run = 0, agent_positions = None, task_positions = None, 
    silent = False, return_iterations = False, log_file=''):
        num_tasks = num_agents

        self.log_file = log_file

        agent_ids = list(range(num_agents))
        tasks = list(range(num_tasks)) # J rappresentato con range(num_agents), 0..4 invece di 1..5
        # Analogo a
        # [0, 1, 0, 0, 0],
        # [1, 0, 1, 0, 0],
        # [0, 1, 0, 1, 0],
        # [0, 0, 1, 0, 1],
        # [0, 0, 0, 1, 0]
        links = np.roll(np.pad(np.eye(num_agents), 1), 1, 0)[1:-1,1:-1] + np.roll(np.pad(np.eye(num_agents), 1), -1, 0)[1:-1,1:-1]
        links = links + np.eye(num_agents) # Commentare per mettere g_ii = 0


        if agent_positions is None or task_positions is None:
            (agent_positions, task_positions) = generate_positions(num_agents, num_tasks)
            write_positions(agent_positions, task_positions)

        bids = np.array([[linear_dist(agent_positions[agent], task_positions[task]) for task in tasks] for agent in agent_ids])

        # Inizializza dati degli agenti
        agents = [AuctionAlgorithm(id, bids[id], None, tasks, agent_ids, verbose) for id in agent_ids]
        agents = [AuctionAlgorithm(id, 
            bids = bids[id],
            agent = None, 
            tasks = tasks, 
            agent_ids = agent_ids, 
            verbose = verbose,
            log_file = log_file,
            ) for id in agent_ids]

        self.iterations = 0
        self.const_winning_bids = [0 for agent in agents]
        force_print = False

        try:
            self.done_agents = 0

            while self.done_agents < num_agents:
                self.iterations = self.iterations + 1

                order = list(range(num_agents))
                random.shuffle(order)

                for id in order:
                    agents[id].auction_phase()

                    # ricevi dati da altri agenti (invio è passivo)
                    for otherAgent in agent_ids:
                        if (self.iterations > 1 or order.index(otherAgent) < order.index(id)) and otherAgent != id and links[id][otherAgent] == 1:
                            agents[id].max_bids[otherAgent] = agents[otherAgent].max_bids[otherAgent]

                random.shuffle(order)
                for id in order:
                    agents[id].converge_phase()
                    
                    if agents[id].check_done():
                        self.done_agents += 1

                    self.const_winning_bids[id] = agents[id].max_bids_equal_cnt

                self.log("Iter", self.iterations, do_console=verbose)
                self.log("Max bids:", do_console=verbose)
                self.log(bids_to_string([agent.max_bids[agent.id] for agent in agents]), do_console=verbose)
                self.log("-------------------", do_console=verbose)
                self.log("Assigned tasks:", do_console=verbose)
                self.log(sol_to_string(agents), do_console=verbose)
                self.log("-------------------", do_console=verbose)
                self.log("Bids:", do_console=verbose)
                self.log(np.round(np.array([agent.bids for agent in agents]), 2), do_console=verbose)
                self.log("###################", do_console=verbose)

                # time.sleep(0s.05)
                # input()

        except KeyboardInterrupt or BrokenPipeError:
            print("CBAA: Keyboard interrupt, early end...")
            print("CBAA: Keyboard interrupt, early end...", file=sys.stderr)
            force_print = True
        finally:
            if not test_mode and (force_print or not silent):
                self.log("Final version after", self.iterations, "self.iterations:")
                self.log("Assigned tasks:")
                self.log(sol_to_string(agents))
                self.log("-------------------")
                self.log("Max bids:")
                self.log(bids_to_string([agent.max_bids[agent.id] for agent in agents]))
                self.log("-------------------")
                self.log("Bids:")
                self.log(bids_to_string([agent.bids for agent in agents]))
                self.log("###################")

        sol = np.array([agent.assigned_tasks for agent in agents])

        if not test_mode and (force_print or not silent):
            self.log("\nDouble-check with Disropt optimization:")
        
        x = Variable(num_agents * num_agents)

        # self.iterations bid rappresentano la funzione di costo del problema di ottimizzazione
        # NEGATO visto che nel nostro caso bisogna massimizzare self.iterations valori, mentre
        # Problem di Disropt trova self.iterations minimi
        bids_line = -np.reshape(bids, (num_agents * num_agents, 1))

        # self.log("Bids line:", bids_line)

        obj_function = bids_line @ x

        # self.iterations vincoli sono descritti nell'articolo di fonte
        sel_tasks = matlib.repmat(np.eye(num_agents), 1, num_agents)
        sel_agents = np.array([ np.roll([1] * num_agents + [0] * num_agents * (num_agents - 1), y * num_agents) for y in range(num_agents)])

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
        check_sol = np.reshape(check_sol_line, (num_agents, num_agents))
        if not test_mode and not silent:
            self.log("Check solution:\n", check_sol)
            self.log("Own solution:\n", sol)

            self.log("c * x (disropt):", -bids_line.T @ check_sol_line)
            self.log("c * x (own):", -bids_line.T @ np.reshape(sol, (num_agents * num_agents, 1)))

            self.log("")

            # self.log("Selected:", np.nonzero(sol_mat)[1])

            self.log("#############################\n")

        cx_dis = float(-bids_line.T @ check_sol_line)
        cx_own = float(-bids_line.T @ np.reshape(sol, (num_agents * num_agents, 1)))

        if test_mode:
            self.log("{},{},{},{},{}".format(run, cx_dis, cx_own, cx_dis - cx_own, round(100 * (cx_dis - cx_own) / cx_dis, 2)))

            return [run, cx_dis, cx_own, cx_dis - cx_own, round(100 * (cx_dis - cx_own) / cx_dis, 2)]
        elif return_iterations:
            return (sol, self.iterations)
        else:
            return sol


if __name__ == "__main__":
    args = parser.parse_args()
    num_agents = args.agents
    verbose = args.verbose
    test_mode = args.test_mode
    runs = args.test_runs

    if test_mode:
        print("run,c * x (disropt),c * x (own), diff, diff %")

    results = []
    times = []

    for run_num in range(runs):
        tester = TesterCBAA()
        pre_time = time.time()
        ret = tester.run(num_agents, verbose, test_mode, run_num)
        times.append(time.time() - pre_time)
        results.append(ret)

    if test_mode:
        print("---------------------")
        print("Average time taken:", average(times))
        print("Average pct diff: {}%".format(round(average([result[4] for result in results]), 2)))
        print("No. of exact results: {}/{}".format(len(list(filter(lambda result: result[3] == 0, results))), runs))
        print("Pct of exact results: {}%".format(round(100 * len(list(filter(lambda result: result[3] == 0, results))) / runs, 2)))