import random
from auction_algo import AuctionAlgorithm
from disropt.functions import Variable
from disropt.problems import Problem
import numpy as np
import numpy.matlib as matlib
import sys, os, time

from numpy.lib.function_base import average, select

N = 5
runs = 1
verbose = False
test_only = True

if len(sys.argv) > 1:
    N = int(sys.argv[1])
    if len(sys.argv) > 2:
        runs = int(sys.argv[2])

agent_ids = list(range(N))
tasks = list(range(N)) # J rappresentato con range(N), 0..4 invece di 1..5
# Analogo a
# [0, 1, 0, 0, 0],
# [1, 0, 1, 0, 0],
# [0, 1, 0, 1, 0],
# [0, 0, 1, 0, 1],
# [0, 0, 0, 1, 0]
links = np.roll(np.pad(np.eye(N), 1), 1, 0)[1:-1,1:-1] + np.roll(np.pad(np.eye(N), 1), -1, 0)[1:-1,1:-1]

links = links + np.eye(N) # Commentare per mettere g_ii = 0

def run(run = 0):
    bids = [np.random.random(len(tasks)) * 40 for id in agent_ids]

    # Inizializza dati degli agenti
    agents = [AuctionAlgorithm(id, bids[id], None, tasks, agent_ids) for id in agent_ids]

    try:
        done = False
        i = 1
        while not done:
            order = list(range(N))
            random.shuffle(order)

            done = True
            for id in order:
                agents[id].auction_phase()

                # ricevi dati da altri agenti (invio è passivo)
                for otherAgent in agent_ids:
                    if otherAgent != id and links[id][otherAgent] == 1:
                        agents[id].max_bids[otherAgent] = agents[otherAgent].max_bids[otherAgent]

                agents[id].converge_phase()
                
                this_done = agents[id].check_done()

                done = done and this_done

            if verbose:
                print("Iter", i)
                print("Max bids [0]:")
                print(agents[0].max_bids)
                print("-------------------")
                print("Assigned tasks:")
                print(np.array([agent.assigned_tasks for agent in agents]))
                print("-------------------")
                print("Bids:")
                print(np.array([agent.bids for agent in agents]))
                print("###################")

            i = i + 1

            # time.sleep(0s.05)
            # input()

    except KeyboardInterrupt:
        print("Keyboard interrupt, early end...")

    sol = np.array([agent.assigned_tasks for agent in agents])

    if not test_only:
        print("Final version after", i, "iterations:")
        print("Assigned tasks:")
        print(np.array(sol))
        print("-------------------")
        print("Max bids [0]:")
        print(agents[0].max_bids)
        print("-------------------")
        print("Bids:")
        print(np.array([agent.bids for agent in agents]))
        print("###################")

        print("\nDouble-check with Disropt optimization:")
    x = Variable(N * N)

    # I bid rappresentano la funzione di costo del problema di ottimizzazione
    # NEGATO visto che nel nostro caso bisogna massimizzare i valori, mentre
    # Problem di Disropt trova i minimi
    bids_line = - np.reshape(bids, (N * N, 1))

    # print("Bids line:", bids_line)

    obj_function = bids_line @ x

    # I vincoli sono descritti nell'articolo di fonte
    sel_tasks = matlib.repmat(np.eye(N), 1, N)
    sel_agents = np.array([ np.roll([1] * N + [0] * N * (N - 1), y * N) for y in range(N)])

    constraints = [
        # Non assegnati più agent allo stesso task
        sel_tasks.T @ x <= np.ones((N, 1)), 
        # Non assegnati più task allo stesso agent
        sel_agents.T @ x <= np.ones((N, 1)), 
        # Non assegnati più o meno task del possibile in totale
        np.ones((N * N, 1)) @ x == N,
        # X appartiene a 0 o 1
        x >= np.zeros((N * N, 1)),
        x <= np.ones((N * N, 1)),
    ]

    problem = Problem(obj_function, constraints)
    check_sol_line = problem.solve()
    check_sol = np.reshape(check_sol_line, (N, N))
    if not test_only:
        print("Check solution:\n", check_sol)
        print("Own solution:\n", sol)

        print("c * x (disropt):", -bids_line.T @ check_sol_line)
        print("c * x (own):", -bids_line.T @ np.reshape(sol, (N * N, 1)))

        print("")

        # print("Selected:", np.nonzero(sol_mat)[1])

        print("#############################\n")

    if not test_only:
        print("Test results for {}:".format(run))
        print("run,c * x (disropt),c * x (own), diff, diff %")
    cx_dis = float(-bids_line.T @ check_sol_line)
    cx_own = float(-bids_line.T @ np.reshape(sol, (N * N, 1)))
    print("{},{},{},{},{}".format(run, cx_dis, cx_own, cx_dis - cx_own, round(100 * (cx_dis - cx_own) / cx_dis, 2)))

    return [run, cx_dis, cx_own, cx_dis - cx_own, round(100 * (cx_dis - cx_own) / cx_dis, 2)]

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

    print("---------------------")
    print("Average time taken:", average(times))
    print("Average pct diff: {}%".format(round(average([result[4] for result in results]), 2)))
    print("No. of exact results: {}/{}".format(len(list(filter(lambda result: result[3] == 0, results))), runs))
    print("Pct of exact results: {}%".format(round(100 * len(list(filter(lambda result: result[3] == 0, results))) / runs, 2)))