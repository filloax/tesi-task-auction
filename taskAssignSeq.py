import random
from task_agent import TaskAgent
from disropt.functions import Variable
from disropt.problems import Problem
import numpy as np
import numpy.matlib as matlib
import time

from numpy.lib.function_base import select

N = 5

agent_ids = list(range(N))
tasks = list(range(N)) # J rappresentato con range(N), 0..4 invece di 1..5
links = [
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
]
links = links + np.eye(N) # Commentare per mettere g_ii = 0

def main():
    bids = [np.random.random(len(tasks)) * 40 for id in agent_ids]

    # Inizializza dati degli agenti
    agents = [TaskAgent(id, bids[id], None, tasks, agent_ids) for id in agent_ids]

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

            # time.sleep(0.05)
            # input()

    except KeyboardInterrupt:
        print("Keyboard interrupt, early end...")

    print("Final version after", i, "runs:")
    print("Assigned tasks:")
    sol = np.array([agent.assigned_tasks for agent in agents])
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
    # print("Check solution:\n", check_sol_line)
    print("Check solution:\n", check_sol)
    print("Own solution:\n", sol)

    print("c * x (disropt):", -bids_line.T @ check_sol_line)
    print("c * x (own):", -bids_line.T @ np.reshape(sol, (N * N, 1)))

    # print("Selected:", np.nonzero(sol_mat)[1])

    print("#############################\n\nDone!")

if __name__ == "__main__":
    main()