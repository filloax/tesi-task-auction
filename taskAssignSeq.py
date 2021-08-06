import random
from task_agent import TaskAgent
import numpy as np
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
    bids = [np.random.random(len(tasks)) for id in agent_ids]

    # Inizializza dati degli agenti
    agents = [TaskAgent(id, bids[id], None, tasks, agent_ids) for id in agent_ids]

    try:
        done = False
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
                done = done and agents[id].done

            print("Max bids [0]:")
            print(agents[0].max_bids)
            print("-------------------")
            print("Assigned tasks:")
            print(np.array([agent.assigned_tasks for agent in agents]))
            print("-------------------")
            print("Bids:")
            print(np.array([agent.bids for agent in agents]))
            print("###################")

            # Controlla terminazione: se la lista max_bids è rimasta
            # invariata per 3 (arbitrario) iterazioni
            # Sarebbe fatta indipendentemente da ogni agent in un vero sistema 
            # distribuito, qua viene controllato su [0]
            agent = agents[0]

            # Confronto tra array numpy
            if not (agent.max_bids[agent.id] == agent.prev_max_bids).all():
                agent.prev_max_bids = agent.max_bids[agent.id].copy()
                agent.max_bids_equal_cnt = 0
            else:
                agent.max_bids_equal_cnt += 1
                if agent.max_bids_equal_cnt >= 3:
                    done = True

            time.sleep(0.2)
            # input()

    except KeyboardInterrupt:
        print("Keyboard interrupt, early end...")

    print("Final version:")
    print("Assigned tasks:")
    print(np.array([agent.assigned_tasks for agent in agents]))
    print("-------------------")
    print("Max bids [0]:")
    print(agents[0].max_bids)
    print("-------------------")
    print("Bids:")
    print(np.array([agent.bids for agent in agents]))
    print("###################")
    print("Done!")

if __name__ == "__main__":
    main()