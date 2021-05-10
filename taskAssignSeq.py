import random
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

class Agent:
    def __init__(self, id):
        self.id = id
        self.assigned_tasks = np.zeros(len(tasks))
        self.bids = np.random.random(len(tasks))
        # ogni agente prova a mantenere aggiornato ciò che sa dei dati circostanti di bid massimi
        # quindi mantiene l'intera matrice (una riga per agente) da aggiornare all'evenienza
        self.max_bids = np.zeros((len(agent_ids), len(tasks)))
        self.bids = np.random.random(len(tasks))
        self.selected_task = -1

        self.prev_max_bids = [-1, -1, -1, -1, -1]
        self.max_bids_equal_cnt = 0

    def auction_phase(self):
        # Sommatoria della riga corrispondente all'agente i
        if (sum(self.assigned_tasks) == 0):
            valid_tasks = [1 if self.bids[j] >= self.max_bids[self.id][j] else 0 for j in tasks]
            # print(self.id, self, valid_tasks)
            if sum(valid_tasks) > 0:
                self.selected_task = find_max_index(valid_tasks[j] * self.bids[j] for j in tasks)[0] # Trova il punto di massimo, equivalente a argmax(h * c)
                # print(self.id, "Selected", self.selected_task)
                self.assigned_tasks[ self.selected_task ] = 1
                self.max_bids[self.id][self.selected_task] = self.bids[self.selected_task]            

    def converge_phase(self):
        linked_agents = list(filter(lambda other_id: links[self.id][other_id] == 1, agent_ids)) # filtra precedentemente, più leggibile della sommatoria * valore booleano
        # print(self.id, "Linked:", linked_agents)

        # Calcolo del punto di minimo fatto prima della convergenza dei valori massimi in max_bids, altrimenti
        # il fatto che il valore attuale (in caso di altro bid più alto) venisse sostituito da quello maggiore
        # portava find_max_index a restituire anche quello come valore "di massimo" e quindi rimanevano entrambi con lo stesso bid
        max_bid_indices = find_max_index(self.max_bids[k][ self.selected_task ] for k in linked_agents)
        max_bid_agents = list(map(lambda link_id: linked_agents[link_id], max_bid_indices))
        self.max_bids[self.id] = [ max(self.max_bids[k][j] for k in linked_agents) for j in tasks ]
        # print(self.id, max_bids[self.id], max_bid_agents)
        if not (self.id in max_bid_agents):
            # print( "Higher bid exists as {}, removing...".format(max_bid_agents) )
            self.assigned_tasks[self.selected_task] = 0
        # print("- - - - - - - -")

    def __repr__(self):
        return str(self.__dict__)

# DA RICONTROLLARE: è effettivamente utile avere più punti di massimo restituiti in caso di valori uguali?
def find_max_index(arr):
    indices = None
    max = None
    for i, val in enumerate(arr):
        if max == None or val > max:
            indices = [i]
            max = val
        elif val == max:
            indices.append(i)
    return indices

def main():
    # Inizializza dati degli agenti
    agents = [Agent(id) for id in agent_ids]

    try:
        done = False
        while not done:
            order = list(range(N))
            random.shuffle(order)
            for id in order:
                agents[id].auction_phase()

                # ricevi dati da altri agenti (invio è passivo)
                for otherAgent in agent_ids:
                    if otherAgent != id and links[id][otherAgent] == 1:
                        agents[id].max_bids[otherAgent] = agents[otherAgent].max_bids[otherAgent]

                agents[id].converge_phase()

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

            time.sleep(1)
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