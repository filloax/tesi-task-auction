import numpy as np
from disropt.agents import Agent
import sys

# Più valori restituiti nel caso di pareggio
# opzionalmente da gestire con criterio a parte, ma è
# comunque abbastanza raro
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

class TaskAgent:
    def __init__(self, id, bids, comm_agent: Agent, tasks, agent_ids, verbose = False):
        self.id = id
        self.bids = bids
        self.agent = comm_agent
        self.tasks = tasks
        self.agent_ids = agent_ids
        self.verbose = verbose

        print("{} bids: {}".format(self.id, self.bids), file=sys.stderr)

        self.done = False
        self.assigned_tasks = np.zeros(len(self.tasks))
        # ogni agente prova a mantenere aggiornato ciò che sa dei dati circostanti di bid massimi
        # quindi mantiene l'intera matrice (una riga per agente) da aggiornare all'evenienza
        self.max_bids = np.zeros((len(self.agent_ids), len(self.tasks)))
        self.selected_task = -1

        self.prev_max_bids = np.array([-1 for _ in range(len(self.tasks))])
        self.max_bids_equal_cnt = 0

    def auction_phase(self):
        # Sommatoria della riga corrispondente all'agente i
        if (sum(self.assigned_tasks) == 0):
            valid_tasks = [1 if self.bids[j] >= self.max_bids[self.id][j] else 0 for j in self.tasks]
            # print(self.id, self, valid_tasks)
            if sum(valid_tasks) > 0:
                self.selected_task = find_max_index(valid_tasks[j] * self.bids[j] for j in self.tasks)[0] # Trova il punto di massimo, equivalente a argmax(h * c)
                if self.verbose:
                    print(self.id, "Selected", self.selected_task)
                self.assigned_tasks[ self.selected_task ] = 1
                self.max_bids[self.id][self.selected_task] = self.bids[self.selected_task]            

    def converge_phase(self):
        # A differenza della versione sequenziale, check fatti su tutti gli id disponibili visto che non c'è
        # bisogno di simulare collegamento o meno con altri agenti a livello di questa funzione

        # Calcolo del punto di minimo fatto prima della convergenza dei valori massimi in max_bids, altrimenti
        # il fatto che il valore attuale (in caso di altro bid più alto) venisse sostituito da quello maggiore
        # portava find_max_index a restituire anche quello come valore "di massimo" e quindi rimanevano entrambi con lo stesso bid
        max_bid_agents = find_max_index(self.max_bids[k][ self.selected_task ] for k in self.agent_ids)

        self.max_bids[self.id] = [ max(self.max_bids[k][j] for k in self.agent_ids) for j in self.tasks ]
        # print(self.id, max_bids[self.id], max_bid_agents)
        
        #lprint("New max bids:", self.max_bids[self.id])

        if not (self.id in max_bid_agents):
            if self.verbose:
                print(self.id, "Higher bid exists as {}, removing...".format(max_bid_agents) )
            self.assigned_tasks[self.selected_task] = 0
        #lprint("- - - - - - - -")

    def run_iter(self):
        self.auction_phase()

        # if self.max_bids_equal_cnt == 0: #se è cambiato dall'ultima iterazione, per evitare invii inutili
        #     #lprint("Sending: {} to {}".format(self.max_bids[self.id], neighbors))
        #     agent.neighbors_send(self.max_bids[self.id])
        data = self.agent.neighbors_exchange(self.max_bids[self.id])

        for other_id in filter(lambda id: id != self.id, data):
            self.max_bids[other_id] = data[other_id]

        self.converge_phase()

        # Controlla terminazione: se la lista max_bids è rimasta
        # invariata per 3 (arbitrario) iterazioni, e nessun elemento è 0

        if not (self.max_bids[self.id] == self.prev_max_bids).all():
            self.prev_max_bids = self.max_bids[self.id].copy()
            self.max_bids_equal_cnt = 0
            if self.verbose and self.verbose >= 2:
                print(self.id, "Max bids table changed:", self.max_bids[self.id])
        else:
            self.max_bids_equal_cnt += 1


        #Se tutti elementi sono != 0 e ha un task assegnato, e la lista è rimasta invariata per 2N+1 iterazioni
        if np.all((self.max_bids[self.id] != 0)) and np.sum(self.assigned_tasks) == 1 and self.max_bids_equal_cnt >= 2 * len(self.agent_ids) + 1:
            self.agent.neighbors_send(self.max_bids[self.id]) # Evitare hang di altri agenti ancora in attesa
            self.done = True

    def run(self, beforeEach = None):
        while not self.done:
            if beforeEach != None:
                beforeEach()
            self.run_iter()

    def __repr__(self):
        return str(self.__dict__)

