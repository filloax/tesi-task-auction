import numpy as np
from disropt.agents import Agent
import sys

# Più valori restituiti nel caso di pareggio
# opzionalmente da gestire con criterio a parte, ma è
# comunque abbastanza raro
def find_max_index(arr):
    max_val = max(arr)
    return [idx for idx in range(len(arr)) if arr[idx] == max_val]

class AuctionAlgorithm:
    def __init__(self, id, bids, agent: Agent, tasks, agent_ids, verbose = False):
        self.id = id
        self.bids = bids
        self.agent = agent
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
            valid_tasks = list(filter(lambda task: self.bids[task] >= self.max_bids[self.id][task], self.tasks))

            # print(self.id, self, valid_tasks)
            if len(valid_tasks) > 0:
                # Trova il punto di massimo, equivalente a argmax(h * c)

                max_valid_bid = max(self.bids[task] for task in valid_tasks)
                self.selected_task = list(filter(lambda task: self.bids[task] == max_valid_bid, valid_tasks))[0]
                
                if self.verbose:
                    print(self.id, "Selected", self.selected_task)
                self.assigned_tasks[ self.selected_task ] = 1
                self.max_bids[self.id][self.selected_task] = self.bids[self.selected_task]            

    def converge_phase(self):
        # Aggiorna la propria lista dei bid massimi in base a quelli ricevuti dagli altri
        # agenti, in modo tale da avere l'effettivo bid globalmente massimo per ogni task
        self.max_bids[self.id] = [ max( self.max_bids[agent_id][task] for agent_id in self.agent_ids ) for task in self.tasks ]
        # print(self.id, max_bids[self.id], max_bid_agents)

        # Se il proprio bid per il task selezionato non è quello più alto,
        # allora lascia il task selezionato all'agente che ha posto il bid
        # più alto, ne verrà cercato un'altro all'iterazione successiva
        max_bid_for_selected_task = max(self.max_bids[agent_id][self.selected_task] for agent_id in self.agent_ids)
        if max_bid_for_selected_task > self.bids[self.selected_task]:
            if self.verbose:
                print(self.id, "Higher bid exists as {}, removing...".format(max_bid_for_selected_task) )
            self.assigned_tasks[self.selected_task] = 0
        #lprint("- - - - - - - -")

    def check_done(self):
        # Controlla terminazione: se la lista max_bids è rimasta
        # invariata per un certo numero di iterazioni (vedi check_done()), e nessun elemento è 0

        # Se la lista dei bid è rimasta invariata
        if not (self.max_bids[self.id] == self.prev_max_bids).all():
            self.prev_max_bids = self.max_bids[self.id].copy()
            self.max_bids_equal_cnt = 0
            if self.verbose and self.verbose >= 2:
                print(self.id, "Max bids table changed:", self.max_bids[self.id])
        else:
            self.max_bids_equal_cnt += 1

        # Numero di iterazioni senza alterazioni nello stato rilevante per considerare l'operazione finita
        num_stable_runs = 2 * len(self.agent_ids) + 1
        return np.all((self.max_bids[self.id] != 0)) and np.sum(self.assigned_tasks) == 1 and self.max_bids_equal_cnt >= num_stable_runs

    def run_iter(self):
        self.auction_phase()

        # if self.max_bids_equal_cnt == 0: #se è cambiato dall'ultima iterazione, per evitare invii inutili
        #     #lprint("Sending: {} to {}".format(self.max_bids[self.id], neighbors))
        #     agent.neighbors_send(self.max_bids[self.id])
        data = self.agent.neighbors_exchange(self.max_bids[self.id])

        for other_id in filter(lambda id: id != self.id, data):
            self.max_bids[other_id] = data[other_id]

        self.converge_phase()

        #Se tutti elementi sono != 0 e ha un task assegnato, e la lista è rimasta invariata per 2N+1 iterazioni
        if self.check_done():
            self.agent.neighbors_send(self.max_bids[self.id]) # Evitare hang di altri agenti ancora in attesa
            self.done = True

    def run(self, beforeEach = None):
        while not self.done:
            if beforeEach != None:
                beforeEach()
            self.run_iter()

    def __repr__(self):
        return str(self.__dict__)

