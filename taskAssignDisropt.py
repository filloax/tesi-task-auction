import time, sys, math
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent

import argparse

start_time = time.time()

parser = argparse.ArgumentParser(description='Distributed task assignment algorithm.')
parser.add_argument('-v', '--verbose', required=False, nargs='?', const=1, type=int)
args = parser.parse_args()

comm = MPI.COMM_WORLD
num_agents = comm.Get_size()
local_rank = comm.Get_rank()

agent_ids = range(num_agents)

neighbors = []
if local_rank == 0:
    neighbors = [local_rank + 1]
elif local_rank == num_agents - 1:
    neighbors = [local_rank - 1]
else:
    neighbors = [local_rank - 1, local_rank + 1]

tasks = list(range(num_agents)) # J rappresentato con range(num_agents), 0..4 invece di 1..5

class TaskAgent:
    def __init__(self, id):
        self.id = id
        self.assigned_tasks = np.zeros(len(tasks))
        np.random.seed(math.floor(local_rank + time.time()))
        self.bids = np.random.random(len(tasks))
        if args.verbose:
            lprint("Bids for agent {}: {}".format(self.id, self.bids))
        print("{} bids: {}".format(self.id, self.bids), file=sys.stderr)
        # ogni agente prova a mantenere aggiornato ciò che sa dei dati circostanti di bid massimi
        # quindi mantiene l'intera matrice (una riga per agente) da aggiornare all'evenienza
        self.max_bids = np.zeros((num_agents, len(tasks)))
        self.selected_task = -1

        self.prev_max_bids = np.array([-1 for _ in range(len(tasks))])
        self.max_bids_equal_cnt = 0

    def auction_phase(self):
        # Sommatoria della riga corrispondente all'agente i
        if (sum(self.assigned_tasks) == 0):
            valid_tasks = [1 if self.bids[j] >= self.max_bids[self.id][j] else 0 for j in tasks]
            # print(self.id, self, valid_tasks)
            if sum(valid_tasks) > 0:
                self.selected_task = find_max_index(valid_tasks[j] * self.bids[j] for j in tasks)[0] # Trova il punto di massimo, equivalente a argmax(h * c)
                if args.verbose:
                    lprint("Selected", self.selected_task)
                self.assigned_tasks[ self.selected_task ] = 1
                self.max_bids[self.id][self.selected_task] = self.bids[self.selected_task]            

    def converge_phase(self):
        # A differenza della versione sequenziale, check fatti su tutti gli id disponibili visto che non c'è
        # bisogno di simulare collegamento o meno con altri agenti a livello di questa funzione

        # Calcolo del punto di minimo fatto prima della convergenza dei valori massimi in max_bids, altrimenti
        # il fatto che il valore attuale (in caso di altro bid più alto) venisse sostituito da quello maggiore
        # portava find_max_index a restituire anche quello come valore "di massimo" e quindi rimanevano entrambi con lo stesso bid
        max_bid_agents = find_max_index(self.max_bids[k][ self.selected_task ] for k in agent_ids)

        self.max_bids[self.id] = [ max(self.max_bids[k][j] for k in agent_ids) for j in tasks ]
        # print(self.id, max_bids[self.id], max_bid_agents)
        
        #lprint("New max bids:", self.max_bids[self.id])

        if not (self.id in max_bid_agents):
            if args.verbose:
                lprint( "Higher bid exists as {}, removing...".format(max_bid_agents) )
            self.assigned_tasks[self.selected_task] = 0
        #lprint("- - - - - - - -")

    def __repr__(self):
        return str(self.__dict__)

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

runs = 0

def lprint(*arg):
    print(local_rank, "#{}".format(runs), *arg)


## Run

if args.verbose:
    lprint("Neighbors:", neighbors)
agent = Agent(in_neighbors=neighbors,
            out_neighbors=neighbors)

task_agent = TaskAgent(agent.id)

done = False

while not done:
    runs += 1
    task_agent.auction_phase()

    # if task_agent.max_bids_equal_cnt == 0: #se è cambiato dall'ultima iterazione, per evitare invii inutili
    #     #lprint("Sending: {} to {}".format(task_agent.max_bids[task_agent.id], neighbors))
    #     agent.neighbors_send(task_agent.max_bids[task_agent.id])
    data = agent.neighbors_exchange(task_agent.max_bids[task_agent.id])

    for other_id in filter(lambda id: id != local_rank, data):
        task_agent.max_bids[other_id] = data[other_id]

    task_agent.converge_phase()

    # Controlla terminazione: se la lista max_bids è rimasta
    # invariata per 3 (arbitrario) iterazioni, e nessun elemento è 0

    if not (task_agent.max_bids[task_agent.id] == task_agent.prev_max_bids).all():
        task_agent.prev_max_bids = task_agent.max_bids[task_agent.id].copy()
        task_agent.max_bids_equal_cnt = 0
        if args.verbose and args.verbose >= 2:
            lprint("Max bids table changed:", task_agent.max_bids[task_agent.id])
    else:
        task_agent.max_bids_equal_cnt += 1


    #Se tutti elementi sono != 0 e ha un task assegnato, e la lista è rimasta invariata per 2N+1 iterazioni
    if np.all((task_agent.max_bids[task_agent.id] != 0)) and np.sum(task_agent.assigned_tasks) == 1 and task_agent.max_bids_equal_cnt >= 2 * num_agents + 1:
        agent.neighbors_send(task_agent.max_bids[task_agent.id]) # Evitare hang di altri agenti ancora in attesa
        done = True
    
    # Non più necessario: vale solo se rimangono asincroni
        # Non mettere nessun tipo di attesa aumentava in maniera sostanziale le iterazioni
        # (da una media di 20 iterazioni con 8 agenti, a una media di 500 molto variabile)
        # Questo probabilmente per maggiori difficoltà nella trasmissione/ricezione di dati
        # con un uso del canale più frequente.
        # time.sleep(0.1)

print("{}: Done in {:.2f}s, {} iterations".format(local_rank, time.time() - start_time, runs), file=sys.stderr)

if args.verbose:
    print("\n###################")
    lprint("Results in {} runs:".format(runs))
    lprint("Assigned tasks:")
    lprint(task_agent.assigned_tasks)
    lprint("-------------------")
    lprint("Max bids:")
    lprint(task_agent.max_bids[task_agent.id])
    lprint("-------------------")
    lprint("Bids:")
    lprint(task_agent.bids)
    lprint("Done!")
    print("###################\n")
else:
    print("{}:{}".format(task_agent.id, " ".join(map(lambda x: str(int(x)), task_agent.assigned_tasks))))
