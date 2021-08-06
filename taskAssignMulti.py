import time, sys, math, random
from disropt import agents
import numpy as np
from mpi4py import MPI
from disropt.agents import Agent

import argparse

start_time = time.time()

parser = argparse.ArgumentParser(description='Distributed multi task assignment algorithm.')
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

max_agent_tasks = 3
num_tasks = num_agents + 9
tasks = range(num_tasks)

class MultiTaskAgent:
    def __init__(self, id):
        self.id = id

        np.random.seed(math.floor(local_rank + time.time()))
        self.bids = np.random.random(num_tasks)
        print("{} bids: {}".format(self.id, self.bids), file=sys.stderr)
        # ogni agente prova a mantenere aggiornato ciò che sa dei dati circostanti di bid massimi
        # quindi mantiene l'intera matrice (una riga per agente) da aggiornare all'evenienza
        self.max_bids = np.zeros((num_agents, num_tasks))
        self.prev_max_bids = np.array([-1 for _ in range(num_tasks)])
        self.max_bids_equal_cnt = 0

        self.max_agents = [-1 for x in tasks]
        self.bundle = []
        self.path = []
        self.score = 0

    def bundle_phase(self):
        while len(self.bundle) < max_agent_tasks:
            task_score = max()

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