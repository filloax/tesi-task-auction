import random
import numpy as np
import time

from numpy.lib.function_base import select

N = 5

agents = list(range(N))
tasks = list(range(N)) # J rappresentato con range(N), 0..4 invece di 1..5
assigned_tasks = np.zeros((N, N))
max_bids = np.zeros((N, N))
bids = np.random.random((N, N)) # Inizializza costi globali con valori causali [0-1]
selected_tasks = [-1 for x in range(N)]
links = [
    [0, 1, 0, 0, 0],
    [1, 0, 1, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 1, 0, 1],
    [0, 0, 0, 1, 0]
]
links = links + np.eye(N) # Commentare per mettere g_ii = 0
# links = np.ones((N, N))

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

def auction_phase(agent_id):
    # Sommatoria della riga corrispondente all'agente i
    if (sum(assigned_tasks[agent_id]) == 0):
        valid_tasks = [1 if bids[agent_id][j] >= max_bids[agent_id][j] else 0 for j in tasks]
        # print(agent_id, valid_tasks)
        if sum(valid_tasks) > 0:
            selected_tasks[agent_id] = find_max_index(valid_tasks[j] * bids[agent_id][j] for j in tasks)[0] # Trova il punto di massimo, equivalente a argmax(h * c)
            # print(agent_id, "Selected", selected_tasks[agent_id])
            assigned_tasks[agent_id][ selected_tasks[agent_id] ] = 1
            max_bids[agent_id][selected_tasks[agent_id]] = bids[agent_id][selected_tasks[agent_id]]            

def converge_phase(agent_id):
    linked_agents = list(filter(lambda other_id: links[agent_id][other_id] == 1, agents)) # filtra precedentemente, più leggibile della sommatoria * valore booleano
    # print(agent_id, "Linked:", linked_agents)

    # Calcolo del punto di minimo fatto prima della convergenza dei valori massimi in max_bids, altrimenti
    # il fatto che il valore attuale (in caso di altro bid più alto) venisse sostituito da quello maggiore
    # portava find_max_index a restituire anche quello come valore "di massimo" e quindi rimanevano entrambi con lo stesso bid
    max_bid_indices = find_max_index(max_bids[k][ selected_tasks[agent_id] ] for k in linked_agents)
    max_bid_agents = list(map(lambda link_id: linked_agents[link_id], max_bid_indices))
    max_bids[agent_id] = [ max(max_bids[k][j] for k in linked_agents) for j in tasks ]
    # print(agent_id, max_bids[agent_id], max_bid_agents)
    if not (agent_id in max_bid_agents):
        # print( "Higher bid exists as {}, removing...".format(max_bid_agents) )
        assigned_tasks[agent_id][selected_tasks[agent_id]] = 0
    # print("- - - - - - - -")

try:
    while True:
        order = list(range(N))
        random.shuffle(order)
        for agent_id in order:
            auction_phase(agent_id)
            converge_phase(agent_id)
        print("Max bids:")
        print(max_bids)
        print("-------------------")
        print("Assigned tasks:")
        print(assigned_tasks)
        print("-------------------")
        print("Bids:")
        print(bids)
        print("###################")
        time.sleep(1)

except KeyboardInterrupt:
    print("Final version:")
    print("Assigned tasks:")
    print(assigned_tasks)
    print("-------------------")
    print("Max bids:")
    print(max_bids)
    print("-------------------")
    print("Bids:")
    print(bids)
    print("###################")
    print("Done!")
