from os import error
import random
import numpy as np
from disropt.agents import Agent
from numpy.lib.function_base import average
try:
    from utils import *
except ImportError:
    from .utils import *

class AuctionAlgorithm:
    def __init__(self, id, bids, agent: Agent, tasks, agent_ids, verbose = False, log_file = ''):
        if any(filter(lambda bid: bid < 0, bids)):
            raise ValueError("Tried to start CBAA with negative bids. They are not supported, if you need to find a minimum instead of a maximum you can usually use 1/x, 0.y^x or similar to convert the bids.")

        self.id = id
        self.bids = bids
        self.agent = agent
        self.tasks = tasks
        self.agent_ids = agent_ids
        self.verbose = verbose
        self.log_file = log_file

        # Spareggio: aumenta i bid di un fattore casuale e relativamente insignificante per evitare blocchi
        # in caso di pareggio tra agenti
        bid_average = average(self.bids)
        for task in self.tasks:
            rand_increase = random.random() * bid_average * 0.001
            self.bids[task] = self.bids[task] + rand_increase

        self.log_verbose(0, "Init with data: \n\tid: {}\n\tbids: {}\n\ttasks: {}\n\tagents: {}\n\tneighbors: {}"
                .format(self.id, self.bids, self.tasks, self.agent_ids, self.agent.in_neighbors if self.agent != None else []))

        self.done = False
        self.assigned_tasks = np.zeros(len(self.tasks))
        # ogni agente prova a mantenere aggiornato ciò che sa dei dati circostanti di bid massimi
        # quindi mantiene l'intera matrice (una riga per agente) da aggiornare all'evenienza

        self.max_bids = np.zeros((len(self.agent_ids), len(self.tasks)))
        self.selected_task = -1

        self.prev_max_bids = np.array([-1 for _ in range(len(self.tasks))])
        self.max_bids_equal_cnt = 0

    def auction_phase(self, iter_num="."):
        # Sommatoria della riga corrispondente all'agente i
        if (sum(self.assigned_tasks) == 0):
            # Task ottenibili solo quelli dove il bid è maggiore del massimo pre-esistente,
            # ignorando bid non interessati
            valid_tasks = list(
                filter(lambda task: self._bid_is_greater(self.bids[task], self.max_bids[self.id][task]) and not self._ignores_task(task), 
                self.tasks))

            # self.log_verbose(iter_num, self.id, self, valid_tasks)
            if len(valid_tasks) > 0:
                # Trova il punto di massimo, equivalente a argmax(h * c)

                max_valid_bid = max(self.bids[task] for task in valid_tasks)
                self.selected_task = list(filter(lambda task: self.bids[task] == max_valid_bid, valid_tasks))[0]
                
                self.log_verbose(iter_num, "Selected {} with bid {}".format(self.selected_task, self.bids[self.selected_task]))
                self.assigned_tasks[ self.selected_task ] = 1
                self.max_bids[self.id][self.selected_task] = self.bids[self.selected_task]            

    # TODO: recuperare in parte il sistema di rilevazione origine max bid
    # per attivare lo spareggio SE e SOLO SE è uguale al max bid ed esso viene da altri

    def converge_phase(self, iter_num="."):
        # Aggiorna la propria lista dei bid massimi in base a quelli ricevuti dagli altri
        # agenti, in modo tale da avere l'effettivo bid globalmente massimo per ogni task
        # La lista viene temporaneamente salvata in variabile a parte per permettere di capire da chi provengano,
        # o meglio se siano cambiati in questa iterazione
        new_max_bids = np.array([max(self.max_bids[agent_id][task] for agent_id in self.agent_ids) for task in self.tasks])

        # Se il proprio bid per il task selezionato non è quello più alto,
        # allora lascia il task selezionato all'agente che ha posto il bid
        # più alto, ne verrà cercato un'altro all'iterazione successiva
        if self.selected_task >= 0:
            max_bid_for_selected_task = new_max_bids[self.selected_task]

            if self._bid_is_greater(max_bid_for_selected_task, self.bids[self.selected_task]):
                self.log_verbose(iter_num, "Higher bid exists as {}, removing...".format(max_bid_for_selected_task) )
                self.assigned_tasks[self.selected_task] = 0
                self.selected_task = -1
            # Disabilitato in quanto mai attivabile, visto che non è possibile (senza tracciare i vincitori dei bid tramite messaggi
            # come in CBBA) capire se il bid uguale che si riceve avvia avuto origine da sè o da un'altro agente
            # Sostituito con aumento casuale del bid all'inizio
            # elif self._bid_is_equal(max_bid_for_selected_task, self.bids[self.selected_task]) and new_max_bids[self.selected_task] != self.max_bids[self.id][self.selected_task]:
            #     #In caso di pareggio, aumenta casualmente il bid per spareggiare, in una quantità trascurabile relativamente ai bid
            #     bid_average = average(self.max_bids)
            #     max_increase = bid_average * 0.001
            #     rand_increase = random.random() * max_increase
            #     self.bids[self.selected_task] = self.bids[self.selected_task] + rand_increase
            #     self.log_verbose(iter_num, "Equal bid exists for {}, increased own by {} ({} -> {})...".format(
            #         self.selected_task, rand_increase, self.bids[self.selected_task] - rand_increase, self.bids[self.selected_task]) )
            #     self.max_bids[self.id][self.selected_task] = self.bids[self.selected_task]

        self.max_bids[self.id] = new_max_bids


    def check_done(self, iter_num="."):
        # Controlla terminazione: se la lista max_bids è rimasta
        # invariata per un certo numero di iterazioni (vedi check_done()), e nessun elemento è 0

        # Se la lista dei bid è rimasta invariata
        if not (self.max_bids[self.id] == self.prev_max_bids).all():
            self.prev_max_bids = self.max_bids[self.id].copy()
            self.max_bids_equal_cnt = 0
            self.log_verbose(iter_num, "Max bids table changed: {}".format(self.max_bids[self.id]))
        else:
            self.max_bids_equal_cnt += 1

        # Numero di iterazioni senza alterazioni nello stato rilevante per considerare l'operazione finita
        num_stable_runs = 2 * len(self.agent_ids) + 1
        # Se tutti i bid massimi sono stati ricevuti, ignorando quelli per i task 
        # che questo agente sta ignorando
        all_max_bids_set = all(self.max_bids[self.id][task] != 0 for task in self.tasks if not self._ignores_task(task))
        return all_max_bids_set and np.sum(self.assigned_tasks) == 1 and self.selected_task >= 0 and self.max_bids_equal_cnt >= num_stable_runs

    def run_iter(self, iter_num = "."):
        self.auction_phase(iter_num)

        # if self.max_bids_equal_cnt == 0: #se è cambiato dall'ultima iterazione, per evitare invii inutili
        #     #lprint("Sending: {} to {}".format(self.max_bids[self.id], neighbors))
        #     agent.neighbors_send(self.max_bids[self.id])
        self.log_verbose(iter_num, "pre exchange")
        data = self.agent.neighbors_exchange(self.max_bids[self.id])
        self.log_verbose(iter_num, "post exchange")

        for other_id in filter(lambda id: id != self.id, data):
            self.max_bids[other_id] = data[other_id]

        self.converge_phase(iter_num)

        #Se tutti elementi sono != 0 e ha un task assegnato, e la lista è rimasta invariata per 2N+1 iterazioni
        if self.check_done(iter_num):
            self.agent.neighbors_send(self.max_bids[self.id]) # Evitare hang di altri agenti ancora in attesa
            self.done = True

    def run(self, beforeEach = None, max_iterations = -1):
        iterations = 0
        while not self.done:
            if beforeEach != None:
                beforeEach()
            self.log_verbose(iterations, "iteration {} started".format(self.id, iterations))

            self.run_iter(iterations)

            self.log_verbose(iterations, "iteration {} done".format(self.id, iterations))
            iterations = iterations + 1
            if max_iterations > 0 and iterations >= max_iterations:
                print("{}: max iterations reached".format(self.id))
                self.agent.neighbors_send(self.max_bids[self.id]) # Evitare hang di altri agenti ancora in attesa
                self.done = True

        self.log_verbose(iterations, "Done, selected task {}".format(self.id, self.selected_task))

    def get_result(self):
        return (self.selected_task, self.assigned_tasks)
        
    def log_verbose(self, iter_num = None, *values):
        string = ""
        if iter_num is None:
            string = ("{:" + str(len(self.agent_ids)) + "d}:").format(self.id)
        else:
            string = ("{:" + str(len(self.agent_ids)) + "d} | {}:").format(self.id, iter_num)

        if self.verbose:
            print(string, *values)
        if self.log_file != '':
            with open(self.log_file, 'a') as f:
                print(string, *values, file=f)

    # Nessuno spareggio qua, viene modificato direttamente il bid
    def _bid_is_greater(self, bid1, bid2):
        # if bid1 == bid2 and id1 >= 0 and id2 >= 0:
        #     return id1 > id2
        return bid1 > bid2

    def _bid_is_equal(self, bid1, bid2):
        return bid1 == bid2

    def _ignores_task(self, task):
        return self.bids[task] == 0

    def __repr__(self):
        return str(self.__dict__)