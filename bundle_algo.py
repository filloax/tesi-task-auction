import numpy as np
from disropt.agents import Agent
import sys

class BundleAlgorithm:
    def __init__(self, id, bids, agent: Agent, tasks, max_agent_tasks, agent_ids, verbose = False):
        self.id = id
        self.bids = bids
        self.agent = agent
        self.max_agent_tasks = max_agent_tasks
        self.tasks = tasks
        self.agent_ids = agent_ids
        self.verbose = verbose

        self.task_bundle = []
        self.task_path = []

        self.winning_bids = np.zeros((len(self.agent_ids), len(self.tasks)))
        self.winning_agents = -np.ones((len(self.agent_ids), len(self.tasks)))

        self.message_times = np.zeros((len(self.agent_ids), len(self.agent_ids)))
        self.changed_ids = []

    def construct_phase(self, iter_num="."):
        while len(self.task_bundle) <= self.max_agent_tasks:
            task_score_improvements = {
                task: self._get_task_score_improvement(task) for task in self.tasks 
                    if not (task in self.task_bundle and not self._ignores_task(task))
            }
            self.selected_task = find_max_index(task_score_improvements[task] for task in self.tasks 
                if bid_is_greater(task_score_improvements[task], self.winning_bids[self.id][task]) )[0]

            task_position = find_max_index(self._calc_path_score(insert_in_list(self.task_path, n, self.selected_task)) for n in range(len(self.task_path)))

            self.task_bundle.append(self.selected_task)
            self.task_path.insert(task_position + 1, self.selected_task)
            self.winning_bids[self.id][self.selected_task] = task_score_improvements[self.selected_task]
            self.winning_agents[self.id][self.selected_task] = self.id

    def conflict_resolve_phase(self, iter_num="."):
        for other_id in self.changed_ids:
            task = 0
            action_default = "leave"
            # Prima chiave tabella: valori dell'agente vincitore secondo l'altro agente che
            # ha inviate i dati;
            # Seconda chiave tabella: valori dell'agente vincitore secondo questo agente
            # Valori: stringa per valore action, o per condizionali lambda id -> stringa
            case_table = {
                other_id: {
                    self.id: lambda id, rec_id: "update" if self.winning_bids[id][task] > self.winning_bids[self.id][task] else action_default,
                    other_id: "update",
                    "default": lambda id, rec_id: "update" if self.message_times[other_id][id] > self.message_times[self.id][id] or 
                        self.winning_bids[other_id][task] > self.winning_bids[self.id][task] else action_default,
                    -1: "update",
                },
                self.id: {
                    self.id: "leave",
                    other_id: "reset",
                    "default": lambda id, rec_id: "reset" if self.message_times[other_id][id] > self.message_times[self.id][id] else action_default,
                    -1: "leave",
                },
                "default": {
                    self.id: lambda id, rec_id: "update" if self.message_times[other_id][id] > self.message_times[self.id][id] and 
                        self.winning_bids[other_id][task] > self.winning_bids[self.id][task] else action_default,
                    other_id: lambda id, rec_id: "update" if self.message_times[other_id][id] > self.message_times[self.id][id] else "reset",
                    "this_id": lambda id, rec_id: "update" if self.message_times[other_id][id] > self.message_times[self.id][id] else action_default,
                    "default": lambda id, rec_id:
                        "update" if self.message_times[other_id][id] > self.message_times[self.id][id] and 
                            (self.message_times[other_id][rec_id] > self.message_times[self.id][rec_id] or
                            self.winning_bids[other_id][task] > self.winning_bids[self.id][task]) else
                        ("reset" if self.message_times[other_id][rec_id] > self.message_times[self.id][rec_id] and
                            self.message_times[self.id][id] > self.message_times[other_id][id] else action_default),
                    -1: lambda id, rec_id: "update" if self.message_times[other_id][id] > self.message_times[self.id][id] else action_default,
                },
                -1: {
                    self.id: "leave",
                    other_id: "update",
                    "default": lambda id, rec_id: "update" if self.message_times[other_id][id] > self.message_times[self.id][id] else action_default,
                    -1: "leave",
                },
            }

            main_ids = [self.id, other_id, -1]

            sender_choice = self.winning_agents[other_id][task]
            if not (sender_choice in main_ids):
                sender_choice = "default"

            this_choice = self.winning_agents[self.id][task]
            if not (this_choice in main_ids):
                if this_choice == sender_choice:
                    this_choice = "this_id"
                else:
                    this_choice = "default"

            choice = case_table.get(sender_choice).get(this_choice)

            # Evaluate conditionals
            if not (type(choice) is str):
                choice = choice(sender_choice, this_choice)

            changed = False

            if choice == "update":
                self.winning_bids[self.id][task] = self.winning_bids[other_id][task]
                self.winning_agents[self.id][task] = self.winning_bids[other_id][task]
                changed = True
            elif choice == "reset":
                self.winning_bids[self.id][task] = 0
                self.winning_agents[self.id][task] = -1
                changed = True

            if changed and task in self.task_bundle:
                start_idx = min(n for n in range(len(self.task_bundle)) if self.winning_agents[self.id][self.task_bundle[n]] != self.id)

                for i in range(start_idx + 1, len(self.task_bundle)):
                    self.winning_bids[self.id][self.task_bundle[i]] = 0
                    self.winning_agents[self.id][self.task_bundle[i]] = -1

                for i in range(start_idx, len(self.task_bundle)):
                    self.task_bundle.pop()



    def check_done(self, iter_num="."):
        return False

    # Da completare
    # Funzione S_{ij}
    def _calc_path_score(self, path):
        if len(path) == 0:
            return 0
        return 0

    # Per calcolare c_{ij} nell'algoritmo
    def _get_task_score_improvement(self, task):
        if task in self.task_bundle:
            return 0
        else:
            start_score = self._calc_path_score(self.task_path)
            return max(self._calc_path_score(insert_in_list(self.task_path, n, task)) for n in range(len(self.task_path))) - start_score

    def _ignores_task(self, task):
        return self.bids[task] == 0

    def run_iter(self, iter_num = "."):
        #Se tutti elementi sono != 0 e ha un task assegnato, e la lista è rimasta invariata per 2N+1 iterazioni
        if self.check_done(iter_num):
            self.agent.neighbors_send(self.max_bids[self.id]) # Evitare hang di altri agenti ancora in attesa
            self.done = True

    def run(self, beforeEach = None, max_iterations = -1):
        iterations = 0
        while not self.done:
            if beforeEach != None:
                beforeEach()
            if self.verbose:
                print("{}: iteration {} started".format(self.id, iterations))
            self.run_iter(iterations)
            if self.verbose:
                print("{}: iteration {} done".format(self.id, iterations))
            iterations = iterations + 1
            if max_iterations > 0 and iterations >= max_iterations:
                print("{}: max iterations reached".format(self.id))
                self.agent.neighbors_send(self.max_bids[self.id]) # Evitare hang di altri agenti ancora in attesa
                self.done = True

        if self.verbose:
            print("{}: Done, selected task {}".format(self.id, self.selected_task))

    def get_result(self):
        return (self.selected_task, self.assigned_tasks)

    def __repr__(self):
        return str(self.__dict__)


## Utilità

# Operazione ⊕ nell'articolo
def insert_in_list(lst, pos, val):
    return lst[:pos + 1] + [val] + lst[pos + 1:]

# Più valori restituiti nel caso di pareggio
# opzionalmente da gestire con criterio a parte, ma è
# comunque abbastanza raro
# Equivalente di argmax
def find_max_index(lst):
    max_val = max(lst)
    return [idx for idx in range(len(lst)) if lst[idx] == max_val]

# Ignora bid nulli (valore default) a scopo bid negativi
def bid_is_greater(bid1, bid2):
    return bid2 == 0 or bid1 > bid2