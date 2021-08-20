import numpy as np
from disropt.agents import Agent
import time
try:
    from utils import *
except ImportError:
    from .utils import *

class ScoreFunction:
    def __init__(self, agent_id: int):
        self.agent_id = agent_id

    def eval(self, path: list) -> float:
        if len(path) == 0:
            return 0
        else:
            return self.do_eval(path)

    def do_eval(self, path: list) -> float:
        pass

class BundleAlgorithm:
    def __init__(self, id, agent: Agent, score_function: ScoreFunction, tasks, max_agent_tasks, agent_ids, verbose = False, log_file = ''):
        self.id = id
        self.agent = agent
        # Funzione S(path, agent)
        self.score_function = score_function
        self.max_agent_tasks = max_agent_tasks
        self.tasks = tasks
        self.agent_ids = agent_ids
        self.verbose = verbose
        self.log_file = log_file

        self.task_bundle = []
        self.task_path = []
        self.assigned_tasks = np.zeros(len(self.tasks))

        self.bids = np.zeros(len(self.tasks)) # Usato solo per debug, impostati in fase di costruzione
        self.winning_bids = np.zeros((len(self.agent_ids), len(self.tasks)))
        self.winning_agents = -np.ones((len(self.agent_ids), len(self.tasks)))
        self.prev_win_bids = np.zeros(len(self.tasks))
        self.win_bids_equal_cnt = 0

        self.message_times = np.zeros((len(self.agent_ids), len(self.agent_ids)))
        self.changed_ids = []

        self.done = False
        # Tieni traccia di quali vicini hanno concluso operazioni e non comunicheranno più,
        # per smettere di rimanere in ascolto nel caso in cui non ce ne siano più
        # IPOTESI: vicini non cambiano nel tempo
        # CHIEDERE
        self.done_neighbors = [] 

    def construct_phase(self, iter_num="."):
        self.log_verbose(iter_num, "pre construct bundle: {} path: {}".format(self.task_bundle, self.task_path))

        while len(self.task_bundle) < self.max_agent_tasks:
            # Questo è c_{ij}, la funzione di costo (o meglio, guadagno)
            # consiste in quanto l'aggiunta di un dato task sia in grado
            # di aumentare il punteggio del percorso
            self.bids = np.array([self.get_task_gain(task) if not self._ignores_task(task) else 0 for task in self.tasks])

            # Uso di dict per non considerare task non validi o già aggiunti
            task_score_improvements = { task: self.bids[task] for task in self.tasks 
                    if task not in self.task_bundle and not self._ignores_task(task) }

            if any(task_score_improvements[task] < 0 for task in task_score_improvements):
                raise ValueError("Got negative score improvement in CBBA. Negative values are not supported, if you need to find a minimum instead of a maximum you can usually use 1/x, 0.y^x or similar somewhere in the score function.")

            selected_task = -1

            self.log_verbose(iter_num, "task_score_improvements: {}".format(bids_to_string([bid if task in task_score_improvements else 0 for (task, bid) in enumerate(self.bids)])))

            if any(task in task_score_improvements and self._bid_is_greater(task_score_improvements[task], self.winning_bids[self.id][task], self.id, self.winning_agents[self.id][task]) for task in self.tasks):
                max_score_improvement = max(task_score_improvements[task] for task in task_score_improvements if self._bid_is_greater(task_score_improvements[task], self.winning_bids[self.id][task], self.id, self.winning_agents[self.id][task]))
                
                selected_task = next(task for task in self.tasks if 
                    task in task_score_improvements 
                    and task_score_improvements[task] == max_score_improvement 
                    # Controlla di nuovo per evitare scelta sbagliata nel caso ci siano task con lo stesso valore ma alcuni già presi da altri
                    and self._bid_is_greater(task_score_improvements[task], self.winning_bids[self.id][task], self.id, self.winning_agents[self.id][task]))

            if selected_task < 0:
                self.log_verbose(iter_num, "Out of tasks".format(self.id, iter_num))
                break

            self.log_verbose(iter_num, "Selected task {}\tbid is {} > {}".format(selected_task, round(task_score_improvements[selected_task], 3), round(self.winning_bids[self.id][selected_task], 3)))

            task_position = 0
            if len(self.task_path) > 0:
                task_position = find_max_index(self.score_function.eval(insert_in_list(self.task_path, n, selected_task)) for n in range(len(self.task_path)))[0]

            self.assigned_tasks[selected_task] = 1
            self.task_bundle.append(selected_task)
            self.task_path.insert(task_position + 1, selected_task)
            self.winning_bids[self.id][selected_task] = task_score_improvements[selected_task]
            self.winning_agents[self.id][selected_task] = self.id
        self.log_verbose(iter_num, "Phase end path: {}".format(self.task_path))
        
    # Per calcolare c_{ij} nell'algoritmo
    def get_task_gain(self, task):
        if task in self.task_bundle:
            return 0
        else:
            start_score = self.score_function.eval(self.task_path)
            if len(self.task_path) > 0:
                return max(self.score_function.eval(insert_in_list(self.task_path, n, task)) for n in range(len(self.task_path))) - start_score
            else:
                return self.score_function.eval([task]) - start_score

    def conflict_resolve_phase(self, iter_num="."):
        self.log_verbose(iter_num, "Pre conf res state: agents: {} bids: {} bundle: {}".format(self.winning_agents[self.id], self.winning_bids[self.id], self.task_bundle))

        self.log_verbose(iter_num, "To update: ")
        for other_id in self.changed_ids:
            self.log_verbose(iter_num, "\t{}: agents: {} bids: {}".format(other_id, self.winning_agents[other_id], self.winning_bids[other_id]))
        self.log_verbose(iter_num, "Message times:")
        for id in self.changed_ids + [self.id]:
            self.log_verbose(iter_num, "\t{}: {}".format(id, self.message_times[id]))

        for other_id in self.changed_ids:
            for task in self.tasks:
                self.update_choice(other_id, task, iter_num)

        # Aggiorna s_{ij} con valori per gli agenti non direttamente raggiungibili
        if len(self.changed_ids) > 0:
            for non_changed_id in filter(lambda id: id not in self.changed_ids and id != self.id, self.agent_ids):
                self.message_times[self.id][non_changed_id] = max(self.message_times[m][non_changed_id] for m in self.changed_ids + [self.id])

        self.log_verbose(iter_num, "Post conf res state: agents: {} bids: {} bundle: {}".format(self.winning_agents[self.id], self.winning_bids[self.id], self.task_bundle))
        self.log_verbose(iter_num, "\tMessage times: {}".format(self.message_times[self.id]))

    def update_choice(self, other_id, task, iter_num):
        action_default = "leave"
        # Prima chiave tabella: valori dell'agente vincitore secondo l'altro agente che
        # ha inviate i dati;
        # Seconda chiave tabella: valori dell'agente vincitore secondo questo agente
        # Valori: stringa per valore action, o per condizionali lambda id -> stringa
        # Tabella adattata dall'articolo
        case_table = {
            other_id: {
                self.id: lambda send_winid, rec_winid: "update" if self._bid_is_greater(self.winning_bids[other_id][task], self.winning_bids[self.id][task], other_id, self.id) else action_default,
                other_id: "update",
                "default": lambda send_winid, rec_winid: "update" if self.message_times[other_id][rec_winid] > self.message_times[self.id][rec_winid] or 
                    self._bid_is_greater(self.winning_bids[other_id][task], self.winning_bids[self.id][task], other_id, self.id) else action_default,
                -1: "update",
            },
            self.id: {
                self.id: "leave",
                other_id: "reset",
                "default": lambda send_winid, rec_winid: "reset" if self.message_times[other_id][rec_winid] > self.message_times[self.id][rec_winid] else action_default,
                -1: "leave",
            },
            "default": {
                self.id: lambda send_winid, rec_winid: "update" if self.message_times[other_id][send_winid] > self.message_times[self.id][send_winid] and 
                    self._bid_is_greater(self.winning_bids[other_id][task], self.winning_bids[self.id][task], other_id, self.id) else action_default,
                other_id: lambda send_winid, rec_winid: "update" if self.message_times[other_id][send_winid] > self.message_times[self.id][send_winid] else "reset",
                "this_id": lambda send_winid, rec_winid: "update" if self.message_times[other_id][send_winid] > self.message_times[self.id][send_winid] else action_default,
                "default": lambda send_winid, rec_winid:
                    "update" if self.message_times[other_id][send_winid] > self.message_times[self.id][send_winid] and 
                        (self.message_times[other_id][rec_winid] > self.message_times[self.id][rec_winid] or
                        self._bid_is_greater(self.winning_bids[other_id][task], self.winning_bids[self.id][task], other_id, self.id)) else
                    ("reset" if self.message_times[other_id][rec_winid] > self.message_times[self.id][rec_winid] and
                        self.message_times[self.id][send_winid] > self.message_times[other_id][send_winid] else action_default),
                -1: lambda send_winid, rec_winid: "update" if self.message_times[other_id][send_winid] > self.message_times[self.id][send_winid] else action_default,
            },
            -1: {
                self.id: "leave",
                other_id: "update",
                "default": lambda send_winid, rec_winid: "update" if self.message_times[other_id][rec_winid] > self.message_times[self.id][rec_winid] else action_default,
                -1: "leave",
            },
        }

        main_ids = [self.id, other_id, -1]

        sender_id = self.winning_agents[other_id][task]
        sender_choice = ""
        if not (sender_id in main_ids):
            sender_choice = "default"
        else:
            sender_choice = sender_id

        this_id = self.winning_agents[self.id][task]
        this_choice = ""
        if not (this_id in main_ids):
            if this_id == sender_id:
                this_choice = "this_id"
            else:
                this_choice = "default"
        else:
            this_choice = this_id

        choice = case_table.get(sender_choice).get(this_choice)

        # Evaluate conditionals
        if not (type(choice) is str):
            choice = choice(int(sender_id), int(this_id))

        changed = False

        if choice == "update":
            self.log_verbose(iter_num, "Updating from {} at task {}".format(other_id, task, self.winning_bids[other_id][task], self.winning_agents[other_id][task]))
            self.winning_bids[self.id][task] = self.winning_bids[other_id][task]
            self.winning_agents[self.id][task] = self.winning_agents[other_id][task]
            changed = True
        elif choice == "reset":
            self.log_verbose(iter_num, "Resetting from {} at task {}".format(other_id, task))
            self.winning_bids[self.id][task] = 0
            self.winning_agents[self.id][task] = -1
            changed = True

        if changed and task in self.task_bundle:
            start_idx = min(n for n in range(len(self.task_bundle)) if self.winning_agents[self.id][self.task_bundle[n]] != self.id)

            for i in range(start_idx + 1, len(self.task_bundle)):
                self.winning_bids[self.id][self.task_bundle[i]] = 0
                self.winning_agents[self.id][self.task_bundle[i]] = -1

            for i in range(start_idx, len(self.task_bundle)):
                task2 = self.task_bundle.pop()
                self.task_path.remove(task2)
                self.assigned_tasks[task2] = 0

    def check_done(self, iter_num="."):
        # Se la lista dei bid è rimasta invariata
        if not (self.winning_bids[self.id] == self.prev_win_bids).all():
            self.prev_win_bids = self.winning_bids[self.id].copy()
            self.win_bids_equal_cnt = 0
            self.log_verbose(iter_num, "Max bids table changed: {}".format(self.winning_bids[self.id]))
        else:
            self.win_bids_equal_cnt += 1
            self.log_verbose(iter_num, "win_bids_equal_cnt: {}".format(self.win_bids_equal_cnt))

        # Numero di iterazioni senza alterazioni nello stato rilevante per considerare l'operazione finita
        num_stable_runs = 2 * len(self.agent_ids) * self.max_agent_tasks + 1
        # Se tutti i bid massimi sono stati ricevuti, ignorando quelli per i task che questo agente
        # sta ignorando (vale a dire quelle per cui ha messo bid 0)
        all_max_bids_set = all(self.winning_bids[self.id][task] != 0 for task in self.tasks if not self._ignores_task(task))

        return sum(self.assigned_tasks) <= self.max_agent_tasks and all_max_bids_set and self.win_bids_equal_cnt >= num_stable_runs

    def run_iter(self, iter_num = "."):
        self.construct_phase(iter_num)

        self.log_verbose(iter_num, "pre exchange, done: {}".format(self.done_neighbors))
        send_data = self._build_send_data()

        # IPOTESI: vicini non cambiano nel tempo
        # Se tutti i vicini hanno concluso, non chiedere informazioni per evitare hang
        if set(self.agent.in_neighbors) != set(self.done_neighbors):
            data = self.agent.neighbors_exchange(send_data)
            rec_time = time.time()
            self.log_verbose(iter_num, "post exchange, received: {}".format(data))

            self.changed_ids = []
            for other_id in filter(lambda id: id != self.id, data):
                self.winning_agents[other_id] = data[other_id]["winning_agents"]
                self.winning_bids[other_id] = data[other_id]["winning_bids"]
                self.message_times[other_id] = data[other_id]["message_times"]
                self.changed_ids.append(other_id)
                self.message_times[self.id][other_id] = rec_time

                if data[other_id]["done"]:
                    self.done_neighbors.append(other_id)
        else:
            self.agent.neighbors_send(send_data)
            self.changed_ids = []

        self.log_verbose(iter_num, "changed: {}".format(self.changed_ids))

        self.conflict_resolve_phase(iter_num)

        #Se tutti elementi sono != 0 e ha un task assegnato, e la lista è rimasta invariata per 2N+1 iterazioni
        if self.check_done(iter_num):
            self.done = True


    def _build_send_data(self):
        return {
            "winning_agents": self.winning_agents[self.id],
            "winning_bids": self.winning_bids[self.id],
            "message_times": self.message_times[self.id],
            # Trasmissione dello stato di conclusione, per permettere a
            # altri agenti di evitare di attendere quando tutti i vicini hanno concluso
            "done": self.done,
        }

    def run(self, beforeEach = None, max_iterations = -1):
        iterations = 0
        while not self.done:
            if beforeEach != None:
                beforeEach()

            self.log_verbose(iterations, "{}: iteration {} started".format(self.id, iterations))

            self.run_iter(iterations)

            self.log_verbose(iterations, "{}: iteration {} done".format(self.id, iterations))
            iterations = iterations + 1
            if max_iterations > 0 and iterations >= max_iterations:
                self.log(iterations, "{}: max iterations reached".format(self.id))
                self.done = True

        send_data = self._build_send_data()
        self.agent.neighbors_send(send_data) # Evitare hang di altri agenti ancora in attesa

        self.log_verbose(iterations, "Done, selected tasks {},".format(self.id, self.task_path))
        self.log_verbose(iterations, "Sent final data: {},".format(self.id, send_data))

    def get_result(self):
        return (self.task_path, self.task_bundle)

    # ~~Ignora bid nulli (valore default) a scopo bid negativi~~
    # Non più, per permettere task nello stesso luogo: usecase di bid negativi
    # sono sostituibili con 1 / x
    # Controllo spareggio con id se possibile
    def _bid_is_greater(self, bid1, bid2, id1 = -1, id2 = -1):
        if bid1 == bid2 and id1 >= 0 and id2 >= 0:
            return id1 > id2

        return bid1 > bid2

    def _ignores_task(self, task):
        return False

    def __repr__(self):
        return str(self.__dict__)

    def log(self, iter_num = None, *values):
        string = ""
        if iter_num is None:
            string = ("{:" + str(len(self.agent_ids)) + "d}:").format(self.id)
        else:
            string = ("{:" + str(len(self.agent_ids)) + "d} | {}:").format(self.id, iter_num)

        if self.log_file != '':
            with open(self.log_file, 'a') as f:
                print(string, *values, file=f)
        print(string, *values)

    def log_verbose(self, iter_num = None, *values):
        string = ""
        if iter_num is None:
            string = "{}:".format(self.id)
        else:
            string = "{} | {}:".format(self.id, iter_num)

        if self.verbose:
            print(string, *values)
        if self.log_file != '':
            with open(self.log_file, 'a') as f:
                print(string, *values, file=f)

class TimeScoreFunction(ScoreFunction):
    """
    S_{i} function for the CBBA algorithm.

    Init:
    agent_id: id of the agent
    time_discount_factors: float list containing static discount factors for each task
    calc_time_fun: function(agent_id, task, path) -> float, to calculate agent travel time to arrive at task location
        through the specified path.
    task_static_scores: static scores for each tasks' result to be multiplied by, must be a list.
    """
    def __init__(self, agent_id, task_discount_factors: list, calc_time_fun, task_static_scores: list = None):
        super().__init__(agent_id)
        self.task_discount_factors = task_discount_factors
        self.calc_time_fun = calc_time_fun
        self.task_static_scores = task_static_scores

    def do_eval(self, path: list) -> float:
        return sum((self.task_discount_factors[task] ** self.calc_time_fun(self.agent_id, task, path)
            * (self.task_static_scores[task] if not self.task_static_scores is None else 1)) for task in path)


## Utilità

# Operazione ⊕ nell'articolo
def insert_in_list(lst, pos, val):
    return lst[:pos + 1] + [val] + lst[pos + 1:]