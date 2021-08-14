import numpy as np

class TaskTester:
    def __init__(self):
        # Non impostati nel costruttore, pensato per poter essere usato più volte con
        # dati diversi; servono principalmente a ottenerli dall'esterno nel multithreading
        self.done_agents = 0
        self.iterations = 0
        self.const_winning_bids = []
        self.log_file = ''
        self.last_solution = None

    def get_cur_status(self):
        return (self.done_agents, self.iterations, self.const_winning_bids, self.last_solution)

    # Parametri lasciati a estensioni, avendo i vari algoritmi diversi parametri, inclusi per autocomplete
    def run(self, num_agents, agent_positions:list = None, task_positions:list = None, verbose=False, log_file='', silent = False, return_iterations = False):
        pass

    def log(self, *values, do_console=True):
        if do_console:
            print(*values)
        if self.log_file != '':
            with open(self.log_file, 'a') as f:
                print(*values, file=f)

    def has_conflicts(self, solution):
        return np.any(np.sum(solution, 0) > 1)    