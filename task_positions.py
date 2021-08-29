import csv
from random import shuffle
from disropt.agents import agent
import numpy as np
import argparse

DEFAULT_AGENT_FILE = "./agent-positions.csv"
DEFAULT_TASK_FILE = "./task-positions.csv"

parser = argparse.ArgumentParser(description='Generate task and agent positions for Disropt test of CBBA.')
parser.add_argument('-n', '--agents', help="Amount of agents", required=True, type=int)
parser.add_argument('-t', '--tasks', help="Amount of tasks", required=True, type=int)
parser.add_argument('--agent-pos-path', type=str, default=DEFAULT_AGENT_FILE)
parser.add_argument('--task-pos-path', type=str, default=DEFAULT_TASK_FILE)

def write_positions(agent_positions: list, task_positions: list, agent_pos_path: str = DEFAULT_AGENT_FILE, task_pos_path: str = DEFAULT_TASK_FILE):
    print("Agents:")
    with open(agent_pos_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(agent_positions)):
            writer.writerow(agent_positions[i])
            print("{}: [{}]".format(i, ' '.join(str(x) for x in agent_positions[i])))
    print("Tasks:")
    with open(task_pos_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=' ', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for i in range(len(task_positions)):
            writer.writerow(task_positions[i])
            print("{}: [{}]".format(i, ' '.join(str(x) for x in task_positions[i])))

def generate_positions(num_agents: int, num_tasks: int, 
        test_same_positions: bool = False, 
        avoid_same_positions: bool = False, 
        test_same_distance: bool = False,
        test_tasks_same_distance: bool = False,
        ):
    plane_size = 40

    agent_positions = []
    task_positions = []

    if not avoid_same_positions:
        agent_positions = np.random.randint(-plane_size / 2, plane_size / 2, (num_agents, 2))
        task_positions = np.random.randint(-plane_size / 2, plane_size / 2, (num_tasks, 2))
    else:
        # Don't make the plane too big if you don't want this to weigh on RAM
        positions = [(x, y) for x in range(plane_size) for y in range(plane_size)]
        shuffle(positions)
        for i in range(num_agents):
            agent_positions.append(positions.pop())
        for i in range(num_tasks):
            task_positions.append(positions.pop())
        agent_positions = np.array(agent_positions)
        task_positions = np.array(task_positions)

    if test_tasks_same_distance:
        task_positions[-1] = -task_positions[0] + agent_positions[0] * 2

    if test_same_positions:
        # Test agenti con stessa posizione iniziale
        agent_positions[-1] = agent_positions[0]
        #Test task con stessa posizione iniziale
        task_positions[-1] = task_positions[0]

    if test_same_distance:
        # Rifletti agente -2 intorno a task 0 partendo da agente 0 così da avere la stessa distanza
        agent_positions[-2] = -agent_positions[0] + task_positions[0] * 2

    return (agent_positions, task_positions)

def load_positions(agent_pos_path = DEFAULT_AGENT_FILE, task_pos_path = DEFAULT_TASK_FILE):
    task_positions = []
    with open(task_pos_path) as task_file:
        reader = csv.reader(task_file, delimiter=' ', quotechar='|')
        for row in reader:
            task_positions.append([float(row[0]), float(row[1])])
    task_positions = np.array(task_positions)

    agent_positions = []
    with open(agent_pos_path) as agent_file:
        reader = csv.reader(agent_file, delimiter=' ', quotechar='|')
        for row in reader:
            agent_positions.append([float(row[0]), float(row[1])])
    agent_positions = np.array(agent_positions)

    return (agent_positions, task_positions)

def sq_linear_dist(pos1, pos2):
    return np.sum((pos1 - pos2) ** 2)

def linear_dist(pos1, pos2):
    return np.sqrt(sq_linear_dist(pos1, pos2))

def gen_distance_calc_time_fun(agent_positions, task_positions):
    def calc_time_fun(agent_id, task, path: list) -> float:
        if task in path:
            task_id = path.index(task)
            out = 0

            for i in range(task_id + 1):
                if i == 0:
                    out += sq_linear_dist(agent_positions[agent_id], task_positions[path[i]])
                else:
                    out += sq_linear_dist(task_positions[path[i - 1]], task_positions[path[i]])

            return out
        else:
            raise ValueError("Task not present in specified path!")
    return calc_time_fun

if __name__ == '__main__':
    args = parser.parse_args()
    (agent_positions, task_positions) = generate_positions(args.agents, args.tasks)
    write_positions(agent_positions, task_positions, args.agent_pos_path, args.task_pos_path)