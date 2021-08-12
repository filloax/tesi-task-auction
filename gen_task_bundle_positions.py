import csv
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

def generate_positions(num_agents: int, num_tasks: int):
    plane_size = 40

    agent_positions = np.random.randint(-plane_size / 2, plane_size / 2, (num_agents, 2))
    task_positions = np.random.randint(-plane_size / 2, plane_size / 2, (num_tasks, 2))
    return (agent_positions, task_positions)

def load_positions(agent_pos_path, task_pos_path):
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

    return (task_positions, agent_positions)

if __name__ == '__main__':
    args = parser.parse_args()
    (agent_positions, task_positions) = generate_positions(args.agents, args.tasks)
    write_positions(agent_positions, task_positions, args.agent_pos_path, args.task_pos_path)