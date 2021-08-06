from disropt.functions import Variable
from disropt.problems import Problem
import numpy as np
import numpy.matlib as matlib

n = 3

x = Variable(n * n)
print("x:", x)

c = np.random.random_sample((n * n, 1))
print("c:", np.reshape(c, (n, n)))

fun = c @ x
print("fun:", fun)

# sel_tasks = np.array([ np.concatenate( tuple(np.roll([1, 0, 0], x + y) for x in range(3)) ) for y in range(3)])
sel_tasks = matlib.repmat(np.eye(n), 1, n)
print("matrice vincolo task:\n", sel_tasks)

sel_agents = np.array([ np.roll([1] * n + [0] * n * (n - 1), y * n) for y in range(n)])
print("matrice vincolo agent:\n", sel_agents)

cons = [
    sel_tasks.T @ x <= np.ones((n, 1)), 
    sel_agents.T @ x <= np.ones((n, 1)), 
    np.ones((9, 1)) @ x == n,
    x >= np.zeros((n * n, 1)),
    x <= np.ones((n * n, 1)),
]

prob = Problem(fun, cons)
sol = prob.solve()
print("===\nsol:", sol)

sol_mat = np.reshape(sol, (n, n))
print("Selected:", np.nonzero(sol_mat)[1])