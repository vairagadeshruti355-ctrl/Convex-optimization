#least square problem

import cvxpy as cp
import numpy as np

A = np.random.randn(3,3)
b = np.random.randn(3)

x = cp.Variable(3)
objective = cp.sum_squares(A @ x - b)
problem = cp.Problem(cp.Minimize(objective))
problem.solve()

print("Optimal value: ",problem.value)
print("x: ",x.value)
