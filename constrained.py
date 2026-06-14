'''consider the optimization problem

      minimize   :max{a, b}
      subject to :2a + b >=1
                   a + 3b>=1
                   a>=0, b>=0'''

import cvxpy as cp

x = cp.Variable()
y = cp.Variable()

objective = cp.Minimize(cp.maximum(x, y))
constraints = [2*x + y >=1, x + 3*y >= 1, x>=0, y>=0]

problem = cp.Problem(objective, constraints)

problem.solve()

print("a=", x.value)
print("b=", y.value)
