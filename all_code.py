#solving basic unconstrained optimization problem:
 '''1)minimize: (x-2)^2'''

!pip install cvxpy
import cvxpy as cp
x = cp.Variable()

objective = cp.Minimize((x-2)**2)
problem = cp.Problem(objective)

problem.solve()

print("Optimal x =",x.value)
print("Minimum value =",problem.value)

'''here the function is convex since it's second derivative is strictly positive and is convex at it's minimum value it diverges to infinity and converges 
at the point (2,0). the objective is to find the point where the function attains it's minimum value'''

'''2)consider the optimization problem

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

'''least square regression: given a set of data points (xi, yi) ,      
determine best-fitting line y = ax + b that minimizes the sum of squared prediction errors.'''

import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


np.random.seed(0)

x_data = np.linspace(0,10,50)
y_data = 2*x_data + 1 + np.random.randn(50)

a = cp.Variable()
b = cp.Variable()

objective = cp.Minimize(cp.sum_squares(a*x_data + b - y_data))

problem = cp.Problem(objective)
problem.solve()

print(problem.status)
print(a.value, b.value)

plt.scatter(x_data, y_data)
plt.plot(x_data, a.value*x_data+b.value)
plt.xlabel("x")
plt.ylabel("y")
plt.title("linear Squares regression")
plt.show()




