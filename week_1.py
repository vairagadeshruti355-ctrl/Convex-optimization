'''The intuition of this week is to understand basically what is the convex function ? how a optimization problem basically look like. 
building foundational knowledge about convex optimization. Understanding why convexity is computationally nice. Studying it's geometric 
intuition. '''

''' Starting with the basic'''

# solving basic unconstrained problem:
'''Problem Statement: Find the value of x that minimizes
                          f(x) = (x - 2)**2'''

!pip install cvxpy
import cvxpy as cp 
x = cp.Variable()

objective = cp.Minimize((x-2)**2)
problem = cp.Problem(objective)

problem.solve()

print("Optimal x =",x.value)
print("Minimum value =",problem.value)

'''here the function is convex since it's second derivative is strictly positive and it have the minima at the point (2, 0) where the 
graph converges and diverges to infinity'''

# solving basic constrained problems:
''' Problem statement: minimize f(x)
                       subject to x >= 1

                       where, f(x) = x**2'''

import cvxpy as cp
x = cp.Variable()

objective = cp.Minimize(x**2)
constraint = [x >= 1]

problem = cp.Problem(objective, constraint)
problem.solve()

print("x: ",x.value)

'''Problem statement: consider the optimization problem

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


# least square problem
''' A least square problem is an optimization problem with no constraints (i.e., m = 0) and an objective which is a sum of squares 
of terms of the form a[i]^T*x - b[i]
                               minimize f(x) = || Ax - b || 
here A belongs to vector space with the dimension of (k * n) where (k >= n), a[i]^T are the rows of A, and the vector x belongs to 
the vector space with the dimenion (n * 1) is the optimization variable'''

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
