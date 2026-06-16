'''Studying Practical convexity, common convex function, Norms & composition rules.
understanding why CVXPY accepts / rejects expressions
learning disciplined convex programming & how to write optomization problems in a way solvers understand.'''

'''A convex optimization problem is an optimization problem that minimizes a convex function subject to affine equality constraint
and convex inequality constraint (f(x) <= 0, where f(x) is a convex function)'''

# DCP problem 
'''A problem is constructed from an objective and a list of constraints. If a problem follows the DCP rules, it is guaranteed to be
convex and solvable by CVXPY. The DCP rules require that the problem objective have one of two forms:
                    1. Minimize(convex)
                    2. Maximize(concave)

The only valid constraints under the DCP rules are:

                                              1.  affine == affine
                                              2.  convex <= concave
                                              3.  concave >= convex'''
import cvxpy as cp
x = cp.Variable()
y = cp.Variable()

# DCP problems.
prob1 = cp.Problem(cp.Minimize(cp.square(x - y)),
                    [x + y >= 0])
prob2 = cp.Problem(cp.Maximize(cp.sqrt(x - y)),
                [2*x - 3 == y,
                 cp.square(x) <= 2])

print("prob1 is DCP:", prob1.is_dcp())
print("prob2 is DCP:", prob2.is_dcp())

# Non-DCP problems.

# A non-DCP objective.
obj = cp.Maximize(cp.square(x))
prob3 = cp.Problem(obj)

print("prob3 is DCP:", prob3.is_dcp())
print("Maximize(square(x)) is DCP:", obj.is_dcp())

# A non-DCP constraint.
prob4 = cp.Problem(cp.Minimize(cp.square(x)),
                    [cp.sqrt(x) <= 2])

print(f"prob4 is DCP: {prob4.is_dcp()}")
print(f"sqrt(x) <= 2 is DCP: {(cp.sqrt(x) <= 2).is_dcp()}")

# practical convexity 

'''showing basic least regression graph'''

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


'''2 plotting graph of f(x, y) = x**2/y'''

import numpy as np
import matplotlib.pyplot as plt

# create grid of x and y values
x = np.linspace(-5, 5, 100)
y = np.linspace(0.1, 5, 100) # avoiding y = 0

# compute function values
X, Y = np.meshgrid(x, y)
Z = X**2/Y

# create 3D plot
fig = plt.figure(figsize = (8,6))
ax = fig.add_subplot(111, projection = '3d')

ax.plot_surface(X, Y, Z, cmap = 'viridis')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title(r'$f(x, y)=x^2/y$')

plt.show()

                                          
