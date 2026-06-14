# solving basic unconstrained optimization problem:
       
#  minimize: (x-2)^2

!pip install cvxpy
import cvxpy as cp

x = cp.Variable()

objective = cp.Minimize((x-2)**2)
problem = cp.Problem(objective)

problem.solve()

print("Optimal x =",x.value)
print("Minimum value =",problem.value)
