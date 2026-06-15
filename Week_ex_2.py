''' In week 2, we are focusing on learning the large language modelling various topics are covered in this week like variables, 
objectives, constraints, linear programming, quadratic optimization. Translating word problems into optimization problems'''

# solving model problems:
'''Problem Statement: Minimum-Cost Healthy Diet
A student wants to design the cheapest diet that satisfies daily nutritional requirements.
Foods Available     
Food      Cost (₹/serving)    Calories    Protein (g)    Fat (g)
Rice          20                200          4            0.5
Milk          30                150          8             5
Eggs          10                70           6             5
Let
x = servings of Rice
y = servings of Milk
z = number of Eggs
These are the decision variables.
Objective Function : Minimize the total cost:
Constraints  1. Minimum calorie requirement Need at least 2000 calories per day; 2. Minimum protein requirement Need at least 50 g protein;
3. Maximum fat intake Fat should not exceed 60 g; 4. Non-negativity of cost variables.'''

import cvxpy as cp
import numpy as np

x, y, z = cp.Variable(3)

objective = cp.Minimize(20*x + 30*y + 10*z)
constraints = [200*x + 150*y + 70*z >=2000, 4*x + 8*y + 6*z >= 50, 0.5*x + 5*y + 5*z <= 60, x >= 0, y >= 0, z >=0]

problem = cp.Problem(objective, constraints)
problem.solve()

print("Servings of Rice: ",x.value)
print("Servings of Milk: ",y.value)
print("number of eggs: ",z.value)

'''problem statement 2:
Problem Statement: Budget Allocation
A company has a budget of 100 units to allocate among 3 projects. Let:
x = budget allocated to Project 1
y = budget allocated to Project 2
z = budget allocated to Project 3
The cost (or dissatisfaction) associated with each project is:
                  f(x, y, z) = x**2 + 2*y**2 + 3*z**2
The company wants to minimize the total cost while using the entire budget.

                Optimization Problem: Minimize: f(x, y, z)
                                      subject to: x + y + z = 100
                                      x, y, z >= 0 '''
import cvxpy as cp
import numpy as np

x, y, z = cp.Variable(3)

objective = cp.Minimize(x**2 + 2*y**2 + 3*z**2)
constraints = [x + y + z == 100, x >= 0, y >= 0, z >= 0]

problem = cp.Problem(objective, constraints)
problem.solve()

print("budget allocated to project 1: ",x.value)
print("budget allocated to project 2: ",y.value)
print("budget allocated to project 3: ",z.value)

'''problem 3:
Transportation Problem Statement
A company has 3 warehouses supplying goods to 4 stores.
Supply at warehouses
Warehouse 1: 40 units
Warehouse 2: 50 units
Warehouse 3: 60 units
Demand at stores
Store 1: 20 units
Store 2: 30 units
Store 3: 50 units
Store 4: 50 units
Transportation cost per unit

                             Store 1          Store 2         Store 3        Store 4
Warehouse 1                    4                 6              8              13
Warehouse 2                    5                11              9              7
Warehouse 3                    7                 4             10              6
Let x[i][j] denote the number of units transported from warehouse i to store j.

optimization problem : incompleted'''

import cvxpy as cp
import numpy as np

A = np.array([
    [4, 6, 8, 13, 5, 11, 9, 7, 7, 4, 10, 6]
    ])
x = cp.Variable(12, nonneg = True)
objective = cp.Minimize(x @ A.T)
constraints = [x[0] + x[3] + x[6] + x[9] == 40, x[1] + x[4] + x[7] + x[10] == 50, x[2] + x[5] + x[8] + x[11] == 60]

problem = cp.Problem(objective, constraints)
problem.solve()

print("units: ",problem.value)  # error
print("x:",x.value)

'''problem 4:
Problem Statement: Job Scheduling on a Single Machine
A company has n jobs to process on a single machine. Let:
x(i) = processing time allocated to job i
w(i) = importance (weight) of job i
a(i) = parameter controlling job efficiency
Total available machine time = T
The completion quality of job i improves with more processing time according to
The goal is to allocate machine time among jobs to minimize total cost.

optimization problem:
minimize sum( -w(i)*log(1 + a(i)*x(i)) ) ......(i = 1 to i = n)
subject to: sum(x(i)) <= T (T == 10)......(i = 1 to i = n)
                x(i)  >= 0, .......(i = 1 to i = n)'''
import cvxpy as cp
import numpy as np

x = cp.Variable(5)
w = np.array([2, 1, 3, 2, 4])
a = np.array([1.0, 0.8, 1.2, 0.5, 1.5])

objective = cp.Minimize(cp.sum(-w @ cp.log(1 + cp.multiply(a, x) )))
constraints = [cp.sum(x) <= 10, x >= 0]

problem = cp.Problem(objective, constraints)
problem.solve()

print("Optimal x: ",x.value)

