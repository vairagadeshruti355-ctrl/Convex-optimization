'''Sparse regression, Ridge regression, and LASSO regression are all techniques used when there are many features (variables) and for
building a model that generalizes well without overfitting.
starting from ordinary linear regression:

                         y = β0 + β1x1 + β2x2 +.........+ βnxn
	​
y = target value
xi = features
βi = coefficients learned by the model  '''

# Ridge regression (L2 regularization)
'''Ridge regression minimizes:

                                       ==     RSS + λ*∑(β(j))**2  ....(for j = 1 to n)
	​


where RSS is:

                                      ==      ∑(y(i) − y(i)^)**2 ..........,       and λ controls the strength of the penalty.  '''

# Basic problem based on ridge regression

# Q1 ridge regression
import cvxpy as cp
import numpy as np
X = np.array([
    [1, 2, 3, 4, 5],
    [2, 1, 0, 1, 2],
    [3, 2, 1, 0, 1],
    [1, 1, 1, 1, 1]
])
beta = cp.Variable(5)
y = np.array([10, 5, 6, 4])
lam = 0.5

objective = cp.Minimize(cp.sum_squares(X @ beta.T - y) + lam*cp.sum_squares(beta))
problem = cp.Problem(objective)

problem.solve()
print("ans: ",problem.value)


#2. LASSO Regression (L1 Regularization)

'''LASSO minimizes:

                  == RSS + λ*∑ ∣ β(j) ∣ .......(for j = 1 to n)

there are absolute value instead of squares.

What does LASSO do?
It can force coefficients to become exactly zero.

Problem Statement:

Given a matrix A∈R^m×n and a vector b∈R^m, find x∈R^n that solves x
                               min   (||Ax−b||2)**2 + λ∥x∥1
    
​1.x is the decision variable,
2.(∥Ax−b∥2)**2 measures the fitting error,
3.∥x∥1 promotes sparsity,
4.λ>0 controls the amount of regularization.'''

#Q2 LASSO Regression
import cvxpy as cp
import numpy as np
x = cp.Variable(3)
A = np.array([
    [1, 2, 3],
    [2, 1, 0],
    [1, 1, 1],
    [3, 2, 1]
])
b = np.array([10, 5, 4, 8])
lam = 0.5
objective = cp.Minimize(cp.sum_squares(A @ x.T - b) + lam*cp.norm1(x))
problem = cp.Problem(objective)

problem.solve()
print("ans: ",problem.value)



'''Sparse Regression

A sparse model is simply a model where most coefficients are zero:

β=[5,0,0,2,0,0,1]

Only a few variables are active.

Geometric Intuition

Imagine coefficient space.

         Ridge Constraint:
                        ∑(β(j))**2 ≤ t

Forms a circle (2D) or sphere (higher dimensions).

The optimum usually touches the smooth boundary away from the axes. '''
# Problem statement:

'''Fit the data while limiting total coefficient magnitude.

                                     min (||Xβ−y||2)**2
	​


                                     subject to     ∥β∥(1) ≤5'''
#Q3 sparse regression

import cvxpy as cp
import numpy as np

X = np.array([
    [1 ,2 , 3, 4, 5],
    [2, 1, 0, 1, 2],
    [3, 2, 1, 0, 1]
]) 
beta = cp.Variable(5)
y = np.array([10, 5, 6])

objective = cp.Minimize(cp.sum_squares(X @ beta.T - y))

constraints = [cp.norm(beta, 1) <= 5]

problem = cp.Problem(objective, constraints)

problem.solve()

print("Ans", problem.value)


