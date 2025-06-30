'''
The general problem formulation
min f(x) + r(x)
s.t. c(x) = 0, x >= 0.

min_x 10 (x2 + 1 - (x1 + 1)^2)^2 + |x1| 
s.t.  x1 >= 0, x2 >= 0.
'''
 
# class Rosenbrock:
#     def __init__(self, n, m, set_type):
#         self.n = n
#         self.m = m
#         self.set_type = set_type

#     # Compute function value f(x,y) along with its gradient
#     def obj(self, x, gradient=False):
#         if gradient:
#             f = None
#             g = None
#             return [f,g]
#         else:
#             f = None
#             return f

#     # Compute the constraint function value c(x)+y along with its Jacobian
#     def cons(self, x, gradient=False):
#         if gradient:
#             c = None
#             J = None
#             return [c,J]
#         else:
#             c = None
#             return c
        

import numpy as np

class Rosenbrock:
    def __init__(self, n=2, m=0, set_type='positive_orthant'):
        self.n = n
        self.m = m
        self.set_type = set_type

    # Compute objective value f(x) + r(x)
    def obj(self, x, gradient=False):
        x1, x2 = x[0], x[1]
        residual = x2 + 1 - (x1 + 1)**2
        f = 10 * residual**2 + x1  # x1 = |x1| since x1 >= 0

        if gradient:
            # Gradient of smooth part
            df_dx1 = -40 * residual * (x1 + 1) + 1
            df_dx2 = 20 * residual
            g = np.array([df_dx1, df_dx2])
            return [f, g]
        else:
            return f

    # There are no equality constraints, return zeros
    def cons(self, x, gradient=False):
        if gradient:
            c = np.array([])  # No constraints
            J = np.zeros((0, len(x)))  # Empty Jacobian
            return [c, J]
        else:
            c = np.array([])  # No constraints
            return c
