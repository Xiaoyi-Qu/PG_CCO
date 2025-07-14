'''
File: cca_data.py
Author: Xiaoyi Qu
File Created: 2025-07-14 00:54
--------------------------------------------
- CUTEst test problem   
'''

import numpy as np

class Cutest:
    def __init__(self, problem):
        self.p = problem

    # Compute function value f(x,y) along with its gradient
    def obj(self, x, gradient=False):
        p = self.p
        mid = p.n
        end = p.n + p.m
        # f, g0 = p.obj(x[0:mid], gradient=True)
        # g = np.concatenate((g0, np.zeros(end-mid)), axis=0)
        if gradient:
            f, g0 = p.obj(x[0:mid], gradient=True)
            g = np.concatenate((g0, np.zeros(end-mid)), axis=0)
            return [f,g]
        else:
            f = p.obj(x[0:mid], gradient=False)
            return f

    # Compute the constraint function value c(x)+y along with its Jacobian
    def cons(self, x, gradient=False):
        p = self.p
        mid = p.n
        end = p.n + p.m
        # J = np.concatenate((J0, np.identity(end-mid, dtype="int")), axis=1)
        # J = np.float32(np.concatenate((J0, np.identity(end-mid, dtype="int")), axis=1))

        if gradient:
            c0, J0 = p.cons(x[0:mid], gradient=True)
            c = c0 + x[mid:end]
            J = np.concatenate((J0, np.identity(end-mid, dtype="int")), axis=1)
            return [c,J]
        else:
            c0= p.cons(x[0:mid], gradient=False)
            c = c0 + x[mid:end]
            return c