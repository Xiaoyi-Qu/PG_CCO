'''
# File: problem.py
# Project: First order method for CCO
# Description: The code is to obtaini nformation for the QCCO test problem.
# Description: Test on problems in the form of
#       min  x^THx + \gamma*\|x\|_1
#       s.t. 0.5x^TA_{i}x + b_{i}^Tx + c_{i}=0, i=1,2,...,m.
# ----------	---	----------------------------------------------------------
'''

import numpy as np
import random

def generate_diagonal_matrix(n):
    """Generate a diagonal matrix with random diagonal elements."""
    diag_elements = np.random.rand(n)
    D = np.diag(diag_elements)
    return D

def generate_random_orthogonal_matrix(n):
    """Generate a random orthogonal matrix."""
    A = np.random.randn(n, n)
    Q, _ = np.linalg.qr(A)
    return Q

def generate_matrix_H(n):
    """Generate matrix H = P.T @ D @ P."""
    D = generate_diagonal_matrix(n)
    P = generate_random_orthogonal_matrix(n)
    H = P.T @ D @ P
    return H

class QCCO:
    def __init__(self, n, m):
        self.n = n
        self.m = m
        self.H = generate_matrix_H(self.n)
        matrices = [generate_matrix_H(self.n) for _ in range(self.m)]
        vectors = np.ones(self.m)
        self.A = matrices
        self.b = vectors

    # Compute function value f(x,y) along with its gradient
    def obj(self, x, gradient=False):
        if gradient:
            f = x.T @ self.H @ x
            g = 2 * (self.H @ x)
            return [f,g]
        else:
            f = x.T @ self.H @ x
            return f

    # Compute the constraint function value c(x)+y along with its Jacobian
    def cons(self, x, gradient=False):
        if gradient:
            c = np.zeros(self.m)
            J = np.zeros((self.m, self.n))
            for i in len(self.matrics):
                c[i] = 0.5 * x.T @ self.matrics[i] @ x + self.b[i].T @ x - 1
                J[i,:] = self.matrics[i] @ x + self.b[i]
            return [c,J]
        else:
            c = np.zeros(self.m)
            for i in len(self.matrics):
                c[i] = 0.5 * x.T @ self.matrics[i] @ x + self.b[i].T @ x - 1
            return c