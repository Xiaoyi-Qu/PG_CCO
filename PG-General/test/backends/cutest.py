"""
File: cca_data.py
Author: Xiaoyi Qu
Created: 2025-07-14 00:54

Description:
------------
This module defines a constrained optimization problem based on the CUTEst test set.
The problem is formulated as:

    minimize    f(x) + r(a)
    subject to  c(x, s) + a = 0
                cl ≤ s ≤ cu
                bl ≤ x ≤ bu

where:
    - c(x, s) = [ c_E(x); c_I(x) - s ]
      * c_E(x): equality constraints
      * c_I(x): inequality constraints
      * s: slack variable

Notes:
    - r(a) typically represents a regularization term or additional objective component.
    - The problem includes both bound constraints and general constraints with slack variables.
"""
        
import numpy as np
import pycutest

class CUTEst:
    def __init__(self, problem_name):
        self.p = pycutest.import_problem(problem_name)
        self.n = self.p.n
        self.m = self.p.m
        self.x0 = self.p.x0

        # Boolean masks
        self.eq_mask = (self.p.cl == self.p.cu)
        self.ineq_mask = ~self.eq_mask

        self.me = np.sum(self.eq_mask)
        self.mi = np.sum(self.ineq_mask)

        # Store indices for convenience
        self.eq_indices = np.where(self.eq_mask)[0]
        self.ineq_indices = np.where(self.ineq_mask)[0]

    def obj(self, x, gradient=False):
        x_true = x[:self.n]
        if gradient:
            f, g = self.p.obj(x_true, gradient=True)
            grad = np.concatenate([g, np.zeros(self.mi + self.m)])
            return f, grad
        else:
            return self.p.obj(x_true)

    def cons(self, x, gradient=False):
        """
        Evaluate the constraint and optionally its Jacobian
            c(x, s) + a = [c_E(x); c_I(x) - s] + a = 0
        """
        x_true = x[:self.n]
        s = x[self.n:self.n + self.mi]
        a = x[self.n + self.mi:]

        if gradient:
            c0, J0 = self.p.cons(x_true, gradient=True)

            # Extract equality and inequality constraints
            c_E = c0[self.eq_indices]
            c_I = c0[self.ineq_indices]

            J_E = J0[self.eq_indices, :]
            J_I = J0[self.ineq_indices, :]

            c = np.concatenate([
                c_E,
                c_I - s
            ]) + a

            # Build Jacobian (Double check this)
            J_x_top = np.hstack([J_E, np.zeros((self.me, self.mi)), np.eye(self.me)])
            J_x_bot = np.hstack([J_I, -np.eye(self.mi), np.eye(self.mi)])

            J = np.vstack([J_x_top, J_x_bot])
            return c, J
        else:
            c0 = self.p.cons(x_true)
            c_E = c0[self.eq_indices]
            c_I = c0[self.ineq_indices]
            c = np.concatenate([
                c_E,
                c_I - s
            ]) + a
            return c