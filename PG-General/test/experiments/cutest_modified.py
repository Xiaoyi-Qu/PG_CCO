"""
File: cutest_modified.py
Author: Xiaoyi Qu
File Created: 2025-07-14 00:54
--------------------------------------------
Test problem 1:
    CUTEst variant test problem
"""

import argparse
import numpy as np
import sys
sys.path.append(["../", "../.."])
sys.path.append("/home/xiq322/PG_CCO/PG-General/test")
sys.path.append("/home/xiq322/PG_CCO/PG-General")
sys.path.append("/home/xiq322/PG_CCO/PG-General/experiments")

from backends.regularizer import L1
from backends.cutest import CUTEst
from src.solver.solve import solve
from src.solver.params import params

def setup_problem(reg_param=10, alpha=10):
    """
    Set up the CUTEst constrained optimization problem with slack and auxiliary variables.

    Returns:
        p: CUTEst problem instance
        r: Regularizer applied to auxiliary variables
        bounds: Tuple of (lower_bounds, upper_bounds)
        x0: Initial guess for [x; s; a]
        alpha: Penalty parameter
    """
    prob_name = "HS65"
    p = CUTEst(prob_name)

    # Number of variables and constraints
    n = p.n                 # Decision variables
    m = p.m                 # Total constraints
    me = p.me               # Equality constraints
    mi = p.mi               # Inequality constraints

    # Variable dimensions
    dim_x = n
    dim_s = mi              # Slack variables only for inequalities
    dim_a = m               # Auxiliary variables for all constraints

    # Initial values
    x0 = p.x0
    s0 = np.zeros(dim_s)
    a0 = np.zeros(dim_a)

    x_init = np.concatenate([x0, s0, a0])  # Full initial vector

    # Regularizer on auxiliary variable a (last m entries)
    r = L1(penalty=reg_param, indices=list(range(n + mi, n + mi + m)))

    # Bounds
    bl = p.p.bl if hasattr(p.p, 'bl') else [-np.inf] * n
    bu = p.p.bu if hasattr(p.p, 'bu') else [np.inf] * n
    cl = p.p.cl[p.ineq_indices] if mi > 0 else []
    cu = p.p.cu[p.ineq_indices] if mi > 0 else []

    lower_bounds = np.concatenate([bl, cl, [-np.inf] * m])
    upper_bounds = np.concatenate([bu, cu, [np.inf] * m])
    bounds = (lower_bounds, upper_bounds)

    return p, r, bounds, x_init, alpha

def main():
    # Set up the problem
    p, r, set_type, x, alpha = setup_problem()

    # Solve the problem
    info = solve(p, r, set_type, x, alpha, params)

    # Output results
    # print(info)

if __name__ == "__main__":
    main()
