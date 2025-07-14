"""
Sparse Canonical Correlation Analysis (SCCA) Problem
"""

import argparse
import numpy as np
import sys
sys.path.append(["../", "../.."])
sys.path.append("/home/xiq322/PG_CCO/PG-General/test")
sys.path.append("/home/xiq322/PG_CCO/PG-General")
sys.path.append("/home/xiq322/PG_CCO/PG-General/experiments")

from backends.regularizer import L1
from backends.cca import CanonicalCorrelation
from backends.cca_data import DataSCCA
from src.solver.solve import solve
from src.solver.params import params

def setup_problem(nx=8, ny=8, N=8, reg_param=10):
    # Initialize the CCA problem
    p = CanonicalCorrelation(DataSCCA(nx, ny, N))

    # Define the regularizer
    dim = len(p.x0)
    r = L1(penalty=reg_param, indices=list(range(dim)))

    # Define bounds for each block
    n_total = nx + ny
    lower_bounds = [-np.inf] * (n_total + 2)
    upper_bounds = [np.inf] * n_total + [1.0, 1.0]

    set_type = (lower_bounds, upper_bounds)

    # Initial solution with slack
    slack = np.array([0, 0]).reshape(2,1)
    x = np.concatenate((p.x0, slack), axis=0)

    alpha = 10
    return p, r, set_type, x, alpha

def main():
    # Set up the problem
    p, r, set_type, x, alpha = setup_problem()

    # Solve the problem
    info = solve(p, r, set_type, x, alpha, params)

    # Output results
    print(info)

if __name__ == "__main__":
    main()
