"""
Sparse Canonical Correlation Analysis (SCCA) Problem
"""

import argparse
import numpy as np
import sys
sys.path.extend(["../", "../.."])

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
    r = L1(indices=list(range(dim)), penalty=reg_param)

    # Define variable blocks
    slack_idx = list(range(nx + ny, nx + ny + 2))  # 2 slack variables

    # Define bounds for each block
    set_type = {((0,) * len(slack_idx), (np.inf,) * len(slack_idx)): slack_idx,}

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
