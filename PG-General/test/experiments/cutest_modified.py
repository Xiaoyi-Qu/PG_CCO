"""
File: cutest_modified.py
Author: Xiaoyi Qu
File Created: 2025-07-14 00:54
--------------------------------------------
Test problem 1:
    CUTEst variant test problem
    Criterion: problem with at least on inequality constraint
    Test on 96 - 14 = 82 test problems
    
    Retest S365
"""

import argparse
import numpy as np
import sys
from csv import DictReader
sys.path.append(["./", "../", "../.."])
sys.path.append("/home/xiq322/PG_CCO/PG-General/test")
sys.path.append("/home/xiq322/PG_CCO/PG-General")
sys.path.append("/home/xiq322/PG_CCO/PG-General/experiments")

from backends.regularizer import L1
from backends.cutest import CUTEst
from src.solver.solve import solve
from src.solver.params import params
from src.utils.helper import projection

def setup_problem(prob_name):
    """
    Set up the CUTEst constrained optimization problem with slack and auxiliary variables.

    Returns:
        p: CUTEst problem instance
        r: Regularizer applied to auxiliary variables
        bounds: Tuple of (lower_bounds, upper_bounds)
        x0: Initial guess for [x; s; a]
        alpha: Penalty parameter
    """
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

    # Access the regularization parameter
    # Specify regularization parameter
    reg_param = 0
    with open("/home/xiq322/PG_CCO/PG-General/test/experiments/preprocessing/multipliers.csv", 'r') as f1:
        dict_reader1 = DictReader(f1)
        prob_list = list(dict_reader1)
        for prob in prob_list:
            if prob["Problem Name"] == prob_name:
                reg_param = round(np.float64(prob["Multiplier"])) + params["reg_param_add"]

    if prob_name == "HS106":
        reg_param = 10

    if prob_name == "HS106":
        reg_param = 10

    if prob_name == "ERRINBAR":
        reg_param = 0.1
    
    if prob_name == "HS93":
        reg_param = 10000

    if prob_name == "TENBARS1":
        reg_param = 100

    if prob_name == "HS69":
        reg_param = 10

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

    # Initial values
    x0 = projection(p.x0, (bl, bu))
    slack0 = projection(np.zeros(dim_s), (cl, cu))
    # a0 = -(p.p.cons(x0) - slack0) # c(x, s) + a = 0
    a0 = np.zeros(dim_a)
    
    # x_init = np.concatenate([x0, slack0, a0])  # Full initial vector
    x_init = np.concatenate([x0, slack0, a0])  # Full initial vector
    a0 = -p.cons_noa(x_init)
    x_init = np.concatenate([x0, slack0, a0])  # Full initial vector

    alpha = 10

    return p, r, bounds, x_init, alpha

def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-n", "--name", type=str, default="HS101", help="Test problem name")
    parser.add_argument("-k", "--kappav", type=float, default=0.02, help="Kappa value for trust region method")

    # Parse arguments
    config = parser.parse_args()

    return config

def main(config):
    # Input problem name
    prob_name = config.name # "HS71" "HS72" "HS65" 
    params["kappav_gurobi"] = config.kappav

    # Set up the problem
    p, r, set_type, x, alpha = setup_problem(prob_name)

    # Solve the problem
    info = solve(p, r, set_type, x, alpha, params)

    # Output results
    # print(info)

if __name__ == "__main__":
    main(get_config())
