'''
# File: main.py
# Project: Proximal gradient type method for solving equality constrained composite optimization
# Author: Xiaoyi Qu
# Description: Test on problems in the form of
#       min  x^THx + \gamma*\|x\|_1
#       s.t. 0.5x^TA_{i}x + b_{i}^Tx + c_{i}=0, i=1,2,...,m.
In our experiments, the matrix H in (5.1) is generated in the following way: We first generate
a diagonal matrix D by using Matlab function D = diag(rand(n, 1)) and a random orthogonal
matrix by P = orth(rand(n)), and then let H = P^TDP. The matrices Ai, i = 1, . . . , m, are
generated in the same way as H. In the implementation of SQP-PG, we use diag(Bk) to
approximate Bk in (3.1), where Bk is updated by the damped LBFGS method. Similar to the
proof of [40, Lemma 3.1], we can prove that diag(Bk) satisfies Assumption 3.2.
# ----------------------------------------------------------------------
'''

import argparse
import numpy as np
import pycutest as pycutest
import sys
from csv import DictReader
sys.path.append("../")
sys.path.append("../..")

from src.funcs.qcco_regularizer import L1
from src.funcs.qcco import QCCO
from src.solvers.solve import solve
from src.solvers.params import *
# from src.solvers.algorithm import AlgoBase
# import src.solvers.helper as helper 

################################
### Start the implementation ###
################################

def main(config):
    # Read from the command line
    prob_name = config.name
    tau = config.tau
    
    # Information
    reg_param = 0.1
    alpha = 10
    n = 100
    m = 50

    # Preparation
    params["filename"] = 'QCCO'
    params["tau"] = tau
    x = 0.1*np.ones(n)
    p = QCCO(n, m)
    r = L1(penalty=reg_param)
    Fopt = 0

    # Main solver
    info = solve(p, r, x, alpha, params, normal_step_strategy="Newton+Cauchy",
                tangential_step_strategy="Gurobi")

    # Handle filename
    outID = params["filename"] + "_tau_" + str(tau) + "_lambda_" + str(r.penalty)
    if outID is not None:
        filename = './log/{}.txt'.format(outID)
    else:
        filename = './log/log.txt'

    # Add result summary
    with open(filename, "a") as logfile:
        if info["ustatus"] == 5:
            content = "Model was proven to be unbounded.\n"
        elif info["ustatus"] == 12:
            content = "Optimization was terminated due to unrecoverable numerical difficulties.\n"
        else:
            content = ""
        content += "******************************************************************************\n"
        content += "Final Results\n"
        content += "Objective value (f):............................................................%8.6e\n"% info["fval"]
        content += "Objective value (f+r):............................................................%8.6e\n"% info["objective"]
        content += "Constraint violation:.......................................................%8.6e\n"% info["constraint_violation"]
        
        # Outputs regarding sparsity
        content += "Number of zero:.............................................................%s\n"% str(info["num_zero"])
        content += "First iteration sparsity is recognized:.....................................%s\n"% str(info["first_iter_sparsity_idnetified"])
        content += "Sparsity pattern exists or not:.............................................%s\n"% info["sparsity_existence"]
        content += "Infinity norm of term y:....................................................%8.6e\n"% info["inf_norm_y"]

        # Outputs regarding termination condition
        content += "Proximal parameter:.........................................................%8.6e\n"% info["prox_param"]
        content += "Chi_criteria:.................................................%s\n"% str(info["chi_criteria"])
        content += "Status:.....................................................................%1d\n"% info["status"]
        
        # Outputs regarding computational cost
        content += "Total iteration:............................................................%s\n"% str(info["iteration"])
        content += "Total function evaluations:.................................................%s\n"% str(info["func_eval"])
        content += "Total gradient evaluations:.................................................%s\n"% str(info["grad_eval"])
        content += "Total constraint evaluations:...............................................%s\n"% str(info["cons_eval"])
        content += "Total Jacobian evaluations:.................................................%s\n"% str(info["jacb_eval"])
        content += "Elapsed time:...............................................................%8.6es\n"% info["elapsed_time"]
        content += "x = "
        content += str(np.array(info['x']))
        logfile.write(content)

def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-n", "--name", type=str, default="HS9", help="Test problem name")
    parser.add_argument("-t", "--tau", type=float, default=1, help="Set tau parameter")

    # Parse arguments
    config = parser.parse_args()

    return config

if __name__ == "__main__":
    main(get_config())
