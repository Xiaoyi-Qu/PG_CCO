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
from src.solver.params import params  # make sure this is properly defined

def setup_problem(nx=8, ny=8, N=8, reg_param=10):
    # Initialize the CCA problem
    p = CanonicalCorrelation(DataSCCA(nx, ny, N))

    # Define the regularizer
    dim = len(p.x0)
    r = L1(indices=list(range(dim)), penalty=reg_param)

    # Initial solution with slack
    slack = np.array([0, 0]).reshape(2,1)
    print(p.x0.shape, slack.shape)
    x = np.concatenate((p.x0, slack), axis=0)

    alpha = 10
    return p, r, x, alpha

def main():
    # Set up the problem
    p, r, x, alpha = setup_problem()

    # Solve the problem
    info = solve(p, r, x, alpha, params)

    # Output results
    # print(info)

if __name__ == "__main__":
    main()



# def main(config):
#     # Read from the command line
#     prob_name = config.name
#     tau = config.tau
    
#     # Specify regularization parameter
#     reg_param = 0
#     with open("./cutest_prob_data/Lagrange_multiplier.csv", 'r') as f1:
#         dict_reader1 = DictReader(f1)
#         prob_list = list(dict_reader1)
#         for prob in prob_list:
#             if prob["\ufeffName"] == prob_name:
#                 reg_param = round(np.float64(prob["Estimate"])) + 10
    
#     # reg_param = 1000
#     # Preparation
#     problem = pycutest.import_problem(prob_name)
#     params["filename"] = prob_name
#     params["tau"] = tau
#     x1 = problem.x0
#     y = -problem.cons(x1)
#     # if prob_name == "BT7":
#     #     x1 = np.array([-20,10,1,1,1])
#     #     y = -problem.cons(x1)
#     dim = len(problem.x0)
#     x = np.concatenate((x1,y), axis=0)
#     alpha = 10
#     p = Problem(problem)
#     r = L1(dim, penalty=reg_param)
#     Fopt = 0

#     # Main solver
#     info = solve(p, r, x, alpha, params, normal_step_strategy="Newton+Cauchy",
#                 tangential_step_strategy="Gurobi")

#     # Handle filename
#     outID = params["filename"] + "_tau_" + str(tau) + "_lambda_" + str(r.penalty)
#     if outID is not None:
#         filename = './log/{}.txt'.format(outID)
#     else:
#         filename = './log/log.txt'

#     # Obtain actual optimal value
#     with open("./cutest_prob_data/opt.csv", 'r') as f2:
#         dict_reader2 = DictReader(f2)
#         prob_list = list(dict_reader2)
#         for prob in prob_list:
#             if prob["\ufeffName"] == prob_name:
#                 Fopt = np.float64(prob["Opt"])

#     # Compute the relative error
#     if Fopt == 0:
#         relative_error = abs(info["objective"])
#     else:
#         relative_error = abs((info["objective"] - Fopt)/Fopt)

#     # Add result summary
#     with open(filename, "a") as logfile:
#         if info["ustatus"] == 5:
#             content = "Model was proven to be unbounded.\n"
#         elif info["ustatus"] == 12:
#             content = "Optimization was terminated due to unrecoverable numerical difficulties.\n"
#         else:
#             content = ""
#         content += "******************************************************************************\n"
#         content += "Final Results\n"
#         content += "Objective value (f):............................................................%8.6e\n"% info["fval"]
#         content += "Objective value (f+r):............................................................%8.6e\n"% info["objective"]
#         content += "Constraint violation:.......................................................%8.6e\n"% info["constraint_violation"]
#         content += "Relative error:.............................................................%8.8e\n"% relative_error
        
#         # Outputs regarding sparsity
#         content += "Number of zero:.............................................................%s\n"% str(info["num_zero"])
#         content += "First iteration sparsity is recognized:.....................................%s\n"% str(info["first_iter_sparsity_idnetified"])
#         content += "Sparsity pattern exists or not:.............................................%s\n"% info["sparsity_existence"]
#         content += "Infinity norm of term y:....................................................%8.6e\n"% info["inf_norm_y"]

#         # Outputs regarding termination condition
#         content += "Proximal parameter:.........................................................%8.6e\n"% info["prox_param"]
#         content += "Chi_criteria:.................................................%s\n"% str(info["chi_criteria"])
#         content += "Status:.....................................................................%1d\n"% info["status"]
        
#         # Outputs regarding computational cost
#         content += "Total iteration:............................................................%s\n"% str(info["iteration"])
#         content += "Total function evaluations:.................................................%s\n"% str(info["func_eval"])
#         content += "Total gradient evaluations:.................................................%s\n"% str(info["grad_eval"])
#         content += "Total constraint evaluations:...............................................%s\n"% str(info["cons_eval"])
#         content += "Total Jacobian evaluations:.................................................%s\n"% str(info["jacb_eval"])
#         content += "Elapsed time:...............................................................%8.6es\n"% info["elapsed_time"]
#         content += "x = "
#         content += str(np.array(info['x']))
#         logfile.write(content)

# def get_config():
#     parser = argparse.ArgumentParser()

#     # Add arguments
#     parser.add_argument("-n", "--name", type=str, default="HS9", help="Test problem name")
#     parser.add_argument("-t", "--tau", type=float, default=1, help="Set tau parameter")

#     # Parse arguments
#     config = parser.parse_args()

#     return config

# if __name__ == "__main__":
#     main(get_config())