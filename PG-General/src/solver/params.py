'''
File: params.py
Author: Xiaoyi Qu(xiq322@lehigh.edu)
File Created: 2025-07-14 00:52
--------------------------------------------
'''
from numpy import inf
params = {}

params["maxit"] = 4000
params["tol_stationarity"] = 1e-4
params["tol_feasibility"] = 1e-4
params["kappav"] = 0.02
params["eta_beta"] = 1e-6
params["gamma_beta"] = 0.5
params["eta_alpha"] = 1e-4
params['xi_alpha'] = 0.8
params["tau"] = 10
params["reg_param_add"] = 10
params["tr_bound_solver"] = "gurobi"  # Options: "cauchy", "gurobi"

params["sigmac"] = 0.1
params["epsilon_tau"] = 0.1

params["approximate_a"] = 1e-8
