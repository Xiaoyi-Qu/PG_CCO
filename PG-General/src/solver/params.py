'''
File: params.py
Author: Xiaoyi Qu(xiq322@lehigh.edu)
File Created: 2025-07-14 00:52
--------------------------------------------
'''
from numpy import inf
params = {}

params["maxit"] = 100
params["tol_stationarity"] = 1e-5
params["tol_feasibility"] = 1e-5
params["kappav"] = 1000
params["eta_beta"] = 1e-4
params["gamma_beta"] = 0.5
params["eta_alpha"] = 1e-4
params['xi_alpha'] = 0.5
params["tau"] = 1

params["sigmac"] = 0.1
params["epsilon_tau"] = 0.1