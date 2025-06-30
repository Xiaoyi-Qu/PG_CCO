'''
File: params.py
Author: Xiaoyi Qu(xiq322@lehigh.edu)
File Created: 2023-07-07 10:02
--------------------------------------------
Description:
    Describe all parameters in detail:
            (1) kappav: parameter for trust region radius
            (2) sigmau: parameter for tauk update rule
            (3) epsilontau: parameter for tauk update rule
            (4) tau: merit parameter
            (5) xi: parameter related to proximal parameter update
'''
from numpy import inf
params = {}
params["max_iter"] = 1000
params["max_iter_admm"] = 1000
params["tol_stationarity"] = 1e-6 #old version 1e-6
params["tol_feasibility"] = 1e-6
params["ADMM_tol"] = 1e-6
params["printlevel"] = 2
params["printevery"] = 20
params["kappav"] = 1000
params["sigmau"] = 0.5+1e-4
params["sigmac"] = 0.1
params["epsilontau"] = 0.1
# params["tau"] = 1e-2
params["eta"] = 1e-4
params["xi"]  = 0.5

# old version
# params["sigmau"] = 1
# params["sigmac"] = 0.9
# params["epsilontau"] = 0.9
