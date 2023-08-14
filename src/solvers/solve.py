'''
# File: solve.py
# Project: First order method for CCO
# Description: The code is to implement the algorithm.
#              status: (1) 0: optimal.
#                      (2) 1: reach the maximum iteration.
#                      (3) 2: infeasible stationary point.
#                      (4) 3: issue with the Gurobi model.
# ----------	---	----------------------------------------------------------
'''

import sys
import numpy as np
import time
from numpy.linalg import matrix_rank
sys.path.append("../")
sys.path.append("../..")
from src.solvers.params import *
from src.solvers.algorithm import AlgoBase
import src.solvers.helper as helper

def solve(p, r, x, alpha, params, normal_step_strategy,
            tangential_step_strategy):
    '''
    Arguments:
        p: problem instance object
        r: regularizer object
        x: starting point
        alpha: initial proximal parameter
        params: a list that stores parameters
    '''

    # Create dummy variables
    status = -1
    ustatus = -1
    iteration = 0
    iter_identify = 0
    s = np.zeros(len(x))
    judge = "No"
    ls_judge = -1
    num_feval = 0
    num_geval = 0
    num_ceval = 0
    num_Jeval = 0
    elapsed_time = 0
    # info['x'] = x
    # info['status'] = status
    # info["iteration"] = iteration

    info = {}
    g = p.obj(x, gradient=True)[1] 
    c, J = p.cons(x, gradient=True)
    dim = len(x)
    maxit = params["max_iter"]
    tol = params["main_tol"]
    eta = params["eta"]
    xi  = params["xi"]
    printevery = params["printevery"]
    outID = params["filename"] + "_tau_" + str(params["tau"]) + "_lambda_" + str(r.penalty)
    algo = AlgoBase(p, r, params)

    # Eval counter
    num_geval += 1
    num_ceval += 1
    num_Jeval += 1

    # Print the problem information
    helper.print_prob(x,p,r,outID)

    # Algorithm starts
    while True:
        start = time.process_time()
        # Print the header every "printevery" lines
        if iteration%printevery == 0:
            helper.print_header(outID)

        # Infeasibility check
        if np.linalg.norm(np.dot(np.transpose(J),c), ord=2) != 0:
            # Compute tangential component vk
            if normal_step_strategy == "Cauchy":
                v = algo.tr_cauchypoint(x, alpha)
            elif normal_step_strategy == "Newton":
                v = algo.tr_newton(alpha, c, J)
            elif normal_step_strategy == "NewtonCG":
                pass
        else:
            v = np.zeros(dim)
            if c.any() != 0:
                status = 2
                print("Infeasible stationary point!")
                break

        # Compute the regularization parameter lower bound
        # nrow = np.shape(J)[0]
        # JJT = np.linalg.solve(np.dot(J,np.transpose(J)), np.identity(nrow))
        # bk = x[p.p.n:p.p.n+p.p.m] + v[p.p.n:p.p.n+p.p.m]
        # term = np.dot(JJT, (1/alpha)*bk + np.dot(J, g)) + (1/alpha)*bk
        # lambda_lb = np.linalg.norm(term, ord=1)
        lambda_lb = 0

        # Compute normal component uk
        if tangential_step_strategy == "ADMM":
            list_uy = algo.admm(x, v, alpha)
        elif tangential_step_strategy == "Gurobi":
            list_uy = algo.qp_gurobi(x, v, alpha, r.penalty, J)

        # Obtain normal component and corresponding dual variable
        u = list_uy[0]
        y = list_uy[1]
        ustatus = list_uy[2]

        # Compute search direction
        s = u + v

        # Termination condition check
        if ustatus == 5 or ustatus == 12:
            status = 3
            break
        elif np.linalg.norm(s, ord=2)/alpha <= tol:
            status = 0
            break
        elif iteration >= maxit:
            status = 1
            break

        # Update merit parameter tau
        # Compute second order model of the merit function 
        Delta_qk = algo.tau_update(x, s, v, alpha, c, J)

        # Define a merit function handle
        Phi = lambda z: algo.params["tau"]*(p.obj(z) + r.obj(z)) + np.linalg.norm(p.cons(z),ord=2) 

        # Update each iterate and proximal parameter
        if Phi(x + s) - Phi(x) <= -eta*Delta_qk:
            x = x + s
            ls_judge= 1
            # alpha = alpha
        else:
            # x = x
            alpha = xi*alpha
            ls_judge= 0

        # Eval counter 
        num_feval += 2
        num_ceval += 2
        
        # Increase the iter and compute constraint function value along with its Jacobian
        iteration += 1  
        g = p.obj(x, gradient=True)[1]
        c, J = p.cons(x, gradient=True)

        # Eval counter 
        if ls_judge == 1:
            num_geval += 1
            num_ceval += 1
            num_Jeval += 1

        elapsed_time += time.process_time() - start

        # Printout information
        fval = p.obj(x)
        normx = np.linalg.norm(x, ord = 2)
        normv = np.linalg.norm(v, ord = 2)
        normu = np.linalg.norm(u, ord = 2)
        norms = np.linalg.norm(s, ord = 2)
        normc = np.linalg.norm(p.cons(x), ord = 2)
        tau   = algo.params["tau"]
        sparsity = len(np.where(x == 0)[0])
        meritf = Phi(x)
        lagrangian_multiplier = np.linalg.norm(y, ord = 2)
        # c, J = p.cons(x, gradient=True)
        rank = matrix_rank(J)
        JTc   = np.dot(np.transpose(J),c)
        condJ = np.linalg.cond(J)
        normJTc = np.linalg.norm(JTc, ord = 2)
        Lipschtiz_const = 0
        # Identify the first iteration where the sparsity is identified
        if np.linalg.norm(x[p.p.n:p.p.n+p.p.m],ord=2) == 0 and iter_identify == 0:
            iter_identify = iteration

        # Print each line
        helper.print_iteration(iteration, fval, normx, normv, normu, norms, normc, alpha,
                               tau, Lipschtiz_const, meritf, Delta_qk, condJ, normJTc, rank, sparsity, lagrangian_multiplier, ustatus, lambda_lb, outID)
    
    # Identify if the sparsity is induced
    if np.linalg.norm(x[p.p.n:p.p.n+p.p.m],ord=2) == 0 and (status == 0 or status == 1):
        judge = "Yes"

    # Output information 
    info["x"] = x
    info["objective"] = p.obj(x)
    info["constraint_violation"] = np.linalg.norm(p.cons(x), ord = 2)
    info["num_zero"] = sparsity
    info["first_iter_sparsity_idnetified"] = iter_identify
    info["sparsity_existence"] = judge
    info["inf_norm_y"] = 0
    info["prox_param"] = alpha
    info["chi_criteria"] = np.linalg.norm(s, ord = 2)/alpha
    info["inf_norm_y"] = np.linalg.norm(x[p.p.n:p.p.n+p.p.m],ord=inf)
    info["status"] = status
    info["ustatus"] = ustatus
    info["iteration"] = iteration
    info["func_eval"] = num_feval
    info["grad_eval"] = num_geval
    info["cons_eval"] = num_ceval
    info["jacb_eval"] = num_Jeval
    info["elapsed_time"] = elapsed_time

    return info