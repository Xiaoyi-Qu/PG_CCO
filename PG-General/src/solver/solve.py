'''
# File: solve.py
# Author: Xiaoyi Qu(xiq322@lehigh.edu)
# File Created: 2025-07-14 00:53
# Description: Algorithm implementation.
#              status: (1) 0: optimal.
#                      (2) 1: reach the maximum iteration.
#                      (3) 2: infeasible stationary point.
#                      (4) 3: issue with the Gurobi model.
# ----------	---	----------------------------------------------------------
'''

import sys
import numpy as np
sys.path.append("../")
sys.path.append("../..")
from src.solver.algorithm import AlgoBase_General
from src.utils.helper import setup_logger, print_header, print_iteration, projected_steepest_descent_direction


def log_final_results(info, outID=None):
    logger = setup_logger(outID)
    lines = []

    # Add status messages
    if info["ustatus"] == 5:
        lines.append("Model was proven to be unbounded.")
    elif info["ustatus"] == 12:
        lines.append("Optimization was terminated due to unrecoverable numerical difficulties.")

    lines.append("******************************************************************************")
    lines.append("Final Results")
    lines.append("Objective value (f):.....................................................%8.6e" % info["fval"])
    lines.append("Objective value (f+r):...................................................%8.6e" % info["objective"])

    # Optional: Uncomment as needed
    # lines.append("Relative error:.......................................................%8.8e" % relative_error)
    # lines.append("Proximal parameter:....................................................%8.6e" % info["prox_param"])
    # lines.append("Chi_criteria:................................................%s" % str(info["chi_criteria"]))
    # lines.append("Status:.................................................................%1d" % info["status"])
    # lines.append("Elapsed time:..........................................................%8.6es" % info["elapsed_time"])

    lines.append("x = " + str(np.array(info['x'])))

    # Write all lines to logger
    for line in lines:
        logger.info(line)

    return None


def solve(p, r, bound_constraints, x, alpha, params):
    '''
    Arguments:
        p: problem instance object
        r: regularizer object
        x: starting point
        alpha: initial proximal parameter
        params: a list that stores parameters
    '''

    info = {}
    iteration= 0
    printevery = 20
    outID = None
    tol_stationarity = params["tol_stationarity"]
    tol_feasibility = params["tol_feasibility"]
    solver = AlgoBase_General(p, r, params)

    # Print the problem information TBD
    # helper.print_prob(x,p,r,outID)

    while True:
        # Print the header every "printevery" lines
        if iteration%printevery == 0:
            print_header(outID)

        '''
        -----------------------------------------------------------------
        # Infeasibility check
        # Obtain normal direction vk by solving a trust region subproblem
        -----------------------------------------------------------------
        '''
        # c,J g = p.obj(x, gradient=True)[1] 
        c, J = p.cons(x, gradient=True)
        delta = projected_steepest_descent_direction(x, J.T@c, bound_constraints=bound_constraints)
        if np.linalg.norm(delta, ord=2) <= 1e-12 and np.linalg.norm(c, ord=2) >= 1e-2:
            status = 2
            print("Infeasible stationary point!")
            break
        elif np.linalg.norm(delta, ord=2) <= 1e-12:
            v = np.zeros_like(x)
        else:
            v, beta_v = solver.solve_tr_bound(x, alpha, np.linalg.norm(delta, ord=2), bound_constraints=bound_constraints)
            # print(v)
        
        '''
        --------------------------------------------------
        # Compute normal component uk.
        # Solve the tangential subproblem with two options.
            - Gurobi
            - HPR-QP https://arxiv.org/html/2507.02470
        --------------------------------------------------
        '''
        list_uy = solver.solve_qp_subproblem_gurobi(x, v, alpha, J, bound_constraints=bound_constraints)
        u = list_uy[0]
        y = list_uy[1]
        ustatus = list_uy[2]

        s = u.reshape(-1,1) + v

        '''
        ---------------------------------------------------------
        # Termination condition check
        #   - Gurobi error
        #   - First-order KKT point
        #   - reach max iteration
        #   - Infeasible stationary point (can be found elsewhere)
        ----------------------------------------------------------
        '''
        if ustatus == 5 or ustatus == 12:
            status = 3
            break
        elif np.linalg.norm(s, ord=2)/alpha <= tol_stationarity and np.linalg.norm(c, ord=2)<= tol_feasibility:
            status = 0
            break
        elif iteration >= params["maxit"]:
            status = 1
            break

        # Update merit parameter tau
        Delta_qk = solver.update_tau(x, s, alpha, params["tau"])

        # Define a merit function handle
        Phi = lambda z: params["tau"]*(p.obj(z) + r.obj(z)) + np.linalg.norm(p.cons(z),ord=2) 

        # Update each iterate and proximal parameter
        if Phi(x + s) - Phi(x) <= -params["eta_alpha"] * Delta_qk:
            x = x + s
            alpha = min(alpha/params['xi_alpha'], 10)
        else:
            # x = x
            alpha = params['xi_alpha']*alpha
        
        # Increase the iter and compute constraint function value along with its Jacobian
        iteration += 1  
        g = p.obj(x, gradient=True)[1]
        c, J = p.cons(x, gradient=True)

        # Printout information
        fval = p.obj(x)
        frval = p.obj(x) + r.obj(x)
        normg = np.linalg.norm(g, ord = 2)
        normx = np.linalg.norm(x, ord = 2)
        normv = np.linalg.norm(v, ord = 2)
        normu = np.linalg.norm(u, ord = 2)
        norms = np.linalg.norm(s, ord = 2)
        normc = np.linalg.norm(p.cons(x), ord = 2)
        KKTnorm = np.linalg.norm(s, ord = 2)/alpha
        tau = params["tau"]
        meritf = Phi(x)
        # sparsity = len(np.where(x == 0)[0])

        # Print each line
        # print(iteration, fval, frval, normg, normx, normv, normu, norms, normc, alpha, KKTnorm,
        #                        tau, meritf)
        print_iteration(iteration, fval, frval, normg, normx, normv, normu, norms, normc, alpha, KKTnorm,
                               tau, meritf, outID)

    # Output information 
    info["x"] = x
    info["fval"] = p.obj(x)
    info["objective"] = p.obj(x) + r.obj(x)
    info["status"] = status
    info["ustatus"] = ustatus
    
    log_final_results(info, outID=None)

    return info









# info["constraint_violation"] = np.linalg.norm(p.cons(x), ord = 2)
# info["num_zero"] = sparsity
# info["first_iter_sparsity_idnetified"] = iter_identify
# info["sparsity_existence"] = judge
# info["inf_norm_y"] = 0
# info["prox_param"] = alpha
# # info["chi_criteria"] = KKTnorm
# info["chi_criteria"] = max(np.linalg.norm(u, ord = 2)/alpha, np.linalg.norm(c, ord = 2))
# info["inf_norm_y"] = np.linalg.norm(x[p.p.n:p.p.n+p.p.m],ord=inf)
# info["status"] = status
# info["ustatus"] = ustatus
# info["iteration"] = iteration
# info["func_eval"] = num_feval
# info["grad_eval"] = num_geval
# info["cons_eval"] = num_ceval
# info["jacb_eval"] = num_Jeval
# info["elapsed_time"] = elapsed_time

# start_u = time.process_time()
# if tangential_step_strategy == "ADMM":
#     list_uy = algo.admm(x, v, alpha)
# elif tangential_step_strategy == "Gurobi":
#     list_uy = solve_qp_subproblem_gurobi(x, v, alpha, r.penalty, J)
# time_u = time.process_time() - start_u

# lagrange_multiplier = np.linalg.norm(y, ord = inf) # change 2-norm to inf-norm
# rank = matrix_rank(J)
# JTc   = np.dot(np.transpose(J),c)
# # condJ = np.linalg.cond(J)
# condJ = np.linalg.svd(J)[1][-1]
# normJTc = np.linalg.norm(JTc, ord = 2)
# Lipschtiz_const = 0
# # Identify the first iteration where the sparsity is identified
# if np.linalg.norm(x[p.p.n:p.p.n+p.p.m],ord=2) == 0 and iter_identify == 0:
#     iter_identify = iteration