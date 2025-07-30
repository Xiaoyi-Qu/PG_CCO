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
from src.solver.params import params
from src.utils.helper import setup_logger, print_prob_info, print_header, print_iteration, projected_steepest_descent_direction
from src.utils.helper import projection

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
    lines.append(f"Objective value (f):.....................................................{info['fval']}")
    lines.append(f"Objective value (f+r):...................................................{info['objective']}")
    lines.append(f"Constraint violation (|c(x)|_2):.........................................{info['constr_violation']}")
    lines.append(f"Chi measure (|s|/alpha):.................................................{info['chi_measure']}")
    lines.append(f"Variable a is zero (Y/N):................................................{info['norma'] == 0}")
    lines.append(f"Variable a is approximate zero (Y/N):....................................{info['norma'] <= params['approximate_a']}")
    lines.append(f"KKT found (Y/N):.........................................................{info['status'] == 0}")
    lines.append(f"Status:.................................................................{info['status']}")


    # Output information 
    # info["x"] = x
    # info["fval"] = p.obj(x)
    # info["objective"] = p.obj(x) + r.obj(x)
    # info["constr_violation"] = normc
    # info["a_infty_norm"] = norma
    # info["status"] = status
    # info["ustatus"] = ustatus

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
    outID = f"{p.name}_{params['kappav_gurobi']}"
    tol_stationarity = params["tol_stationarity"]
    tol_feasibility = params["tol_feasibility"]
    solver = AlgoBase_General(p, r, params)
    vmethod = ""

    # Print the problem information
    if p.name == "SCCA":
        outID = f"SCCA_{p.nx}_{r.penalty}"
    print_prob_info(p, r, x, outID)

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
        delta_vk = projection(x-J.T@c, bound_constraints=bound_constraints) - x
        v = np.zeros_like(x)
        if np.linalg.norm(delta_vk, ord=2) <= 1e-12 and np.linalg.norm(c, ord=2) >= 1e-1:
            status = 2
            print("Infeasible stationary point!")
            break
        elif np.linalg.norm(delta_vk, ord=2) <= 1e-12:
            v = np.zeros_like(x)
        else:
            print(np.linalg.matrix_rank(J, tol=1e-10), J.shape[0])
            if params["tr_bound_solver"] == "cauchy" or (np.linalg.matrix_rank(J, tol=1e-10) < J.shape[0]):
                v, beta_v = solver.solve_tr_bound_cauchy(x, alpha, np.linalg.norm(delta, ord=2), bound_constraints=bound_constraints)
                vmethod = "cauchy"
            elif params["tr_bound_solver"] == "gurobi":
                v, v_y, vstatus = solver.solve_tr_bound_gurobi(x, alpha, np.linalg.norm(delta, ord=np.inf), bound_constraints=bound_constraints)
                vmethod = "gurobi"
                if vstatus == 100001:
                    exit() # Gurobi internal error
                    v, beta_v = solver.solve_tr_bound_cauchy(x, alpha, np.linalg.norm(delta, ord=2), bound_constraints=bound_constraints)
            else:
                raise ValueError("Unknown trust region bound solver specified.")
            # print(beta_v)
        
        JTJ = J.T @ J
        JTc = J.T @ c

        # print(f"Value = {v @ JTJ @ v + 2 * JTc @ v}")
        if (v @ JTJ @ v + 2 * JTc @ v) > 0:
            # Check the rank of matrix J
            v, beta_v = solver.solve_tr_bound_cauchy(x, alpha, np.linalg.norm(delta, ord=2), bound_constraints=bound_constraints)
            vmethod = "cauchy"
            rank = np.linalg.matrix_rank(J, tol=1e-10)
            if rank < J.shape[0]:
                print(rank, J.shape[0])
                print("The Jacobian matrix J is not full rank, using Cauchy point instead.")
                print("The quadratic model is not positive definite, using Cauchy point instead.")
                exit()
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
        # y = list_uy[1]
        ustatus = list_uy[2]

        # print(f"u value:{u}, v value:{v}")
        if p.name == "SCCA":
            s = u.reshape(-1,1) + v.reshape(-1,1)
            # s = s.reshape(-1,1)
            # s = u.reshape(-1,1) + v # scca test uncomment this.
        else:
            s = u + v

        '''
        ---------------------------------------------------------
        # Termination condition check
        #   - Gurobi error
        #   - First-order KKT point
        #   - reach max iteration
        #   - Infeasible stationary point (can be found elsewhere)
        ----------------------------------------------------------
        '''
        KKTnorm = np.linalg.norm(s, ord=2)/alpha
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
        Delta_qk = solver.update_tau(x, s, v, alpha, params["tau"])

        # Define a merit function handle
        Phi = lambda z: params["tau"]*(p.obj(z) + r.obj(z)) + np.linalg.norm(p.cons(z),ord=2) 

        # Update each iterate and proximal parameter
        Phixs = Phi(x + s)
        Phix = Phi(x)
        if Phi(x + s) - Phi(x) <= -params["eta_alpha"] * Delta_qk:
            x = x + s
            alpha = min(alpha/params['xi_alpha'], 1000)
            # alpha = min(alpha/params['xi_alpha'], 1.25)
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
        norma = np.linalg.norm(x[-p.m:], ord=np.inf)
        tr_radius = params["kappav_gurobi"] * alpha * np.linalg.norm(delta, ord=2)
        delta_qk = Delta_qk
        tau = params["tau"]
        meritf = Phix
        meritfs = Phixs
        feasibility_check = np.linalg.norm(projection(x, bound_constraints=bound_constraints) - x, ord=2)
        vk1 = np.linalg.norm(delta_vk, ord=2)
        # Print each line
        print_iteration(iteration, fval, frval, normg, normx, normv, normu, norms, normc, norma, tr_radius, delta_qk, alpha, KKTnorm,
                               tau, meritf, meritfs, feasibility_check, vmethod, vk1, outID)

    # Output information 
    info["x"] = x
    info["fval"] = p.obj(x)
    info["objective"] = p.obj(x) + r.obj(x)
    info["constr_violation"] = np.linalg.norm(p.cons(x), ord = 2)
    info["norma"] = np.linalg.norm(x[-p.m:], ord=np.inf)
    info["chi_measure"] = KKTnorm
    info["status"] = status
    info["ustatus"] = ustatus
    
    log_final_results(info, outID=outID)

    return info

