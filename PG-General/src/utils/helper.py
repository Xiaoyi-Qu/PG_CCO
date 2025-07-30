'''
File: helper.py
Author: Xiaoyi Qu
File Created: 2025-07-14 00:54
--------------------------------------------
Description: Helper functions
'''  

import logging
import os
import numpy as np

def setup_logger(outID=None):
    log_dir = '/home/xiq322/PG_CCO/PG-General/test/experiments/log'
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = f"{outID}.txt" if outID is not None else "log.txt"
    log_path = os.path.join(log_dir, log_file)

    logger = logging.getLogger(str(outID))  # Create a unique logger for each outID
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_path, mode='a')
        formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def print_prob_info(p, r, x, outID):
    logger = setup_logger(outID)

    content = "==============================================================================\n"
    content += "                Proximal Gradient Method    Version: 0.1 (2025-07-20)                \n"
    content += "==============================================================================\n"
    content += f"Problem Name:...................................{p.name}\n"
    content += f"   (1) Number of variables: {p.n}\n   (2) Number of constraints: {p.m}\n"
    content += f"   (3) Number of equality constraints: {p.me}\n   (4) Number of inequality constraints: {p.mi}\n"
    content += f"   (5) Constraint violation: {np.linalg.norm(p.cons(x), ord=2)}\n"
    content += "Regularizer Type:....................................L1\n"
    content += f"Regularization Parameter:..............................Lambda={r.penalty}\n"
    content += "******************************************************************************\n"
    content += "Comments: a. |.| represents 2-norm unless specified. b. y represents the dual variable.\n"
    content += "******************************************************************************\n"

    logger.info(content)


def print_header(outID):
    logger = setup_logger(outID)
    column_titles = ' {Iter:^5s} {f:^11s} {fr:^11s} {g:^11s} {x:^11s} {v:^11s} {u:^11s} ' \
                    '{s:^11s} {c:^11s} {a:^11s} {tr_radius:^11s} {delta_qk:^11s} {alpha:^11s} {KKT:^11s} {tau:^11s} {phi:^11s} {phis:^11s} {feasibility_check:^11s} {vmethod:^11s} {vk1:^11s}'.format(
        Iter='Iter', f='f', fr='f+r', g='g', x='|x|', v='|v|', u='|u|',
        s='|s|', c='|c|', a='|a|_inf', delta_qk='delta_qk', alpha='alpha', tr_radius="tr_radius", KKT='KKT', tau='tau', phi='Merit fval', phis='Merit fval s',
        feasibility_check='x_Omega', vmethod='vmethod', vk1='vk(1)'
    )
    logger.info(column_titles)


def print_iteration(iteration, fval, frval, normg, normx, normv, normu, norms, normc, norma, tr_radius, delta_qk, alpha, KKTnorm,
                    tau, meritf, meritfs, feasibility_check, vmethod, vk1, outID):
    logger = setup_logger(outID)
    contents = "{it:5d} {fval:8.5e} {frval:8.5e} {normg:8.5e} {normx:8.5e} {normv:8.5e} " \
               "{normu:8.5e} {norms:8.5e} {normc:8.5e} {norma:8.5e} {tr_radius:8.5e} {delta_qk:8.5e} {alpha:8.5e} {KKT:8.5e} {tau:8.5e} " \
               "{meritf:8.5e} {meritfs: 8.5e} {feasibility_check: 8.5e} {vmethod:13s} {vk1: 8.5e}".format(
        it=iteration, fval=fval, frval=frval, normg=normg, normx=normx,
        normv=normv, normu=normu, norms=norms, normc=normc, norma=norma, tr_radius=tr_radius, delta_qk=delta_qk,
        alpha=alpha, KKT=KKTnorm, tau=tau, meritf=meritf, meritfs = meritfs, feasibility_check=feasibility_check, vmethod=vmethod, vk1 = vk1
    )
    logger.info(contents)
    

def projected_steepest_descent_direction(x, grad_f, bound_constraints=None):
    """
    Compute the projected steepest descent direction for bound constraints.

    Parameters:
        x: np.ndarray, current point.
        grad_f: np.ndarray, gradient at x.
        bound_constraints: tuple (lower_bounds, upper_bounds) with each a list or array.
    
    Returns:
        direction: np.ndarray, the projected steepest descent direction.
    
    Example of bound_constraints:
        bound_constraints = (
            [-np.inf, -np.inf, -np.inf, -np.inf, 0, 0],
            [ np.inf,  np.inf,  np.inf,  np.inf, np.inf, np.inf],
        )
    """
    direction = -grad_f.copy()

    if bound_constraints is not None:
        lower_bounds, upper_bounds = bound_constraints
        if len(x.shape) == 1:
            lower_bounds = np.asarray(lower_bounds)
            upper_bounds = np.asarray(upper_bounds)
        else:
            lower_bounds = np.asarray(lower_bounds).reshape(-1,1)
            upper_bounds = np.asarray(upper_bounds).reshape(-1,1)

        # Identify variables at bounds where descent direction would violate constraints
        at_lower = (x <= lower_bounds) & (grad_f > 0) 
        at_upper = (x >= upper_bounds) & (grad_f < 0)

        # Block descent in those directions
        direction[at_lower | at_upper] = 0.0

    return direction


def projection(x, bound_constraints=None):
    """
    Bound constraints is a tuple of lists.
    Example input:
        bound_constraints = (
            (-np.inf, -np.inf, -np.inf, -np.inf, 0, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
        )
    """
    x_proj = np.copy(x)

    if bound_constraints is not None:
        lower_bounds, upper_bounds = bound_constraints
        if len(x.shape) == 1:
            lower = np.asarray(lower_bounds)
            upper = np.asarray(upper_bounds)
        else:
            lower = np.asarray(lower_bounds).reshape(-1,1)
            upper = np.asarray(upper_bounds).reshape(-1,1)
        x_proj = np.minimum(np.maximum(x_proj, lower), upper)

    return x_proj