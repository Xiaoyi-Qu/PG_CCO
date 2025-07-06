'''
Main algorithm
'''

import sys
sys.path.append("../")
sys.path.append("../..")

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.utils.projection import ProjectionOperator

class PG_General:
    def __init__(self, p, r, params):
        self.p = p
        self.r = r
        self.params = params
        self.proj = ProjectionOperator(set_type=params["set_type"])
    
    
    def solve_tr_bound(self, xk, alpha_k):
        '''
        # compute direction vk
        '''
        kappav = self.params["kappav"]
        eta = self.params["eta_beta"]
        gamma = self.params["gamma_beta"]
        p = self.p
        c, J = p.cons(xk, gradient=True)
        grad_mk_0 = Jk @ ck  # ∇mk(0) = Jk @ ck
        beta_k = beta_init = 1

        mk = lambda v: 0.5 * np.dot(ck + Jk @ v, ck + Jk @ v)

        while True:
            x_trial = xk - beta_k * grad_mk_0
            x_proj = self.proj.project(x_trial)  # Projection onto ℝ^n_≥0
            vk_beta = x_proj - xk

            if (np.linalg.norm(vk_beta) <= kappav * alpha_k * delta_k and
                mk(vk_beta) <= mk(np.zeros_like(xk)) + eta2 * grad_mk_0.T @ vk_beta):
                break

            beta_k *= gamma

        return vk_beta, beta_k


    def solve_qp_subproblem_gurobi(gk, vk, xk, alpha_k, Jk):
        '''
        Compute direction uk
        '''
        n = len(gk)
        m = Jk.shape[0]

        # Create Gurobi model
        model = gp.Model("QP_L1_pq")
        model.setParam('OutputFlag', 0)  # silence Gurobi output

        # Variables
        u = model.addMVar(n, name="u", lb=-GRB.INFINITY)
        p = model.addMVar(n, name="p", lb=0.0)  # positive part
        q = model.addMVar(n, name="q", lb=0.0)  # negative part

        # Objective: quadratic + linear + L1 as sum(p + q)
        Q = np.eye(n) / (2 * alpha_k)
        lin_obj = gk + vk / alpha_k
        model.setObjective(
            lin_obj @ u + 0.5 * u @ Q @ u + (p + q).sum(),
            GRB.MINIMIZE
        )

        # Equality constraint: Jk u = 0
        model.addConstr(Jk @ u == 0, name="equality")

        # Define shifted variable: xk + vk + u = p - q
        shifted = xk + vk
        model.addConstr(p - q == shifted + u, name="pq_split")

        # Inequality constraint: xk + vk + u >= 0  →  p - q >= 0
        model.addConstr(p - q >= 0, name="nonneg")

        # Solve
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return u.X
        else:
            raise ValueError("Gurobi did not find an optimal solution.")
    
    
    def solve_qp_subproblem_alternative(gk, vk, xk, alpha_k, Jk):
        # more general solver
        pass
    
    
    def update_tau(tau_prev, xk, sk, alpha_k):
        """
        Update rule for the merit parameter tau_k as described in Equation (3.4).
        """
        sigmac = self.params["sigmac"]
        epsilon_tau = self.params["epsilon_tau"]
        gk = p.obj(x, gradient=True)[1]
        ck, Jk = p.cons(x, gradient=True)
        
        # Compute A_k
        Ak = gk @ sk + (0.5 / alpha_k) * np.linalg.norm(sk)**2 + r_new - rk

        # Compute constraint violation change
        ck_norm = np.linalg.norm(ck)
        cksk_norm = np.linalg.norm(ck + Jk @ sk)

        # Compute tau_k trial
        if Ak <= 0:
            tau_trial = np.inf
        else:
            tau_trial = (1 - sigma_c) * (ck_norm - cksk_norm) / Ak

        # Update tau_k
        if tau_prev <= tau_trial:
            tau_k = tau_prev
        else:
            tau_k = min((1 - epsilon_tau) * tau_prev, tau_trial)

        return tau_k
