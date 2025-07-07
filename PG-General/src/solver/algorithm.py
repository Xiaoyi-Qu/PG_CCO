'''
Main algorithm
'''

import sys
sys.path.append("../")
sys.path.append("../..")

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from src.utils.projection import projection

class AlgoBase_General:
    def __init__(self, p, r, params):
        self.p = p
        self.r = r
        self.params = params
    
    def solve_tr_bound(self, x, alpha, delta, set_type):
        '''
        # compute direction vk
        '''
        kappav = self.params["kappav"]
        eta = self.params["eta_beta"]
        gamma = self.params["gamma_beta"]
        p = self.p
        c, J = p.cons(x, gradient=True)
        grad_mk_0 = J.T @ c  # ∇mk(0) = Jk.T @ ck
        beta_k = beta_init = 1

        mk = lambda v: 0.5 * np.dot(c + J @ v, c + J @ v)

        while True:
            x_trial = x - beta_k * grad_mk_0
            x_proj = projection(x_trial, set_type)  # Projection onto ℝ^n_≥0
            vk_beta = x_proj - xk

            if (np.linalg.norm(vk_beta) <= kappav * alpha * delta and
                mk(vk_beta) <= mk(np.zeros_like(xk)) + eta2 * grad_mk_0.T @ vk_beta):
                break

            beta_k *= gamma

        return vk_beta, beta_k


    def solve_qp_subproblem_gurobi(g, x, v, alpha, set_type):
        '''
        Compute direction uk
        '''
        g = p.obj(x, gradient=True)
        _, J = p.cons(x, gradient=True)
        n = len(g)
        m = J.shape[0] # SCCA problem

        # Create Gurobi model
        model = gp.Model("QP_L1_pq")
        model.setParam('OutputFlag', 0)  # silence Gurobi output

        # Variables
        u = model.addMVar(n, name="u", lb=-GRB.INFINITY)
        p = model.addMVar(m, name="p", lb=0.0)  # positive part
        q = model.addMVar(m, name="q", lb=0.0)  # negative part

        # Objective
        model.setObjective(gp.quicksum(g[i]*(u[i]) for i in range(n)) + 
                           (1/(2*alpha))*gp.quicksum(pow(u[i],2) for i in range(n)) +
                           penalty*gp.quicksum(p[i] for i in range(m)) + 
                           penalty*gp.quicksum(q[i] for i in range(m)), GRB.MINIMIZE)

        # Equality constraint: Jk u = 0
        model.addConstr(J @ u == 0, name="Equality")

        # Constraints due to reformulation
        for i in range(n):
            model.addConstr(x[i] + v[i] + u[i] == p[i] - q[i], name="Reformulate")

        # Inequality constraint: xk + vk + u ∈ Ω (elementwise box constraints)
        for (lower_bounds, upper_bounds), indices in bound_constraints.items():
            h += len(indices)
            lower = np.array(lower_bounds)
            upper = np.array(upper_bounds)
            for i in indices:
                if lower[0] != -np.inf or lower[0] != np.inf:
                    model.addConstr(x[i] + v[i] + u[i] >= lower[0], name=f"lower_bound_{i}")
                if upper[0] != -np.inf or upper[0] != np.inf:
                    model.addConstr(x[i] + v[i] + u[i] <= upper[0], name=f"upper_bound_{i}")

        # Optimize the model
        model.optimize()
        status = model.status

        ustore = []
        pstore = []
        qstore = []
        dual = []
        u = np.zeros(n)
        y = np.zeros(m+n+h) # incorrect
        # y = np.zeros(m+m)
        
        # print information regarding the model 
        # model.printQuality()

        # Handle error from Gurobi
        if status == 12 or status == 5:
            return [u,y,status]
        
        # Obtain primal and dual solution
        for var in model.getVars():
            if "u" in var.VarName:
                ustore.append(var.x)
            if "p" in var.VarName:
                pstore.append(var.x)
            if "q" in var.VarName:
                qstore.append(var.x)
        for var in model.getConstrs():
            dual.append(var.Pi)

        # Store the solution in an array
        for i in range(n):
            u[i] = pstore[i] - qstore[i] - x[i] - v[i]
        for j in range(m+n+2):
            y[j] = dual[j]

        return [u,y,status]
    
    
    def solve_qp_subproblem_alternative(gk, vk, xk, alpha_k, Jk):
        # more general solver
        pass
    
    
    def update_tau(x, s, alpha, tau_prev):
        """
        Update rule for the merit parameter tau as described in Equation (3.4).
        """
        sigmac = self.params["sigmac"]
        epsilon_tau = self.params["epsilon_tau"]
        g = self.p.obj(x, gradient=True)[1]
        c, J = self.p.cons(x, gradient=True)
        rs = self.r.obj(x+s)
        rx = self.r.obj(x)
        
        # Compute denominator
        denominator = g.T @ s + (0.5 * alpha) * np.linalg.norm(s, ord=2)**2 + rs - rx

        # Compute numerator
        numerator = np.linalg.norm(c, ord=2) - np.linalg.norm(c + J @ s, ord=2)

        # Compute tau_trial
        if demon <= 0:
            tau_trial = np.inf
        else:
            tau_trial = (1 - sigma_c) * numerator / denominator

        # Update tau
        if tau_prev <= tau_trial:
            tau = tau_prev
        else:
            tau = min((1 - epsilon_tau) * tau_prev, tau_trial)

        params["tau"] = tau
        
        Delta_qk = (params["tau"]/(2 *alpha)) * np.linalg.norm(s, ord=2)**2 + params["sigmac"] * numerator
        
        return Delta_qk
