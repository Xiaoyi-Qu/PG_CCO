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
    
    def solve_tr_bound(self, x, alpha, delta, bound_constraints=None):
        '''
        # compute direction vk
        '''
        v = np.zeros_like(x)
        kappav = self.params["kappav"]
        eta = self.params["eta_beta"]
        gamma = self.params["gamma_beta"]
        p = self.p
        c, J = p.cons(x, gradient=True)
        grad_mk_0 = J.T @ c  # ∇mk(0) = Jk.T @ ck
        beta_k = beta_init = 1

        mk = lambda v: 0.5 * np.linalg.norm(c + J @ v, ord=2)**2 # check this line

        while True:
            x_trial = x - beta_k * grad_mk_0
            x_proj = projection(x_trial, bound_constraints=bound_constraints)  # Projection onto ℝ^n_≥0
            vk_beta = x_proj - x

            if (np.linalg.norm(vk_beta, ord=2) <= kappav * alpha * delta and
                mk(vk_beta) <= mk(np.zeros_like(x)) + self.params['eta_alpha'] * grad_mk_0.T @ vk_beta):
                v = vk_beta
                break

            beta_k *= gamma

        return v, beta_k


    def solve_qp_subproblem_gurobi(self, x, v, alpha, J, bound_constraints=None):
        '''
        Compute direction uk
        '''
        p = self.p
        r = self.r
        _, g = p.obj(x, gradient=True)
        n = len(x)
        indices = r.indices # SCCA problem

        # Create Gurobi model
        model = gp.Model("QP_L1_pq")
        model.setParam('OutputFlag', 0)  # silence Gurobi output
        
        # Set parameters
        model.setParam("Method", 0)
        # model.setParam("NumericFocus", 3)
        # model.setParam("ScaleFlag", 2)
        # model.setParam("DualReductions", 0) # Enable more definitive conclusion when having INF_OR_UNBD status

        # Variables
        u = model.addMVar(n, name="u", lb=-GRB.INFINITY)
        p = model.addMVar(n, name="p", lb=0.0)  # positive part
        q = model.addMVar(n, name="q", lb=0.0)  # negative part

        # Objective
        model.setObjective(gp.quicksum(g[i]*(u[i]) for i in range(n)) + 
                           (1/(2*alpha))*gp.quicksum(pow(u[i],2) for i in range(n)) +
                           r.penalty*gp.quicksum(p[i] for i in indices) + 
                           r.penalty*gp.quicksum(q[i] for i in indices), GRB.MINIMIZE)

        # Equality constraint: Jk u = 0
        model.addConstr(J @ u == 0, name="Equality")

        # Constraints due to reformulation
        for i in range(n):
            model.addConstr(x[i] + v[i] + u[i] == p[i] - q[i], name="Reformulate")

        # Inequality constraint: xk + vk + u ∈ Ω (elementwise box constraints)
        if bound_constraints is not None:
            lower_bounds, upper_bounds = bound_constraints
            lower = np.asarray(lower_bounds).reshape(-1,1)
            upper = np.asarray(upper_bounds).reshape(-1,1)
            for i in range(len(lower)):
                if lower[i] != -np.inf:
                    model.addConstr(x[i] + v[i] + u[i] >= lower[i], name=f"lower_bound_{i}")
                if upper[i] != np.inf:
                    model.addConstr(x[i] + v[i] + u[i] <= upper[i], name=f"upper_bound_{i}")

        # Optimize the model
        model.optimize()
        status = model.status

        ustore = []
        pstore = []
        qstore = []
        dual = []
        u = np.zeros(n)
        # y = np.zeros(m+n+h) # incorrect modify this.
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
        # for j in range(m+n+2):
        #     y[j] = dual[j]
        y = np.asarray(dual)

        return [u,y,status]
    
    
    def solve_qp_subproblem_alternative(self, gk, vk, xk, alpha_k, Jk):
        # more general solver
        pass
    
    
    def update_tau(self, x, s, alpha, tau_prev):
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
        denominator = (g.T @ s + (0.5 * alpha) * np.linalg.norm(s, ord=2)**2 + rs - rx).item()

        # Compute numerator
        numerator = np.linalg.norm(c, ord=2) - np.linalg.norm(c + J @ s, ord=2)

        # Compute tau_trial
        if denominator <= 0:
            tau_trial = np.inf
        else:
            tau_trial = (1 - sigmac) * numerator / denominator

        # Update tau
        if tau_prev <= tau_trial:
            tau = tau_prev
        else:
            tau = min((1 - epsilon_tau) * tau_prev, tau_trial)

        self.params["tau"] = tau
        
        Delta_qk = (self.params["tau"]/(2 *alpha)) * np.linalg.norm(s, ord=2)**2 + self.params["sigmac"] * numerator
        
        return Delta_qk
