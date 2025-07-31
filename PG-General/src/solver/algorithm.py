'''
File: algorithm.py
Author: Xiaoyi Qu(xiq322@lehigh.edu)
File Created: 2025-07-14 00:41
--------------------------------------------
- A collection of subcomponents of the proposed algorithm
'''

import sys
sys.path.append("../")
sys.path.append("../..")

import numpy as np
import gurobipy as gp
from gurobipy import GRB, GurobiError
from src.utils.helper import projection

class AlgoBase_General:
    def __init__(self, p, r, params):
        self.p = p
        self.r = r
        self.params = params
    
    def solve_tr_bound_cauchy(self, x, alpha, delta, bound_constraints=None):
        '''
        - compute direction vk
        '''
        v = np.zeros_like(x)
        kappav = self.params["kappav_cauchy"]
        eta = self.params["eta_beta"]
        gamma = self.params["gamma_beta"]
        p = self.p
        c, J = p.cons(x, gradient=True)
        grad_mk_0 = J.T @ c
        beta_k = beta_init = 1

        mk = lambda v: 0.5 * np.linalg.norm(c + J @ v, ord=2)**2

        iter_monitor = 0

        while iter_monitor <= 10000:
            x_trial = x - beta_k * grad_mk_0
            x_proj = projection(x_trial, bound_constraints=bound_constraints)
            vk_beta = x_proj - x
            
            # print(kappav * alpha * delta, grad_mk_0.T @ vk_beta)
            # print(vk_beta, 0)
            # print(mk(vk_beta), mk(np.zeros_like(x)))

            if (np.linalg.norm(vk_beta, ord=2) <= kappav * alpha * delta and
                mk(vk_beta) <= mk(np.zeros_like(x)) + self.params['eta_alpha'] * grad_mk_0.T @ vk_beta):
                v = vk_beta
                break

            beta_k *= gamma

            iter_monitor += 1

        if iter_monitor == 1001:
            raise SyntaxError

        return v, beta_k


    def solve_tr_bound_gurobi(self, x, alpha, delta, bound_constraints=None):
        '''
        - compute direction vk 
        - replace \ell_2 norm with \ell_inf norm
        '''
        p = self.p
        kappav = self.params["kappav_gurobi"]
        c, J = p.cons(x, gradient=True)
        n = len(x)
        m = p.m
        box_radius = kappav * alpha * delta
        if box_radius <= 1e-16:
            box_radius = 1e-16

        # Create Gurobi model
        model = gp.Model("TR_Bound")
        # model.setParam('OutputFlag', 0)  # silence Gurobi output
        model.setParam("Method", 2)
        # model.setParam("NumericFocus", 3)
        model.setParam("DualReductions", 0)
        model.setParam('FeasibilityTol', 1e-8)

        # Decision variables v ∈ R^n
        # box_radius = kappav * alpha * delta
        v = model.addMVar(n, name="v", lb=-box_radius, ub=box_radius)

        # Objective
        # objective = gp.quicksum(
        #     (c[i] + gp.quicksum(J[i, j] * v[j] for j in range(n))) *
        #     (c[i] + gp.quicksum(J[i, j] * v[j] for j in range(n)))
        #     for i in range(m)
        # )
        # model.setObjective(objective, GRB.MINIMIZE)

        # Quadratic objective: (1/2) * || c + J v ||_2^2
        JTJ = J.T @ J
        JTc = J.T @ c

        quad_expr = v @ JTJ @ v + 2 * JTc @ v
        model.setObjective(quad_expr, GRB.MINIMIZE)

        if bound_constraints is not None:
            lower_bounds, upper_bounds = bound_constraints
            if len(x.shape) == 1:
                lower = np.asarray(lower_bounds)
                upper = np.asarray(upper_bounds)
            else:
                lower = np.asarray(lower_bounds).reshape(-1,1)
                upper = np.asarray(upper_bounds).reshape(-1,1)
            for i in range(len(lower)):
                if lower[i] != -np.inf and lower[i] != -1e+20:
                    model.addConstr(x[i] + v[i] >= lower[i], name=f"lower_bound_{i}")
                if upper[i] != np.inf and upper[i] != 1e+20:
                    model.addConstr(x[i] + v[i] <= upper[i], name=f"upper_bound_{i}")

        # Write the model to an LP file
        model.write("tr_bound.lp")

        # Optimize the model
        status = 0
        try:
            model.optimize()
        except GurobiError as e:
            if e.errno == 100001:
                print("Internal Gurobi error: ", e.message)
                status = 100001
                v = np.zeros_like(x)
                y = np.zeros(p.m + n)
                return [v,y,status]
        status = model.status

        vstore = []
        dual = []

        # Handle error from Gurobi
        if status == 12 or status == 5:
            v = np.zeros_like(x)
            y = np.zeros(p.m + n)
            return [v,y,status]
        
        # Obtain primal and dual solution
        for var in model.getVars():
            if "v" in var.VarName:
                vstore.append(var.x)
        # for var in model.getConstrs():
        #     dual.append(var.Pi)

        # Store the dual in an array
        v = np.asarray(vstore)
        # y = np.asarray(dual)
        y = np.zeros(p.m + n)

        # Explicitly dispose of the model and env
        model.dispose()

        return [v,y,status]


    def solve_qp_subproblem_gurobi(self, x, v, alpha, J, bound_constraints=None):
        '''
        Compute direction uk
        '''
        p = self.p
        r = self.r
        _, g = p.obj(x, gradient=True)
        n = len(x)
        indices = r.indices # SCCA problem
        # print(indices)

        # Create Gurobi model
        model = gp.Model("QP_L1_pq")
        model.setParam('OutputFlag', 0)  # silence Gurobi output
        
        # Set parameters
        model.setParam("Method", 1)
        # model.setParam("NumericFocus", 3)
        # model.setParam("ScaleFlag", 2)
        model.setParam("DualReductions", 0) # Enable more definitive conclusion when having INF_OR_UNBD status
        # model.setParam('FeasibilityTol', 1e-9)

        # Variables
        u = model.addMVar(n, name="u", lb=-GRB.INFINITY)
        p = model.addMVar(n, name="p", lb=0.0)  # positive part
        q = model.addMVar(n, name="q", lb=0.0)  # negative part

        # Objective
        model.setObjective(0.0001 * (gp.quicksum(g[i]*(u[i]) for i in range(n)) + 
                           (1/(2*alpha))*gp.quicksum(pow(u[i],2) for i in range(n)) +
                           r.penalty*gp.quicksum(p[i] for i in indices) + 
                           r.penalty*gp.quicksum(q[i] for i in indices)), GRB.MINIMIZE)

        # Equality constraint: Jk u = 0
        model.addConstr(J @ u == 0, name="Equality")

        # Constraints due to reformulation
        for i in range(n):
            model.addConstr(x[i] + v[i] + u[i] == p[i] - q[i], name="Reformulate")
            # model.addConstr(p[i] >= 0, name=f"Positive_{i}")
            # model.addConstr(q[i] >= 0, name=f"Negative_{i}")
        # for i in range(n-1):
        #     model.addConstr(p[i] == 1.0, name=f"p_{i}")
        #     model.addConstr(q[i] == 1.0, name=f"q_{i}")

        # Inequality constraint: xk + vk + u ∈ Ω (elementwise box constraints)
        if bound_constraints is not None:
            lower_bounds, upper_bounds = bound_constraints
            if len(x.shape) == 1:
                lower = np.asarray(lower_bounds)
                upper = np.asarray(upper_bounds)
            else:
                lower = np.asarray(lower_bounds).reshape(-1,1)
                upper = np.asarray(upper_bounds).reshape(-1,1)
            for i in range(len(lower)):
                if lower[i] != -np.inf and lower[i] != -1e+20:
                    model.addConstr(x[i] + v[i] + u[i] >= lower[i], name=f"lower_bound_{i}")
                if upper[i] != np.inf and upper[i] != 1e+20:
                    model.addConstr(x[i] + v[i] + u[i] <= upper[i], name=f"upper_bound_{i}")

        # Optimize the model
        model.optimize()
        status = model.status

        # Write the model to an LP file
        # model.write("qp_gurobi.lp")

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
            y = np.zeros_like(x) # to be changed
            return [u,y,status]
        
        print(status)
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
    
    
    # def solve_qp_subproblem_alternative(self, gk, vk, xk, alpha_k, Jk):
    #     # More general solver
    #     # For instance, a recently developed solver called HPR-QP
    #     # https://arxiv.org/pdf/2507.02470
    #     pass
    
    
    def update_tau(self, x, s, v, alpha, tau_prev):
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
        numerator = np.linalg.norm(c, ord=2) - np.linalg.norm(c + J @ v, ord=2)
        # print("Numerator:", numerator)
        if numerator < 0:
            exit()
        
        # if np.linalg.norm(c, ord=2) < 1e-12:
        #     tauk_trial = math.inf
        # else:
        #     if condition > 0: # Be careful, related to KKT residual
        #         tauk_trial = (1 - sigmac)*(norm_c - norm_cJv)/condition
        #     else:
        #         tauk_trial = math.inf

        # Compute tau_trial
        if np.linalg.norm(c, ord=2) < 1e-12 or numerator <= 0:
            tau_trial = np.inf
        else:
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
        
        Delta_qk = (self.params["tau"]/(4 *alpha)) * np.linalg.norm(s, ord=2)**2 + self.params["sigmac"] * numerator
        
        return Delta_qk
