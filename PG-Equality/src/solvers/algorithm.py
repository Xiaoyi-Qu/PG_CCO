'''
# File: regularizer.py
# Project: First order method for CCO
# Description: The code is to implement the algorithm.
# ----------	---	----------------------------------------------------------
'''

import sys
sys.path.append("../")
sys.path.append("../..")
import math
import numpy as np
import gurobipy as gp
from numpy.linalg import matrix_rank
from gurobipy import GRB
from src.solvers.helper import nullspace
from src.funcs.regularizer import L1

class AlgoBase:
    def __init__(self, p, r, params):
        self.p = p
        self.r = r
        self.params = params

    '''
    # Compute the normal component vk
    # The trust region subproblem is written as 
            min  1/2*\|ck + Jk*v\|_2^2
            s.t. \|v\|_2 <= kappav*alphak*\|Jk^T*ck\|_2
    # which can be written as
            min  (Jk'*ck)^T v + v^T (Jk^T Jk) v 
            s.t. \|v\|_2 <= radius
    # Note that g = Jk*ck and H = Jk^T Jk.
    # The following approach is to find a Cauchy point.
    # There is a bug for LINSPANH test problem. Function value increases.
    '''
    def tr_cauchypoint(self, x, alpha):
        kappav = self.params["kappav"]
        p = self.p
        
        # Preparation
        c, J = p.cons(x, gradient=True)
        g = np.dot(np.transpose(J), c)
        H = np.matmul(np.transpose(J), J)
        gHg = np.dot(g, np.dot(H,g))
        norm_g = np.linalg.norm(g, ord=2)
        radius = kappav*alpha*norm_g
        
        # msk = @(s) f + g'*s + 0.5*s'*H*s;
        # Check if the Cauchy point is on the boundary of the trust region
        if gHg > 0:
            value1 = pow(norm_g,2)/gHg
            value2 = radius/norm_g
            stepsize = min(value1, value2)
        else:
            stepsize = radius/norm_g

        return -stepsize*g
    
    '''
    # Compute the normal component vk
    # The trust region subproblem is written as 
            min  1/2*\|ck + Jk*v\|_2^2
            s.t. \|v\|_2 <= kappav*alphak*\|Jk^T*ck\|_2
    # which can be written as
            min  (Jk*ck)^T v + v^T (Jk^T Jk) v 
            s.t. \|v\|_2 <= radius
    # Note that g = Jk*ck and H = Jk^T Jk.
    # The following approach is a newton type method.
    # Note that this method is restrictive in the sense that J is a matrix with full row rank.
    '''
    def tr_newton(self, alpha, c, J):
        kappav = self.params["kappav"]
        p = self.p
        
        # Preparation
        # c, J = p.cons(x, gradient=True)
        nrow = np.shape(J)[0]
        JTc = np.dot(np.transpose(J), c)
        JJT = np.linalg.solve(np.dot(J,np.transpose(J)), np.identity(nrow))
        radius = kappav*alpha*np.linalg.norm(JTc, ord=2)
        vn = np.dot(np.transpose(J), np.dot(JJT,c))

        stepsize = min(1, radius/np.linalg.norm(vn, ord=2))

        return -stepsize*vn

    '''
    # Compute the tangential component uk
    # The linearly constrained quadratic subproblem is written as
            min  g^T Z w + 1/(2*alphak)*norm(w)^2 + r(xk + vk + u) 
            s.t. u = Zw
    # where A = null(J). J: size = (m,n)
    # The following approach is called "Alternating Direction Multiplier Method"
    # Comments: There is a bug in this code, not include penalty parameter.
    '''
    def admm(self, x, v, alpha):
        # Obtain the null space of J
        p = self.p
        ADMM_tol = self.params["ADMM_tol"]
        ADMM_maxit = self.params["max_iter_admm"]
        f, g = p.obj(x, gradient=True)
        c, J = p.cons(x, gradient=True)
        Z = nullspace(J) # dim = (n,n-m)
        n = J.shape[1] # number of columns of matrix J
        rankZ = matrix_rank(Z) # m = matrix_rank(J)

        # Initialization
        # There is a bug here, not considering the case that J is not full rank.
        # need to fix it tonight.
        w = np.zeros(rankZ) # w = np.zeros(n-m) # dim = col-row rank
        u = np.zeros(n) # dim = col
        y = np.zeros(n) # dim = col
        rho = 5001
        status = 0
        iter = 0

        while True:
            # Update w
            I = np.eye(rankZ)
            ZTZ = np.dot(np.transpose(Z),Z)
            ZTu = np.dot(np.transpose(Z),u)
            ZTy = np.dot(np.transpose(Z),y)
            ZTg = np.dot(np.transpose(Z),g)
            w = np.dot(np.linalg.inv(1/alpha*I + rho*ZTZ),(rho*ZTu - ZTy - ZTg))

            # Update u
            Zw = np.dot(Z,w)
            u_temp = u # u_temp for termination check
            for i in range(n):
                b = (x + v)[i]
                c = (Zw + (1/rho)*y)[i]
                if (b+c) >= 1/rho:
                    u[i] = c - 1/rho
                elif (b+c) <= -1/rho:
                    u[i] = c + 1/rho
                else:
                    u[i] = -b
            
            # Update Lagrangian multiplier
            y = y + rho*(Zw - u)

            # Increment the iterate
            iter = iter+1

            # Termination condition
            primal_residual = np.linalg.norm(Zw - u)
            dual_residual = np.linalg.norm(rho*np.dot(np.transpose(Z), u_temp - u))
            if primal_residual <= ADMM_tol and dual_residual <= ADMM_tol:
                status = 0
                break
            elif iter >= ADMM_maxit:
                status = 1
                break

        return [u,y,status]


    '''
    # Compute the tangential component uk
    # The linearly constrained quadratic subproblem is written as
            min  g^T u + 1/(2*alphak)*norm(u)^2 + penalty*(\sum{p} + \sum{q}) 
            s.t. Jk*u = 0; xk + vk + u = p - q; p,q >= 0.
    # Solver only for L1 norm.
    # The above quadratic problem is solved using Gurobi solver.
    # Comments: Proper model formulation matters.
    # Bug in Gurobi, forget the coefficient of l1 norm.
    '''
    def qp_gurobi(self, x, v, alpha, penalty, J):
        # Preparation
        p = self.p
        g = p.obj(x, gradient=True)[1]
        # print(np.linalg.norm(g,ord=2))
        # J = p.cons(x, gradient=True)[1]
        m, n = J.shape # m is the number of row, n is the number of column

        # Create the model
        model = gp.Model()

        # Set parameters
        model.setParam("Method", 0)
        model.setParam("NumericFocus", 3)
        model.setParam("ScaleFlag", 2)
        model.setParam("DualReductions", 0) # Enable more definitive conclusion when having INF_OR_UNBD status
        # model.setParam("InfUnbdInfo", 1)

        # Add variables
        u = model.addVars(n, lb = -GRB.INFINITY, vtype=GRB.CONTINUOUS, name="u")
        p = model.addVars(n, lb = np.zeros(n), vtype=GRB.CONTINUOUS, name="p")
        q = model.addVars(n, lb = np.zeros(n), vtype=GRB.CONTINUOUS, name="q")
        # p = model.addVars(m, lb = np.zeros(m), vtype=GRB.CONTINUOUS, name="p")
        # q = model.addVars(m, lb = np.zeros(m), vtype=GRB.CONTINUOUS, name="q")

        # Objective function
        # option to change the model.
        model.setObjective(gp.quicksum(g[i]*(u[i]) for i in range(n)) + 
                           (1/(2*alpha))*gp.quicksum(pow(u[i],2) for i in range(n)) +
                           penalty*gp.quicksum(p[i] for i in range(n-m,n)) + 
                           penalty*gp.quicksum(q[i] for i in range(n-m,n)), GRB.MINIMIZE) 
        # model.setObjective(gp.quicksum(g[i]*(u[i]) for i in range(n)) + 
        #                    (1/(2*alpha))*gp.quicksum(pow(u[i],2) for i in range(n)) +
        #                    penalty*gp.quicksum(p[i] for i in range(m)) + 
        #                    penalty*gp.quicksum(q[i] for i in range(m)), GRB.MINIMIZE)

        # Add constraints
        for i in range(m):
            model.addConstr(gp.quicksum(J[i][j]*(u[j]) for j in range(n)) == 0, name="Linear")
        for i in range(n):
            model.addConstr(x[i] + v[i] + u[i] == p[i] - q[i], name="Reformulate")
        # for i in range(m):
        #     model.addConstr(x[i+n-m] + v[i+n-m] + u[i+n-m] == p[i] - q[i], name="Reformulate")
        # for i in range(n-m,n):
        #     model.addConstr(x[i] + v[i] + u[i] == p[i] - q[i], name="Reformulate")

        # Optimize the model
        model.optimize()
        status = model.status

        ustore = []
        pstore = []
        qstore = []
        dual = []
        u = np.zeros(n)
        y = np.zeros(m+n)
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
        # for i in range(n-m):
        #     u[i] = ustore[i]
        # for i in range(n-m,n):
        #     u[i] = pstore[i-(n-m)] - qstore[i-(n-m)] - x[i] - v[i]
        for i in range(n):
            u[i] = pstore[i] - qstore[i] - x[i] - v[i] # what I do currently (0)
        for j in range(m+n):
            y[j] = dual[j]
        # for j in range(m+m):
        #     y[j] = dual[j]

        return [u,y,status]
    
    # maybe reformulate this Gurobi model with less constraints.
    
    '''
    # Update the merit parameter tau
    # Compute the second order model
    '''
    def tau_update(self, x, s, v, alpha, c, J):
        p = self.p
        r = self.r
        sigmau = self.params["sigmau"]
        sigmac = self.params["sigmac"]
        tau = self.params["tau"]
        epsilontau = self.params["epsilontau"]

        g = p.obj(x, gradient=True)[1]
        # c, J = p.cons(x, gradient=True)
        norm_s = np.linalg.norm(s, ord=2)
        norm_c = np.linalg.norm(c, ord=2)
        norm_cJv = np.linalg.norm(c + np.dot(J,v), ord=2)  # use np.dot(J,s) or np.dot(J,v)?
        rx = r.obj(x)
        rs = r.obj(x+s)

        # Set parameter tauk_trial
        # np.sqrt is not square of a value, use pow instead.  
        condition = np.dot(g,s) + sigmau*pow(norm_s,2)/alpha + rs - rx
        if norm_c < 1e-12:
            tauk_trial = math.inf
        else:
            if condition > 0: # Be careful, related to KKT residual
                tauk_trial = (1 - sigmac)*(norm_c - norm_cJv)/condition
            else:
                tauk_trial = math.inf

        # Set parameter tauk
        if tau <= tauk_trial:
            pass 
        else:
            self.params["tau"] = min((1 - epsilontau)*tau, tauk_trial)

        # Compute the reduction of the merit function's second order model
        Delta_qk = -self.params["tau"]*condition + (norm_c - norm_cJv)

        return Delta_qk
