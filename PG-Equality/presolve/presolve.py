'''
Estimate the Lagrange multiplier for the following optimization problem
    min  f(x)
    s.t. c(x)=0
'''

from pyoptsparse import PSQP, SLSQP, IPOPT, Optimization
import pycutest as pycutest
import argparse
import numpy as np

def main(config):
    prob_name = config.name
    problem = pycutest.import_problem(prob_name)
    
    # begin objfunc
    def objfunc(xdict):
        x = xdict["xvars"]
        funcs = {}
        funcs["obj"] = problem.obj(x, gradient=False)
        funcs["con"] = problem.cons(x, gradient=False)
        fail = False
        
        return funcs, fail
    
    # Optimization Object
    optProb = Optimization("Problem BT8", objfunc)

    # Design Variables
    # optProb.addVarGroup("xvars", problem.n, "c", value = problem.x0, lower=None, upper=None)
    optProb.addVarGroup("xvars", problem.n, "c", value = 0.1, lower=None, upper=None)
    
    # Constraints
    optProb.addConGroup("con", problem.m, lower=0, upper=0)

    # Objective
    optProb.addObj("obj")

    # Check optimization problem
    # print(optProb)

    # Optimizer
    optOptions = {"print_level": 7}
    opt = IPOPT(options=optOptions)

    # Solve
    sol = opt(optProb, sens="CD")

    # Check Solution
    # print(sol.lambdaStar)
    
def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-n", "--name", type=str, default="HS7", help="Test problem name")

    # Parse arguments
    config = parser.parse_args()

    return config

if __name__ == "__main__":
    main(get_config())
    
    
# prob_name = "ELEC"
# problem = pycutest.import_problem(prob_name)

# # begin objfunc
# def objfunc(xdict):
#     x = xdict["xvars"]
#     funcs = {}
#     funcs["obj"] = problem.obj(x, gradient=False)
#     funcs["con"] = problem.cons(x, gradient=False)
#     fail = False

#     return funcs, fail

# # Optimization Object
# optProb = Optimization("Problem BT8", objfunc)

# # Design Variables
# optProb.addVarGroup("xvars", problem.n, "c", value = problem.x0, lower=None, upper=None)

# # Constraints
# optProb.addConGroup("con", problem.m, lower=0, upper=0)

# # Objective
# optProb.addObj("obj")

# # Check optimization problem
# # print(optProb)

# # Optimizer
# optOptions = {"print_level": 7}
# # opt = PSQP(options=optOptions)
# opt = IPOPT(options=optOptions)
# # opt = NLPQLP(options=optOptions)

# # Solve
# sol = opt(optProb, sens="CD")

# # Check Solution
# print(sol.lambdaStar)

# # begin objfunc
# def objfunc(xdict):
#     x = xdict["xvars"]
#     funcs = {}
#     funcs["obj"] = problem.obj(x, gradient=False)
#     funcs["con"] = problem.cons(x, gradient=False)
#     fail = False
    
#     return funcs, fail

# /home/xiq322/miniconda3/bin/python presolve.py --name MSS1
# /home/xiq322/miniconda3/bin/python presolve.py --name ELEC  
# /home/xiq322/miniconda3/bin/python presolve.py --name SPIN2OP  
# /home/xiq322/miniconda3/bin/python presolve.py --name LCH  
# /home/xiq322/miniconda3/bin/python presolve.py --name CHAIN  
# /home/xiq322/miniconda3/bin/python presolve.py --name MSS2 
# /home/xiq322/miniconda3/bin/python presolve.py --name EXTROSNBNE  
# /home/xiq322/miniconda3/bin/python presolve.py --name SPINOP

    
