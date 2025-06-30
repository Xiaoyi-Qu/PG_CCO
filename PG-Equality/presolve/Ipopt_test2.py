'''
Solve the following optimization problem
    min  f(x) + \lambda \|x\|_1
    s.t. c(x)=0
Consider introducing artificial variables
    min  f(x) + \lambda (y + z)
    s.t. c(x)=0 
         x = y - z
         y \geq 0 
         z \geq 0
'''

from pyoptsparse import IPOPT, Optimization
import pycutest as pycutest
import argparse
import numpy as np
from csv import DictReader
import os
import glob

def main(config):
    prob_name = config.name
    problem = pycutest.import_problem(prob_name)
    global reg_param
    # with open("Lagrange_multiplier.csv", 'r') as f1:
    #         dict_reader1 = DictReader(f1)
    #         prob_list = list(dict_reader1)
    #         for prob in prob_list:
    #             if prob["\ufeffName"] == prob_name:
    #                 reg_param = round(np.float64(prob["Estimate"])) + 10
    reg_param = 1
    
    # begin objfunc
    def objfunc(xdict):
        x = xdict["xvars"]
        y = xdict["yvars"]
        z = xdict["zvars"]
        funcs = {}
        funcs["obj"] = problem.obj(x, gradient=False) + reg_param * (np.sum(y) + np.sum(z))
        funcs["con1"] = problem.cons(x, gradient=False)
        funcs["con2"] = x - y + z #x = y-z
        funcs["con3"] = y 
        funcs["con4"] = z
        fail = False
        
        return funcs, fail
    
    # Optimization Object
    optProb = Optimization("Problem", objfunc)

    # Design Variables # np.random.normal(0,1,problem.n)
    optProb.addVarGroup("xvars", problem.n, "c", value = problem.x0, lower=None, upper=None)
    optProb.addVarGroup("yvars", problem.n, "c", value = 1, lower=0.0, upper=None)
    optProb.addVarGroup("zvars", problem.n, "c", value = 1, lower=0.0, upper=None)
    
    # Constraints
    optProb.addConGroup("con1", problem.m, lower=0, upper=0)
    optProb.addConGroup("con2", problem.n, lower=0, upper=0)
    # optProb.addConGroup("con3", problem.n, lower=0, upper=None)
    # optProb.addConGroup("con4", problem.n, lower=0, upper=None)
    

    # Objective
    optProb.addObj("obj")

    # Check optimization problem
    # print(optProb)

    # Optimizer
    optOptions = {"print_level": 5, "max_iter": 1000}
    opt = IPOPT(options=optOptions)

    # Solve
    sol = opt(optProb, sens="CD")

    # Walk through the directory
    file_path = './output/test.out'
    
    with open(file_path, 'r') as lines:
        for line in lines:
            if 'Total seconds in IPOPT' in line:
                parts = line.split('=')
                cpu_time = parts[1]
            if 'Constraint violation' in line:
                parts = line.split('   ')
                constr_vio = parts[1]
            if 'EXIT' in line:
                parts = line.split(': ')
                status = parts[1]
    
    # Delete everything in the file
    with open(file_path, 'w') as file:
        pass

    # Check Solution
    num_zero = len(np.where(sol.xStar['yvars'] - sol.xStar['zvars'] == 0)[0])
    print(num_zero)
    print(sol.xStar['yvars'] - sol.xStar['zvars'])
    if prob_name == 'BT1':
        column_titles = '{Prob_name:^8s} {f:^10s} {numzero:^10s} {a:^10s} {constr_vio:^10s} {cpu_time:^10s} {status:^10s} \n'.format(Prob_name='Name',
                        f='f', a='|a|', numzero='numzero', constr_vio='constr_vio', cpu_time='cpu_time', status = 'status')
    column_values = '{Prob_name:^5s} {f:^8.5e} {numzero:^8d} {a:^8.5e} {constr_vio:^8.5e} {cpu_time:^8s} {status:^10s}'.format(Prob_name=prob_name,
                    f=sol.fStar, numzero=num_zero, a=np.linalg.norm(sol.xStar['yvars'] - sol.xStar['zvars']), constr_vio=float(constr_vio), cpu_time=cpu_time, status=status)
    # column_values = '{Prob_name:^5s} {f:^8.5e} {a:^8.5e}'.format(Prob_name=prob_name,
    #                 f=sol.fStar, a=np.linalg.norm(sol.xStar['yvars'] - sol.xStar['zvars']))
    with open("output_info.txt", "a") as logfile:
        if prob_name == 'BT1':
            logfile.write(column_titles)
        logfile.write(column_values)

def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-n", "--name", type=str, default="EXTROSNBNE", help="Test problem name")

    # Parse arguments
    config = parser.parse_args()

    return config

if __name__ == "__main__":
    main(get_config())

