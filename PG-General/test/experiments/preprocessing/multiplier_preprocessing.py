'''
Estimate the Lagrange multiplier for the following optimization problem
    min  f(x)
    s.t. c(x)=0
# A total of 96 test problems. 14 test problems are excluded since the return status is not optimal.
'''

from pyoptsparse import PSQP, SLSQP, IPOPT, Optimization
import pycutest as pycutest
import argparse
import os
import csv

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
    optProb = Optimization(f"Problem {problem.name}", objfunc)

    # Design Variables
    # optProb.addVarGroup("xvars", problem.n, "c", value = problem.x0, lower=None, upper=None)
    optProb.addVarGroup("xvars", problem.n, "c", value = 0.1, lower=problem.bl, upper=problem.bu)
    
    # Constraints
    optProb.addConGroup("con", problem.m, lower=problem.cl, upper=problem.cu)

    # Objective
    optProb.addObj("obj")

    # Check optimization problem
    # print(optProb)

    # Optimizer
    optOptions = {"print_level": 7, "file_print_level": 7}
    opt = IPOPT(options=optOptions)

    # CSV saving logic
    output_csv = "multipliers.csv"
    file_exists = os.path.isfile(output_csv)

    # Solve
    sol = opt(optProb, sens="CD")
    print(sol.optInform['value'])
    if sol.optInform['value'] != 0:
        with open(output_csv, mode='a', newline='') as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(['Problem Name', 'Multiplier'])  # Write header only if file doesn't exist
            writer.writerow([problem.name, sol.optInform['value']])
            raise SyntaxError

    # check if the optimal solution exists

    # Check Solution
    # print(sol.lambdaStar)
    # Example usage:
    ipopt_out_file = 'IPOPT.out'
    name = problem.name
    yc_inf, yd_inf = find_first_yc_yd_inf(ipopt_out_file)
    print(f"First ||curr_y_c||_inf: {yc_inf}")
    print(f"First ||curr_y_d||_inf: {yd_inf}")

    max_multiplier = max(yc_inf, yd_inf)

    with open(output_csv, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Problem Name', 'Multiplier'])  # Write header only if file doesn't exist
        writer.writerow([name, max_multiplier])


def find_first_yc_yd_inf(ipopt_out_path):
    yc_val, yd_val = None, None

    with open(ipopt_out_path, 'r') as f:
        lines = f.readlines()

    for line in reversed(lines):
        if '||curr_y_c||_inf' in line and yc_val is None:
            parts = line.strip().split()
            for i, part in enumerate(parts):
                if '||curr_y_c||_inf' in part:
                    try:
                        yc_val = float(parts[i+2])
                    except (IndexError, ValueError):
                        pass

        if '||curr_y_d||_inf' in line and yd_val is None:
            parts = line.strip().split()
            for i, part in enumerate(parts):
                if '||curr_y_d||_inf' in part:
                    try:
                        yd_val = float(parts[i+2])
                    except (IndexError, ValueError):
                        pass

        if yc_val is not None and yd_val is not None:
            break

    return yc_val, yd_val

    
def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-n", "--name", type=str, default="HS92", help="Test problem name")

    # Parse arguments
    config = parser.parse_args()

    return config


if __name__ == "__main__":
    main(get_config())
    


    
