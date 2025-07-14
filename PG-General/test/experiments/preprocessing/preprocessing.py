# Preprocess CUTEst test problem.

# import os
import pycutest
import argparse
import csv

# print(os.environ['PYCUTEST_CACHE'])
# equality constraint first

# exit()
def main(config):
    start_index = config.index
    probs = pycutest.find_problems(objective=['linear', 'quadratic', 'sum of squares', 'other'], n=[1,100], m=[1,100])
    print(len(probs))

    # Create a list to store the filtered problem info
    filtered_problems = []

    # Iterate through the first 500 problems
    for i in range(start_index, start_index + 100):
        num_eq = num_ineq = num_bound = 0
        if i == 565: # 188 382 514 633
            break
        print(i)
        try:
            p = pycutest.import_problem(probs[i])
            if 1 <= p.m < p.n and p.n <= 100: # and p.cl.any() != p.cu.any()
                for i in range(p.m):
                    if p.cl[i] == p.cu[i]:
                        num_eq += 1
                    else:
                        num_ineq += 1
                for i in range(p.n):
                    if p.bl[i] != -1e20 or p.bu[i] != 1e20:
                        num_bound += 1
                filtered_problems.append([p.name, p.n, p.m, num_eq, num_ineq, num_bound])
        except Exception as e:
            print(f"Failed to load problem {probs[i]}: {e}")

    # Write the information to a CSV file
    with open("cutest.csv", mode="a", newline="") as file:
        writer = csv.writer(file)
        # writer.writerow(["Problem Name", "Num Vars", "Num Cons", "Num Cons(Eq)", "Num Cons(In)", "Num Cons(Bound)"])
        writer.writerows(filtered_problems)

def get_config():
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-i", "--index", type=int, default=0, help="Start Index")

    # Parse arguments
    config = parser.parse_args()

    return config

# print(f"Saved {len(filtered_problems)} problems to filtered_problems.csv")
if __name__ == "__main__":
    main(get_config())