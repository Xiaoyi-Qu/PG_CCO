import pycutest as pycutest
import numpy as np


# probs = pycutest.find_problems()
# sss = len(probs) 1528 in total.

pycutest.print_available_sif_params("CHAIN")

# Determine number of variables
# for i in range(377, 378):
#     print(i)
#     p = pycutest.import_problem(probs[i])
#     n = p.n
#     print(p.n)
#     n_cons = p.m
#     n_cone = 0
#     for j in range(n_cons):
#         if p.cl[j] == p.cu[j]:
#             n_cone += 1
#     if n_cons == 0:
#         pass
#     elif n_cone == n_cons and p.n > p.m:
#         temp1 = -1e20*np.ones(n)
#         temp2 = 1e20*np.ones(n)
#         if np.array_equal(p.bl, temp1) and np.array_equal(p.bu, temp2):
#             print('No bound constraints: {}, m: {}, n: {}'.format(p.name, p.m, p.n))
            