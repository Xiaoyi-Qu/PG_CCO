'''
# File: regularizer.py
# Project: First order method for CCO
# Description: The code is to compute the regularizer.
# ----------	---	----------------------------------------------------------
'''

import numpy as np

class L1:
    def __init__(self, xdim, penalty=None):
        self.xdim = xdim
        self.penalty = penalty

    # Compute L1-regularizer
    def obj(self, x):
        penalty = self.penalty
        mid = self.xdim
        end = len(x)
        return penalty*np.linalg.norm(x[mid:end], ord=1)