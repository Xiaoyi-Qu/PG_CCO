'''
# File: regularizer.py
# Project: First order method for CCO
# Description: The code is to compute the regularizer.
# ----------	---	----------------------------------------------------------
'''

import numpy as np

class L1:
    def __init__(self, penalty=None):
        self.penalty = penalty

    # Compute L1-regularizer
    def obj(self, x):
        penalty = self.penalty
        return penalty*np.linalg.norm(x, ord=1)