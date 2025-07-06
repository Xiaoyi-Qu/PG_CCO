"""
Computes the L1 regularizer for selected indices of a decision variable vector.
"""

import numpy as np

class L1:
    def __init__(self, indices=[], penalty=None):
        """
        Initialize the L1 regularizer.

        Parameters:
        - indices (list or array): Indices (relative to x) to apply L1 penalty on.
        - penalty (float): Regularization strength.
        """
        self.indices = indices
        self.penalty = penalty if penalty is not None else 1.0

    def obj(self, x):
        """
        Compute the L1 regularization term.

        Parameters:
        - x (np.ndarray): The full decision variable vector.

        Returns:
        - float: The L1 regularization value.
        """
 
        selected = x[self.indices] 
        return self.penalty * np.linalg.norm(selected, ord=1)