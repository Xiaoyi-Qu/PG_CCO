'''
Helper function.
    Compute projected steepest descent direction
'''

import numpy as np

def projected_steepest_descent_direction(x, grad_f, bound_constraints=None):
    """
    Compute the projected steepest descent direction for bound constraints.

    Example of bound constraints input
        bound_constraints = {
            ((0.0, 0.0), (2.0, 2.0)): [0, 1],
            ((1.0, 1.0), (3.0, 3.0)): [2, 3],
            ((-np.inf,), (np.inf,)): [4],
        }
    """
    direction = -grad_f.copy()

    for (lower_bounds, upper_bounds), indices in bound_constraints.items():
        indices = np.asarray(indices)
        x_sub = x[indices]
        grad_sub = grad_f[indices]

        # At lower bounds and gradient wants to increase → set direction to 0
        mask_lower_blocked = (x_sub <= lower_bounds) & (grad_sub > 0)

        # At upper bounds and gradient wants to decrease → set direction to 0
        mask_upper_blocked = (x_sub >= upper_bounds) & (grad_sub < 0)

        # Zero-out blocked components
        blocked = mask_lower_blocked | mask_upper_blocked
        direction[indices[blocked]] = 0.0

    return direction

