'''
Helper function.
    - Compute projected steepest descent direction
    - Compute projected direction
'''

import numpy as np

def projected_steepest_descent_direction(x, grad_f, bound_constraints=None):
    """
    Compute the projected steepest descent direction for bound constraints.

    Parameters:
        x: np.ndarray, current point.
        grad_f: np.ndarray, gradient at x.
        bound_constraints: tuple (lower_bounds, upper_bounds) with each a list or array.
    
    Returns:
        direction: np.ndarray, the projected steepest descent direction.
    
    Example of bound_constraints:
        bound_constraints = (
            [-np.inf, -np.inf, -np.inf, -np.inf, 0, 0],
            [ np.inf,  np.inf,  np.inf,  np.inf, np.inf, np.inf],
        )
    """
    direction = -grad_f.copy()

    if bound_constraints is not None:
        lower_bounds, upper_bounds = bound_constraints
        lower_bounds = np.asarray(lower_bounds).reshape(-1,1)
        upper_bounds = np.asarray(upper_bounds).reshape(-1,1)

        # Identify variables at bounds where descent direction would violate constraints
        at_lower = (x <= lower_bounds) & (grad_f > 0) 
        at_upper = (x >= upper_bounds) & (grad_f < 0)

        # Block descent in those directions
        direction[at_lower | at_upper] = 0.0

    return direction


def projection(x, bound_constraints=None):
    """
    Bound constraints is a tuple of lists.
    Example input:
        bound_constraints = (
            (-np.inf, -np.inf, -np.inf, -np.inf, 0, 0),
            (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf),
        )
    """
    x_proj = np.copy(x)

    if bound_constraints is not None:
        lower_bounds, upper_bounds = bound_constraints
        lower = np.array(lower_bounds).reshape(-1,1)
        upper = np.array(upper_bounds).reshape(-1,1)
        x_proj = np.minimum(np.maximum(x_proj, lower), upper)

    return x_proj
