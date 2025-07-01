import numpy as np

class ProjectionOperator:
    def __init__(self, set_type='nonnegative', **kwargs):
        """
        Initializes the projection operator for a specified convex set.

        Parameters:
        - set_type : str, type of convex set
        - kwargs : additional parameters for specific sets
        """
        self.set_type = set_type
        self.params = kwargs

    def project(self, x):
        """
        Projects the input vector x onto the chosen convex set.

        Parameters:
        - x : np.ndarray, the input vector

        Returns:
        - projected vector (np.ndarray)
        """
        if self.set_type == 'nonnegative':
            return np.maximum(x, 0)
        elif self.set_type == 'box':
            l = self.params.get('lower', -np.inf)
            u = self.params.get('upper', np.inf)
            return np.clip(x, l, u)
        elif self.set_type == 'mixed':
            pass
        else:
            raise NotImplementedError(f"Projection onto '{self.set_type}' not implemented.")

