'''
# Canonical correlation analysis problem
'''     

import numpy as np

class CanonicalCorrelation:
    def __init__(self, data):
        self.Qxy = data['Qxy']
        self.Qxx = data['Qxx']
        self.Qyy = data['Qyy']
        self.x0 = data['x0']
        self.nx = data['nx']
        self.ny = data['ny']

    def split_vars(self, z):
        wx = z[:self.nx]
        wy = z[self.nx:self.nx + self.ny]
        s1 = z[-2] # slack variable
        s2 = z[-1] # slack variable
        return wx, wy, s1, s2

    def obj(self, z, gradient=False):
        wx, wy, _, _ = self.split_vars(z)
        obj = wx.T @ self.Qxy @ wy

        if gradient is True:
            grad = np.zeros_like(z)
            grad[:self.nx] = self.Qxy @ wy
            grad[self.nx:self.nx + self.ny] = self.Qxy.T @ wx
            # grad[-2] and grad[-1] remain zero
            return obj, grad
        else:
            return obj

    def cons(self, z, gradient=False):
        wx, wy, s1, s2 = self.split_vars(z)

        cons = np.array([
            wx.T @ self.Qxx @ wx + s1 - 1,
            wy.T @ self.Qyy @ wy + s2 - 1
        ])

        if gradient is True:
            jac = np.zeros((2, len(z)))
            print(jac[0, :self.nx].shape, (2 * self.Qxx @ wx).shape)
            jac[0, :self.nx] = (2 * self.Qxx @ wx).reshape(self.ny)
            jac[0, -2] = 1  # ∂g1/∂s1
            jac[1, self.nx:self.nx + self.ny] = (2 * self.Qyy @ wy).reshape(self.ny)
            jac[1, -1] = 1  # ∂g2/∂s2
            return cons, jac
        else:
            return cons

