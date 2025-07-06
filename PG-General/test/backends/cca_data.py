"""
Generate synthetic data for canonical correlation analysis problem

    max_{w_x ∈ ℝⁿˣ, w_y ∈ ℝⁿʸ}  ⟨w_x, Σ_xy w_y⟩ + λ (‖w_x‖₁ + ‖w_y‖₁)
                   subject to:   ⟨w_x, Σ_xx w_x⟩ ≤ 1
                                 ⟨w_y, Σ_yy w_y⟩ ≤ 1

    - Σ_xy: cross-covariance matrix between X and Y
    - Σ_xx, Σ_yy: covariance matrices of X and Y
    - Note that the regularization is added in a separate file
    
    - nx: num of features for matrix X
    - ny: number of features for matrix Y
    - N: number of samples
"""

import numpy as np
from sklearn.cross_decomposition import CCA

def DataSCCA(nx, ny, N):
    print('Data is generating ...')

    # Construct v1
    v1 = np.zeros((nx, 1))
    v1[:nx // 8] = 1
    v1[nx // 8:nx // 4] = -1

    # Construct v2
    v2 = np.zeros((ny, 1))
    v2[ny - nx // 4:ny - nx // 8] = 1
    v2[ny - nx // 8:] = -1

    # Shared latent variable
    u = np.random.randn(N, 1)

    # Add noise and generate X, Y
    X = (v1 + np.random.normal(0, 0.1, (nx, 1))) @ u.T  # shape: (nx, N)
    Y = (v2 + np.random.normal(0, 0.1, (ny, 1))) @ u.T  # shape: (ny, N)

    # Canonical Correlation Analysis (Starting point)
    cca = CCA(n_components=1)
    cca.fit(X.T, Y.T)
    a, b = cca.transform(X, Y) # canonical vectors

    data = {
        'Qxy': X @ Y.T,
        'Qxx': X @ X.T,
        'Qyy': Y @ Y.T,
        'nx': nx,
        'ny': ny,
        'x0': np.concatenate((a,b),axis=0)
    }

    print('Done with canonical correlation analysis data generation!!!')
    return data

