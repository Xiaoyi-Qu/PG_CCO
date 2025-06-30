'''
File: print.py
Author: Xiaoyi Qu
--------------------------------------------
Description: Helper functions
             (1) Nullspace function
             (2) Lipschitz estimate function
             (3) Print function (problem, header, each iteration)
'''

import numpy as np
from numpy.linalg import svd

def nullspace(A, atol=1e-13, rtol=0):
    """Compute an approximate basis for the nullspace of A.

    The algorithm used by this function is based on the singular value
    decomposition of `A`.

    Parameters
    ----------
    A : ndarray
        A should be at most 2-D.  A 1-D array with length k will be treated
        as a 2-D with shape (1, k)
    atol : float
        The absolute tolerance for a zero singular value.  Singular values
        smaller than `atol` are considered to be zero.
    rtol : float
        The relative tolerance.  Singular values less than rtol*smax are
        considered to be zero, where smax is the largest singular value.

    If both `atol` and `rtol` are positive, the combined tolerance is the
    maximum of the two; that is::
        tol = max(atol, rtol * smax)
    Singular values smaller than `tol` are considered to be zero.

    Return value
    ------------
    ns : ndarray
        If `A` is an array with shape (m, k), then `ns` will be an array
        with shape (k, n), where n is the estimated dimension of the
        nullspace of `A`.  The columns of `ns` are a basis for the
        nullspace; each element in numpy.dot(A, ns) will be approximately
        zero.
    """

    A = np.atleast_2d(A)
    u, s, vh = svd(A)
    tol = max(atol, rtol * s[0])
    nnz = (s >= tol).sum()
    ns = vh[nnz:].conj().T
    return ns


def Lipschitz_estimate(p,x):
    # Preparation
    n = len(x)          # dim of iterate x
    m = 2*n             # number of random points
    ratio = 1e-3
    M = np.zeros(m)     # store Lipschitz estimate values
    f,g = p.obj(x, gradient=True)

    # Generate the random number
    arr = 2*ratio*(np.random.rand(m,n) - 1)

    # Compute the Lipschitz estimate
    for i in range(m):
        h = arr[i]
        fnew,gnew = p.obj(x+h, gradient=True)
        M[i] = np.linalg.norm(g-gnew, ord=2)/np.linalg.norm(h, ord=2)
    
    return np.max(M)


def print_prob(x, p, r, outID):
    c, J = p.cons(x, gradient=True)
    JTc  = np.dot(np.transpose(J),c)

    if outID is not None:
        filename = './log/{}.txt'.format(outID)
    else:
        filename = './log/log.txt'
    content = "==============================================================================\n"
    content += "                Solver: NAME TODO    Version: 0.1 (2023-07-29)                \n"
    content += "==============================================================================\n"
    content += "Problem Name:...................................%s\n"% p.p.name
    content += "   (1) Number of variables: %s\n   (2) Number of constraints: %s\n"% (len(x), p.p.m)
    content += "   (3) |J^Tc|: %8.5e\n   (4) |c|: %8.5e\n   (5) Num of zero(x): %5d\n   (6) Num of zero(y): %5d\n"% (np.linalg.norm(JTc), 
                np.linalg.norm(c), len(np.where(x[0:p.p.n] == 0)[0]), len(np.where(x[p.p.n:len(x)] == 0)[0]))
    content += "Regularizer Type:....................................L1\n"
    content += "Regularization Parameter:..............................Lambda=%5.2e\n"% (r.penalty)
    content += "******************************************************************************\n"
    content += "Comments: a. |.| represents 2-norm. b. y represents the dual variable.\n"
    content += "******************************************************************************\n"

    with open(filename, "w") as logfile:
        logfile.write(content)


def print_header(outID):
    if outID is not None:
        filename = './log/{}.txt'.format(outID)
    else:
        filename = './log/log.txt'
    column_titles = ' {Iter:^5s} {f:^11s} {fr:^11s} {g:^11s} {x:^11s} {v:^11s} {u:^11s} {s:^11s} {c:^11s} {alpha:^11s} {KKT:^11s} {tau:^11s} {H:^11s} {phi:^11s} {delta_qk:^11s} {J:^11s} {JTc:^11s} {rank:^11s} {sparse:^11s} {LM:^10s} {ustatus:^11s} {vstatus:^11s} {time_v:^11s} {time_u:^11s} \n'.format(Iter='Iter',
                    f='f', fr='f+r', g='g', x='|x|', v='|v|', u='|u|', s = '|s|', c ='|c|', alpha = 'alpha', KKT='KKT', tau = 'tau', 
                    H = "Lipschitz", phi = 'Merit fval', delta_qk = 'Delta_qk', J = 'singular(J)', JTc = '|J^Tc|', 
                    rank = 'rank(J)', sparse = 'sparsity', LM = '|y|', ustatus = 'ustatus', vstatus = 'vstatus',
                    time_v = 'time(v)', time_u = 'time(u)')
    with open(filename, "a") as logfile:
        logfile.write(column_titles)


def print_iteration(iteration, fval, frval, normg, normx, normv, normu, norms, normc, alpha, KKTnorm,
                    tau, normH, meritf, Delta_qk, condJ, normJTc, rank, sparsity, 
                    lagrange_multiplier, ustatus, vstatus, time_v, time_u, outID):
    if outID is not None:
        filename = './log/{}.txt'.format(outID)
    else:
        filename = './log/log.txt'

    contents = "{it:5d} {fval:8.5e} {frval:8.5e} {normg:8.5e} {normx:8.5e} {normv:8.5e} {normu:8.5e} {norms:8.5e} {normc:8.5e} {alpha:8.5e} {KKT:8.5e} {tau:8.5e} {normH:8.5e} {meritf:8.5e} {Delta_qk:8.5e} {condJ:8.5e} {normJTc:8.5e} {rankJ:10d} {sparsity:11d} {lm:11.5e} {ustatus:10d} {vstatus:10s} {time_v:8.5e} {time_u:8.5e} |\n".format(it=iteration, 
                fval=fval, frval=frval, normg=normg, normx=normx, normv = normv, normu = normu, norms = norms, 
                normc = normc, alpha = alpha, KKT = KKTnorm, tau = tau, normH=normH, meritf = meritf, Delta_qk = Delta_qk, 
                condJ = condJ, normJTc = normJTc, rankJ = rank, sparsity = sparsity, lm = lagrange_multiplier, 
                ustatus = ustatus, vstatus = vstatus, time_v = time_v, time_u = time_u)
    with open(filename, "a") as logfile:
        logfile.write(contents)
    

