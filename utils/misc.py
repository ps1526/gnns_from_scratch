'''
Miscellaneous helpers used across the GNN pipeline.
'''

import numpy as np


def one_hot(labels, num_classes):
    '''
    Converts integer label array to one-hot matrix.
    labels     : (N,) integer array
    num_classes: C
    Returns    : (N, C) float array
    '''
    N = len(labels)
    out = np.zeros((N, num_classes), dtype=float)
    out[np.arange(N), labels] = 1.0
    return out


def numerical_gradient_check(f, x, eps=1e-5):
    '''
    Finite-difference numerical gradient of scalar function f at point x.
    Useful for verifying analytic gradient implementations.

    f   : callable that takes an array of shape x.shape and returns a scalar
    x   : numpy array (the point to evaluate at)
    eps : finite difference step size

    Returns array of same shape as x.
    '''
    grad = np.zeros_like(x, dtype=float)
    it = np.nditer(x, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = f(x)
        x[idx] = orig - eps
        f_minus = f(x)
        x[idx] = orig              # restore
        grad[idx] = (f_plus - f_minus) / (2 * eps)
        it.iternext()
    return grad
