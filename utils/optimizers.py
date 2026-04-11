'''
Stateless optimizer update functions.

Moment state (m, v for Adam) is stored and passed in by the caller,
keeping these functions pure and easy to use with any parameter set.
'''

import numpy as np


def sgd_update(param, grad, lr):
    '''Vanilla SGD: param <- param - lr * grad.'''
    return param - lr * grad


def adam_update(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    '''
    Adam optimizer update.

    param : current parameter array
    grad  : gradient array (same shape)
    m, v  : first and second moment estimates (initialise as zeros)
    t     : current timestep (1-indexed)

    Returns (updated_param, updated_m, updated_v).
    '''
    m = beta1 * m + (1.0 - beta1) * grad
    v = beta2 * v + (1.0 - beta2) * grad ** 2
    m_hat = m / (1.0 - beta1 ** t)        # bias correction
    v_hat = v / (1.0 - beta2 ** t)
    param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
    return param, m, v


def gradient_clip(grads, max_norm):
    '''
    Clips a list of gradient arrays so their global L2 norm <= max_norm.
    Returns the clipped list (same structure).
    '''
    total_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads))
    if total_norm > max_norm:
        scale = max_norm / (total_norm + 1e-12)
        grads = [g * scale for g in grads]
    return grads
