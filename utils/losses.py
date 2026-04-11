'''
Loss functions and their gradients.

Cross-entropy     — node / graph classification (multi-class)
Binary CE         — link prediction (binary labels)
MSE               — regression tasks on graphs
'''

import numpy as np
from utils.activations import softmax
from utils.misc import one_hot


def cross_entropy_loss(logits, y_true):
    '''
    Softmax cross-entropy loss for multi-class classification.

    logits : (N, C) raw scores
    y_true : (N, C) one-hot labels  OR  (N,) integer labels

    Returns scalar mean loss.
    '''
    N = logits.shape[0]
    if y_true.ndim == 1:
        y_true = one_hot(y_true, logits.shape[1])
    probs = softmax(logits.T).T           # apply softmax row-wise
    log_probs = -np.log(np.clip(probs, 1e-12, 1.0))
    return np.sum(log_probs * y_true) / N


def cross_entropy_grad(logits, y_true):
    '''
    Combined softmax + cross-entropy gradient.
    Simplifies to (probs - y_true) / N.
    '''
    N = logits.shape[0]
    if y_true.ndim == 1:
        y_true = one_hot(y_true, logits.shape[1])
    probs = softmax(logits.T).T
    return (probs - y_true) / N


def binary_cross_entropy_loss(probs, y_true):
    '''
    Binary cross-entropy for link prediction / binary classification.

    probs  : (N,) or (N, 1) predicted probabilities in (0, 1)
    y_true : (N,) or (N, 1) binary labels {0, 1}
    '''
    probs = np.clip(probs.ravel(), 1e-12, 1 - 1e-12)
    y = y_true.ravel()
    return -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))


def binary_cross_entropy_grad(probs, y_true):
    '''Gradient of BCE w.r.t. probs.'''
    N = len(y_true)
    probs = np.clip(probs.ravel(), 1e-12, 1 - 1e-12)
    y = y_true.ravel()
    return (-(y / probs) + (1 - y) / (1 - probs)) / N


def mse_loss(y_pred, y_true):
    '''Mean squared error — useful for regression tasks on graphs.'''
    return np.mean((y_pred - y_true) ** 2)


def mse_grad(y_pred, y_true):
    N = y_pred.size
    return 2.0 * (y_pred - y_true) / N
