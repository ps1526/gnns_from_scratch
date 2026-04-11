'''
Activation functions and their gradients for backpropagation.
'''

import numpy as np


def softmax(x):
    '''
    Numerically stable softmax along axis=0.
    Subtracts max before exponentiating to prevent overflow.
    '''
    e_x = np.exp(x - np.max(x, axis=0, keepdims=True))
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    '''Gradient of ReLU: 1 where x > 0, else 0.'''
    return (x > 0).astype(float)


def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_grad(x, alpha=0.01):
    return np.where(x > 0, 1.0, alpha)


def sigmoid(x):
    # Clamp to prevent overflow in exp for large negative x
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_grad(x):
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x):
    return np.tanh(x)


def tanh_grad(x):
    return 1.0 - np.tanh(x) ** 2
