'''
Weight initialisation strategies.

Xavier — tanh / sigmoid activations
He      — ReLU / leaky-ReLU activations
'''

import numpy as np


def xavier_init(fan_in, fan_out):
    '''
    Xavier / Glorot uniform init — recommended for tanh and sigmoid.
    Draws from Uniform[-limit, limit] where limit = sqrt(6 / (fan_in + fan_out)).
    '''
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, size=(fan_in, fan_out))


def he_init(fan_in, fan_out):
    '''
    He / Kaiming normal init — recommended for ReLU activations.
    Draws from N(0, sqrt(2 / fan_in)).
    '''
    std = np.sqrt(2.0 / fan_in)
    return np.random.randn(fan_in, fan_out) * std
