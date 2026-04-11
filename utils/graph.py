'''
Graph preprocessing utilities.

These primitives are called by message-passing layers in modules.py before
any learnable transformation is applied to node features.
'''

import numpy as np


def add_self_loops(A):
    '''Returns A + I (adds a self-edge for every node).'''
    return A + np.eye(A.shape[0])


def degree_matrix(A):
    '''
    Returns the diagonal degree matrix D where D[i,i] = sum of row i in A.
    '''
    degrees = np.sum(A, axis=1)
    return np.diag(degrees)


def normalize_adjacency(A):
    '''
    Symmetric normalisation used in GCN: D^{-1/2} (A + I) D^{-1/2}.
    Avoids exploding/vanishing feature scale during message passing.
    '''
    A_hat = add_self_loops(A)
    D = np.sum(A_hat, axis=1)              # degree vector
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.where(D == 0, 1, D)))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def compute_laplacian(A):
    '''
    Unnormalised graph Laplacian L = D - A.
    Used in spectral GNN methods.
    '''
    return degree_matrix(A) - A
