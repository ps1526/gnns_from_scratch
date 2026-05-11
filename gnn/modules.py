"""
Base GCN in NumPy: one layer = sigma(A_hat @ X @ W + b).

Stack two layers for node classification (hidden + logits).
Aggregation is fixed in A_hat (use utils.graph.normalize_adjacency).
"""

import numpy as np

from utils.activations import relu, relu_grad, sigmoid, sigmoid_grad, tanh, tanh_grad
from utils.weight_init import he_init, xavier_init
from utils.losses import cross_entropy_grad, cross_entropy_loss


def _identity(x):
    return x


def _identity_grad(x):
    return np.ones_like(x, dtype=float)


_ACTIVATION_FORWARD = {
    "relu": relu,
    "sigmoid": sigmoid,
    "tanh": tanh,
    "linear": _identity,
    None: _identity,
}

_ACTIVATION_GRAD = {
    "relu": relu_grad,
    "sigmoid": sigmoid_grad,
    "tanh": tanh_grad,
    "linear": _identity_grad,
    None: _identity_grad,
}


def _pick_init(activation: str):
    if activation in ("relu", None, "linear"):
        return he_init
    return xavier_init


class GCNLayer:
    """
    Single GCN convolution: H = sigma(A_hat @ X @ W + b).

    A_hat : (N, N) precomputed normalized adjacency (e.g. symmetric GCN norm)
    X     : (N, in_dim)
    """

    def __init__(self, in_dim: int, out_dim: int, activation: str = "relu"):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation if activation is not None else "linear"
        init_fn = _pick_init(self.activation if self.activation != "linear" else "relu")
        # Last layer often uses linear; He is still fine for W init
        if self.activation == "linear":
            init_fn = xavier_init
        self.W = init_fn(in_dim, out_dim)
        self.b = np.zeros(out_dim, dtype=float)
        self.grad_W = None
        self.grad_b = None
        self._A_hat = None
        self._X = None
        self._M = None
        self._Z = None

    def _act(self, x):
        fn = _ACTIVATION_FORWARD.get(self.activation, _identity)
        return fn(x)

    def _act_grad(self, x):
        fn = _ACTIVATION_GRAD.get(self.activation, _identity_grad)
        return fn(x)

    def forward(self, A_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
        self._A_hat = A_hat
        self._X = X
        self._M = A_hat @ X
        self._Z = self._M @ self.W + self.b
        return self._act(self._Z)

    def backward(self, grad_H: np.ndarray) -> np.ndarray:
        grad_Z = grad_H * self._act_grad(self._Z)
        self.grad_W = self._M.T @ grad_Z
        self.grad_b = np.sum(grad_Z, axis=0)
        grad_M = grad_Z @ self.W.T
        return self._A_hat.T @ grad_M


class GCN:
    """
    Two-layer GCN for node-level logits: hidden (nonlinear) + readout (linear).
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        hidden_activation: str = "relu",
    ):
        self.layer1 = GCNLayer(in_dim, hidden_dim, activation=hidden_activation)
        self.layer2 = GCNLayer(hidden_dim, out_dim, activation="linear")

    def forward(self, A_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
        h = self.layer1.forward(A_hat, X)
        return self.layer2.forward(A_hat, h)

    def backward(self, grad_logits: np.ndarray) -> np.ndarray:
        g = self.layer2.backward(grad_logits)
        return self.layer1.backward(g)

    def parameters(self):
        return [
            (self.layer1.W, self.layer1.grad_W),
            (self.layer1.b, self.layer1.grad_b),
            (self.layer2.W, self.layer2.grad_W),
            (self.layer2.b, self.layer2.grad_b),
        ]

    def zero_grad(self):
        self.layer1.grad_W = self.layer1.grad_b = None
        self.layer2.grad_W = self.layer2.grad_b = None

    def step_sgd(self, lr: float):
        self.layer1.W -= lr * self.layer1.grad_W
        self.layer1.b -= lr * self.layer1.grad_b
        self.layer2.W -= lr * self.layer2.grad_W
        self.layer2.b -= lr * self.layer2.grad_b


def train_step(
    model: GCN,
    A_hat: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    lr: float,
) -> float:
    """
    One full forward / backward / SGD update. y is (N,) integer labels.
    Returns scalar loss.
    """
    logits = model.forward(A_hat, X)
    loss = cross_entropy_loss(logits, y)
    grad = cross_entropy_grad(logits, y)
    model.backward(grad)
    model.step_sgd(lr)
    return float(loss)


def predict(model: GCN, A_hat: np.ndarray, X: np.ndarray) -> np.ndarray:
    logits = model.forward(A_hat, X)
    return np.argmax(logits, axis=1)
