'''
utils package — re-exports all public symbols so callers can do either:

    from utils import relu, normalize_adjacency, adam_update  # flat import
    from utils.activations import relu                         # scoped import
'''

from utils.activations import (
    softmax,
    relu, relu_grad,
    leaky_relu, leaky_relu_grad,
    sigmoid, sigmoid_grad,
    tanh, tanh_grad,
)

from utils.losses import (
    cross_entropy_loss, cross_entropy_grad,
    binary_cross_entropy_loss, binary_cross_entropy_grad,
    mse_loss, mse_grad,
)

from utils.weight_init import (
    xavier_init,
    he_init,
)

from utils.graph import (
    add_self_loops,
    degree_matrix,
    normalize_adjacency,
    compute_laplacian,
)

from utils.optimizers import (
    sgd_update,
    adam_update,
    gradient_clip,
)

from utils.metrics import (
    accuracy,
    auc_roc,
    f1_score,
)

from utils.misc import (
    one_hot,
    numerical_gradient_check,
)
