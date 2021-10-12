from collections import defaultdict
import math
import numpy
import smallpebble.array_library as np
import smallpebble.core as core
# ---------------- SMALLPEBBLE OPERATIONS
# Operations on variables, that return variables.
# The operations are either,
# written in terms of NumPy/CuPy operations,
# in which case local_gradients needs to be defined,
# or,
# written in terms of other SmallPebble operations, in which case,
# local_gradients doesn't need to be defined.


def leaky_relu(a, alpha=0.02):
    "Elementwise leaky relu."
    multiplier = np.where(a.array > 0, np.array(
        1, a.dtype), np.array(alpha, a.dtype))
    value = a.array * multiplier
    local_gradients = [(a, lambda path_value: path_value * multiplier)]
    return core.Variable(value, local_gradients)


def softmax(a, axis=-1):
    "Softmax on `axis`."
    exp_a = core.exp(a - core.Variable(np.max(a.array)))
    # ^ Shift to improve numerical stability. See:
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    sum_shape = list(a.shape)
    sum_shape[axis] = 1
    return exp_a / core.reshape(core.sum(exp_a, axis=axis), sum_shape)


# ---------------- LEARNABLES


def learnable(variable):
    "Flag `variable` as learnable."
    variable.is_learnable = True
    return variable


def get_learnables(lazy_node):
    "Get `variables` where is_learnable=True from a lazy graph."
    learnable_vars = []

    def find_learnables(node):
        for child in getattr(node, "arguments", []):
            if getattr(child, "is_learnable", False):
                learnable_vars.append(child)
            find_learnables(child)

    find_learnables(lazy_node)
    return learnable_vars


# ----------------
# ---------------- NEURAL NETWORKS
# ----------------
# Helper functions for creating neural networks.


class Adam:
    """Adam optimization for SmallPebble variables.
    See Algorithm 1, https://arxiv.org/abs/1412.6980
    Kingma, Ba. Adam: A Method for Stochastic Optimization. 2017.
    """

    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = defaultdict(lambda: 0)
        self.m = defaultdict(lambda: 0)
        self.v = defaultdict(lambda: 0)

    def training_step(self, variables, gradients):
        for variable in variables:
            self.t[variable] += 1
            g = gradients[variable]
            self.m[variable] = self.beta1 * \
                self.m[variable] + (1 - self.beta1) * g
            self.v[variable] = self.beta2 * \
                self.v[variable] + (1 - self.beta2) * g ** 2
            m_ = self.m[variable] / (1 - self.beta1 ** self.t[variable])
            v_ = self.v[variable] / (1 - self.beta2 ** self.t[variable])
            variable.array = variable.array - \
                self.alpha * m_ / (np.sqrt(v_) + self.eps)


def batch(X, y, size, seed=None):
    "Yield sub-batches of X,y, randomly selecting with replacement."
    assert (y.ndim == 1) and (X.shape[0] == y.size), "unexpected dimensions."
    if seed:
        np.random.seed(seed)
    while True:
        idx = np.random.randint(0, y.size, size)
        yield X[idx, ...], y[idx]


def convlayer(height, width, depth, n_kernels, padding="VALID", strides=(1, 1)):
    "Create a convolutional neural network layer."
    sigma = np.sqrt(6 / (height * width * depth + height * width * n_kernels))
    kernels_init = sigma * \
        (np.random.random([height, width, depth, n_kernels]) - 0.5)
    kernels = learnable(core.Variable(kernels_init))
    def func(images, kernels): return core.conv2d(
        images, kernels, padding, strides)
    return lambda images: core.Lazy(func)(images, kernels)


def cross_entropy(y_pred: core.Variable, y_true: np.array, axis=-1) -> core.Variable:
    """Cross entropy.
    Args:
        y_pred: A sp.Variable instance of shape [batch_size, n_classes]
        y_true: A NumPy array, of shape [batch_size], containing the true class labels.
    Returns:
        A scalar, reduced by summation.
    """
    indices = (np.arange(len(y_true)), y_true)
    return core.neg(sum(core.log(core.getitem(y_pred, indices))))


def he_init(insize, outsize) -> np.array:
    "He weight initialization."
    sigma = np.sqrt(4 / (insize + outsize))
    return np.random.random([insize, outsize]) * sigma - sigma / 2


def linearlayer(insize, outsize) -> core.Lazy:
    "Create a linear fully connected neural network layer."
    weights = learnable(core.Variable(he_init(insize, outsize)))
    bias = learnable(core.Variable(np.ones([outsize], np.float32)))
    def func(a, weights, bias): return core.matmul(a, weights) + bias
    return lambda a: core.Lazy(func)(a, weights, bias)


def onehot(y, n_classes) -> np.array:
    "Onehot encode vector y with classes 0 to n_classes-1."
    result = np.zeros([len(y), n_classes])
    result[np.arange(len(y)), y] = 1
    return result


def sgd_step(variables, gradients, learning_rate=0.001) -> None:
    "A single step of gradient descent. Modifies each variable.array directly."
    for variable in variables:
        gradient = gradients[variable]
        variable.array -= learning_rate * gradient
