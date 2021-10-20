# Copyright 2021 The SmallPebble authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ----------------
# ---------------- AUTOMATIC DIFFERENTIATION
# ----------------
# This is the essence of the autodiff engine.
# Reverse mode automatic differentiation is carried out on variables using get_gradients.
from collections import defaultdict
import math
import numpy
import smallpebble.array_library as np


class Variable:
    "To be used in calculations to be differentiated."

    def __init__(self, array, local_gradients=()):
        self.array = array
        self.local_gradients = local_gradients


# ----------------
# ---------------- LAZY GRAPHS
# ----------------
# Forward pass laziness.
# SmallPebble operations are computed immediately,
# whereas here we enable lazy graphs, that are only evaluated when .run() is called.


class AssignmentError(Exception):
    pass


class Lazy:
    """A lazy node, for creating lazy graphs, that are only
    evaluated with .run() is called.

    Example:
    >> a = sp.Placeholder()
    >> b = sp.Variable(np.ones([4, 2]))
    >> y = sp.Lazy(sp.matmul)(a, b)
    >> a.assign_value(sp.Variable(np.ones([3, 4])))
    >> result = y.run()
    """

    def __init__(self, function):
        "Create a lazy node"
        self.function = function
        self.arguments = []

    def __call__(self, *args):
        """Set self.arguments, i.e. the child nodes of this lazy node.

        *args: Nodes that `function` will take as input;
            either sp.Variable or sp.Lazy instances.
        """
        if self.arguments:
            raise AssignmentError(f"Arguments already assigned to {self}.")
        self.arguments = args
        return self

    def run(self):
        "Compute the value of this node."
        if not self.arguments:
            raise AssignmentError(f"No arguments have been assigned to {self}.")
        argvals = (a.run() if hasattr(a, "run") else a for a in self.arguments)
        return self.function(*argvals)


class Placeholder(Lazy):
    "A placeholder lazy graph node, in which to place SmallPebble variables."

    def __init__(self):
        super().__init__(lambda a: a)

    def assign_value(self, variable):
        "Assign a Variable instance to this placeholder."
        self.arguments = [variable]


def get_gradients(variable):
    "Compute the first derivatives of `variable` with respect to child variables."
    gradients = defaultdict(lambda: 0)

    def compute_gradients(variable, path_value):
        for child_variable, multiply_by_locgrad in variable.local_gradients:
            value_of_path_to_child = multiply_by_locgrad(path_value)
            gradients[child_variable] += value_of_path_to_child
            compute_gradients(child_variable, value_of_path_to_child)

    gradients[variable] = np.ones(variable.array.shape, variable.array.dtype)
    compute_gradients(variable, gradients[variable])
    return gradients


def reshape(a, shape):
    "Reshape `a` into shape `shape`."
    value = np.reshape(a.array, shape)
    local_gradients = [(a, lambda path_value: path_value.reshape(a.array.shape))]
    return Variable(value, local_gradients)


def mul(a, b):
    "Elementwise multiplication."
    value = a.array * b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = [
        (a_, lambda path_value: path_value * b.array),
        (b_, lambda path_value: path_value * a.array),
    ]
    return Variable(value, local_gradients)


def neg(a):
    "Negate a variable."
    value = -a.array
    local_gradients = [(a, lambda path_value: -path_value)]
    return Variable(value, local_gradients)


def add(a, b):
    "Elementwise addition."
    value = a.array + b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = [
        # gradient is 1, so multiply by 1.
        (a_, lambda path_value: path_value),
        (b_, lambda path_value: path_value),
    ]
    return Variable(value, local_gradients)


def add_at(a, indices, b):
    """Add the elements of `b` to the locations in `a` specified by `indices`.
    Allows adding to an element of `a` repeatedly.
    """
    value = a.array.copy()
    np_add_at(value, indices, b.array)
    local_gradients = [
        (a, lambda path_value: path_value),
        (b, lambda path_value: path_value[indices]),
    ]
    return Variable(value, local_gradients)


def div(a, b):
    "Elementwise division."
    value = a.array / b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = [
        (a_, lambda path_value: path_value / b.array),
        (b_, lambda path_value: -path_value * a.array / np.square(b.array)),
    ]
    return Variable(value, local_gradients)


def enable_broadcast(a, b, matmul=False):
    "Enables gradients to be calculated when broadcasting."
    a_shape = a.array.shape[:-2] if matmul else a.array.shape
    b_shape = b.array.shape[:-2] if matmul else b.array.shape
    a_repeatdims, b_repeatdims = broadcastinfo(a_shape, b_shape)

    def multiply_by_locgrad_a(path_value):
        path_value = np.sum(path_value, axis=a_repeatdims).reshape(a.array.shape)
        return np.zeros(a.array.shape, a.array.dtype) + path_value

    def multiply_by_locgrad_b(path_value):
        path_value = np.sum(path_value, axis=b_repeatdims).reshape(b.array.shape)
        return np.zeros(b.array.shape, b.array.dtype) + path_value

    a_ = Variable(a.array, local_gradients=[(a, multiply_by_locgrad_a)])
    b_ = Variable(b.array, local_gradients=[(b, multiply_by_locgrad_b)])
    return a_, b_


def setat(a, indices, b):
    """Similar to NumPy's `setitem`. Set values of `a` at `indices` to `b`...
    BUT `a` is not modified, a new object is returned."""
    value = a.array.copy()
    value[indices] = b.array

    def multiply_by_locgrad_a(path_value):
        path_value[indices] = np.array(0, a.array.dtype)
        return path_value

    def multiply_by_locgrad_b(path_value):
        return path_value[indices]

    local_gradients = [(a, multiply_by_locgrad_a), (b, multiply_by_locgrad_b)]
    return Variable(value, local_gradients)


def sum(a, axis=None):
    "Sum elements of `a`, along axes specified in `axis`."
    value = np.sum(a.array, axis)

    def multiply_by_locgrad(path_value):
        result = np.zeros(a.array.shape, a.array.dtype)
        if axis:  # Expand dims so they can be broadcast.
            path_value = np.expand_dims(path_value, axis)
        return result + path_value

    local_gradients = [(a, multiply_by_locgrad)]
    return Variable(value, local_gradients)


def where(condition, a, b):
    "Condition is a boolean NumPy array, a and b are Variables."
    value = np.where(condition, a.array, b.array)
    a_, b_ = enable_broadcast(a, b)

    def multiply_by_locgrad_a(path_value):
        return np.where(
            condition,
            path_value,
            np.zeros(path_value.shape, a.array.dtype),
        )

    def multiply_by_locgrad_b(path_value):
        return np.where(
            condition,
            np.zeros(path_value.shape, a.array.dtype),
            path_value,
        )

    local_gradients = [
        (a_, multiply_by_locgrad_a),
        (b_, multiply_by_locgrad_b),
    ]
    return Variable(value, local_gradients)


def sub(a, b):
    "Elementwise subtraction."
    value = a.array - b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = [
        (a_, lambda path_value: path_value),
        (b_, lambda path_value: -path_value),
    ]
    return Variable(value, local_gradients)


def exp(a):
    "Elementwise exp of `a`."
    value = np.exp(a.array)
    local_gradients = [(a, lambda path_value: path_value * np.exp(a.array))]
    return Variable(value, local_gradients)


def expand_dims(a, axis):
    "Add new axes with size of 1, indices specified by `axis`."
    value = np.expand_dims(a.array, axis)
    local_gradients = [(a, lambda path_value: path_value.reshape(a.array.shape))]
    return Variable(value, local_gradients)


def getitem(a, indices):
    "Get elements of `a` using NumPy indexing."
    value = a.array[indices]

    def multiply_by_locgrad(path_value):
        "(Takes into account elements indexed multiple times.)"
        result = np.zeros(a.array.shape, a.array.dtype)
        np_add_at(result, indices, path_value)
        return result

    local_gradients = [(a, multiply_by_locgrad)]
    return Variable(value, local_gradients)


def log(a):
    "Elementwise log of `a`."
    value = np.log(a.array)
    local_gradients = [(a, lambda path_value: path_value / a.array)]
    return Variable(value, local_gradients)


def matmul(a, b):
    "Matrix multiplication."
    value = np.matmul(a.array, b.array)
    a_, b_ = enable_broadcast(a, b, matmul=True)
    local_gradients = [
        (a_, lambda path_value: np.matmul(path_value, np.swapaxes(b.array, -2, -1))),
        (b_, lambda path_value: np.matmul(np.swapaxes(a.array, -2, -1), path_value)),
    ]
    return Variable(value, local_gradients)


def matrix_transpose(a):
    "Swap the end two axes."
    value = np.swapaxes(a.array, -2, -1)
    local_gradients = [(a, lambda path_value: np.swapaxes(path_value, -2, -1))]
    return Variable(value, local_gradients)


def maxax(a, axis):
    "Reduce an axis, `axis`, to its max value."
    # Note, implementation now more complicated because CuPy doesn't have put_along_axis.
    axis = axis if axis >= 0 else a.ndim + axis
    value = np.swapaxes(a.array, axis, -1)
    value = value.reshape([-1, value.shape[-1]])
    flatshape = value.shape
    idx = np.argmax(value, axis=-1)
    value = np.take_along_axis(value, idx[..., np.newaxis], -1)
    value = value.reshape(tuple(1 if i == axis else v for i, v in enumerate(a.shape)))

    def multiply_by_locgrad(path_value):
        result = np.zeros(flatshape)
        result[np.arange(result.shape[0]), idx] = 1
        swapped_shape = list(a.shape)
        swapped_shape[axis], swapped_shape[-1] = swapped_shape[-1], swapped_shape[axis]
        result = result.reshape(swapped_shape)
        result = np.swapaxes(result, axis, -1)
        return path_value * result

    local_gradients = ((a, multiply_by_locgrad),)
    return Variable(value, local_gradients)


def square(a):
    "Square each element of `a`"
    value = np.square(a.array)

    def multiply_by_locgrad(path_value):
        return path_value * 2 * a.array

    local_gradients = [(a, multiply_by_locgrad)]
    return Variable(value, local_gradients)


def broadcastinfo(a_shape, b_shape):
    "Get which dimensions are added or repeated when `a` and `b` are broadcast."
    ndim = max(len(a_shape), len(b_shape))

    add_ndims_to_a = ndim - len(a_shape)
    add_ndims_to_b = ndim - len(b_shape)

    a_shape_ = np.array([1] * add_ndims_to_a + list(a_shape))
    b_shape_ = np.array([1] * add_ndims_to_b + list(b_shape))

    if not all((a_shape_ == b_shape_) | (a_shape_ == 1) | (b_shape_ == 1)):
        raise ValueError(f"could not broadcast shapes {a_shape} {b_shape}")

    a_repeatdims = (a_shape_ == 1) & (b_shape_ > 1)  # the repeated dims
    a_repeatdims[:add_ndims_to_a] = True  # the added dims
    a_repeatdims = np.where(a_repeatdims == True)[0]  # indices of axes where True
    a_repeatdims = [int(i) for i in a_repeatdims]

    b_repeatdims = (b_shape_ == 1) & (a_shape_ > 1)
    b_repeatdims[:add_ndims_to_b] = True
    b_repeatdims = np.where(b_repeatdims == True)[0]
    b_repeatdims = [int(i) for i in b_repeatdims]

    return tuple(a_repeatdims), tuple(b_repeatdims)


def np_add_at(a, indices, b):
    "Apply either numpy.add.at or cupy.scatter_add, depending on which library is used."
    if np.library.__name__ == "numpy":
        np.add.at(a, indices, b)
    elif np.library.__name__ == "cupy":
        np.scatter_add(a, indices, b)
    else:
        raise ValueError("Expected np.library.__name__ to be `numpy` or `cupy`.")


def np_strided_sliding_view(x, window_shape: tuple, strides: tuple):
    """Similar to np.sliding_window_view [1], but with strides,
    (the unit of strides is index number, not bytes).

    Args:
        x: an n-dimensional array.
        window_shape: a window size, for each dimension of x.
        strides: number of indices to skip. Like tf.nn.conv2d strides,
        rather than ndarray.strides.
    Returns:
        A view into x, based on the window size.

    [1] https://github.com/numpy/numpy/blob/main/numpy/lib/stride_tricks.py
    """
    # Need the checks, because as_strided is not memory safe.
    if not len(window_shape) == x.ndim:
        raise ValueError(f"Must provide one window size for each dimension of x.")
    if not len(strides) == x.ndim:
        raise ValueError(f"Must provide one stride size for each dimension of x.")
    if any(size < 0 for size in window_shape):
        raise ValueError("`window_shape` cannot contain negative values")
    if any(stride < 0 for stride in strides):
        raise ValueError("`strides` cannot contain negative values")
    if any(x_size < w_size for x_size, w_size in zip(x.shape, window_shape)):
        raise ValueError("window shape cannot be larger than input array shape")
    reduced_shape = tuple(
        math.ceil((x - w + 1) / s) for x, s, w in zip(x.shape, strides, window_shape)
    )
    out_shape = reduced_shape + window_shape
    skipping_strides = tuple(
        xstride * stride for xstride, stride in zip(x.strides, strides)
    )
    out_strides = skipping_strides + x.strides
    return np.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape)


# ---------------- AUGMENTING `VARIABLE`
# Add methods/properties to the Variable class for convenience/user-experience.
# This isn't essential to autodiff.


# Enable the use of +, -, *, /...
Variable.__add__ = add
Variable.__mul__ = mul
Variable.__neg__ = neg
Variable.__sub__ = sub
Variable.__truediv__ = div

# Enable indexing via `[` and `]`. (No __setitem__, use the function `setat` instead.)
Variable.__getitem__ = getitem

# Useful attributes:
Variable.dtype = property(lambda self: self.array.dtype)
Variable.ndim = property(lambda self: self.array.ndim)
Variable.shape = property(lambda self: self.array.shape)
