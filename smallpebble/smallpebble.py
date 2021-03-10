# Copyright 2021 Sidney Radcliffe
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
#
"""SmallPebble - Minimal automatic differentiation implementation in Python, NumPy.

For an introduction to autodiff and the basic concepts of this framework, see:
https://sidsite.com/posts/autodiff/

Consider this a resource on autodiff, rather than a library you should use.
(Popular libraries are: JAX, PyTorch, TensorFlow...)

Features:
- Various operations, such as matmul, conv2d, maxpool2d.
- Supports broadcasting.
- Nth derivatives.
"""
from collections import defaultdict
import math
import numpy


# ---------------- Enable switching between NumPy and CuPy dynamically.


array_library = numpy  # numpy or cupy


class ArrayLibraryProxy:
    """Enable switching between NumPy and CuPy dynamically.
    
    E.g.
    import smallpebble as sp
    import cupy

    sp.array_library = cupy
    """

    def __getattribute__(self, name):
        return getattr(array_library, name)


def np_add_at(a, indices, b):
    """Apply either np.add.at or cupy.scatter_add (which are equivalent),
    depending on which library is being used. 
    Do this because CuPy has no cupy.add.at.
    """
    if array_library.__name__ == "numpy":
        return array_library.add.at(a, indices, b)
    elif array_library.__name__ == "cupy":
        return array_library.scatter_add(a, indices, b)
    else:
        raise ValueError("Expected array_library.__name__ to be `numpy` or `cupy`.")


np = ArrayLibraryProxy()


# ----------------
# ---------------- AUTOMATIC DIFFERENTIATION
# ----------------
# Reverse mode automatic differentiation is carried out on `Variables` using `get_gradients`.


class Variable:
    "To be used in calculations that are to be differentiated."

    def __init__(self, array, local_gradients=()):
        self.array = array
        self.local_gradients = local_gradients


def get_gradients(variable):
    """Compute the first derivatives of `variable` with respect to child variables."""
    gradients = defaultdict(lambda: Variable(np.array(0, variable.array.dtype)))

    def compute_gradients(variable, path_value):
        for child_variable, multiply_by_locgrad in variable.local_gradients:
            value_of_path_to_child = multiply_by_locgrad(path_value)
            gradients[child_variable] = add(
                gradients[child_variable], value_of_path_to_child
            )
            compute_gradients(child_variable, value_of_path_to_child)

    gradients[variable] = Variable(np.ones(variable.array.shape, variable.array.dtype))
    compute_gradients(variable, gradients[variable])
    return gradients


# ---------------- BASE OPS
# Operations where `value` is not calculated in terms of other SmallPebble operations.
# Note: local_gradients' functions *are* calculated using SmallPebble operations,
# enabling higher order derivatives to be calculated.


def add(a, b):
    "Elementwise addition."
    value = a.array + b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = (
        (a_, lambda path_value: path_value),
        (b_, lambda path_value: path_value),
    )
    return Variable(value, local_gradients)


def add_at(a, indices, b):
    """Add the elements of `b` to the locations in `a` specified by `indices`.
    Allows adding to an element of `a` repeatedly.
    """
    value = a.array.copy()
    np_add_at(value, indices, b.array)
    local_gradients = (
        (a, lambda path_value: path_value),
        (b, lambda path_value: getitem(path_value, indices)),
    )
    return Variable(value, local_gradients)


def div(a, b):
    "Elementwise division."
    value = a.array / b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = (
        (a_, lambda path_value: div(path_value, b)),
        (b_, lambda path_value: mul(neg(path_value), div(a, square(b)))),
    )
    return Variable(value, local_gradients)


def enable_broadcast(a, b, matmul=False):
    "Enables gradients to be calculated when broadcasting."

    a_shape = a.array.shape[:-2] if matmul else a.array.shape
    b_shape = b.array.shape[:-2] if matmul else b.array.shape

    a_repeatdims, b_repeatdims = broadcastinfo(a_shape, b_shape)

    def multiply_by_locgrad_a(path_value):
        result = Variable(np.zeros(a.array.shape, a.array.dtype))
        path_value = reshape(sum(path_value, axis=a_repeatdims), a.array.shape)
        return result + path_value

    def multiply_by_locgrad_b(path_value):
        result = Variable(np.zeros(b.array.shape, b.array.dtype))
        path_value = reshape(sum(path_value, axis=b_repeatdims), b.array.shape)
        return result + path_value

    a_ = Variable(a.array, local_gradients=((a, multiply_by_locgrad_a),))
    b_ = Variable(b.array, local_gradients=((b, multiply_by_locgrad_b),))

    return a_, b_


def exp(a):
    "Elementwise exp of `a`."
    value = np.exp(a.array)
    local_gradients = ((a, lambda path_value: mul(path_value, exp(a))),)
    return Variable(value, local_gradients)


def expand_dims(a, axis):
    "Add new axes at axes specified in `axis`, with size of 1."
    value = np.expand_dims(a.array, axis)
    local_gradients = ((a, lambda path_value: reshape(path_value, a.array.shape)),)
    return Variable(value, local_gradients)


def getitem(a, indices):
    "Get elements of `a` using NumPy indexing."
    value = a.array[indices]

    def multiply_by_locgrad(path_value):
        "(Takes into account elements indexed multiple times.)"
        result = Variable(np.zeros(a.array.shape, a.array.dtype))
        return add_at(result, indices, path_value)

    local_gradients = ((a, multiply_by_locgrad),)
    return Variable(value, local_gradients)


def matmul(a, b):
    "Matrix multiplication."
    value = np.matmul(a.array, b.array)
    a_, b_ = enable_broadcast(a, b, matmul=True)
    local_gradients = (
        (a_, lambda path_value: matmul(path_value, matrix_transpose(b))),
        (b_, lambda path_value: matmul(matrix_transpose(a), path_value)),
    )
    return Variable(value, local_gradients)


def matrix_transpose(a):
    "Swap the end two axes."
    value = np.swapaxes(a.array, -2, -1)
    local_gradients = ((a, lambda path_value: matrix_transpose(path_value)),)
    return Variable(value, local_gradients)


def maxax(a, axis):
    "Reduce an axis, `axis`, to its max value."
    max_idx = np.argmax(a.array, axis)
    max_idx = np.expand_dims(max_idx, axis)
    value = np.take_along_axis(a.array, max_idx, axis)

    def multiply_by_locgrad(path_value):
        result = np.zeros(a.array.shape)
        np.put_along_axis(result, max_idx, np.array(1, a.array.dtype), axis)
        result = Variable(result)
        return mul(path_value, result)

    local_gradients = ((a, multiply_by_locgrad),)
    return Variable(value, local_gradients)


def mul(a, b):
    "Elementwise multiplication."
    value = a.array * b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = (
        (a_, lambda path_value: mul(path_value, b)),
        (b_, lambda path_value: mul(path_value, a)),
    )
    return Variable(value, local_gradients)


def neg(a):
    "Negate a variable."
    value = -a.array
    local_gradients = ((a, lambda path_value: neg(path_value)),)
    return Variable(value, local_gradients)


def pad(a, pad_width):
    "Zero pad `a`, where pad_width[d] = (left width, right width)."
    value = np.pad(a.array, pad_width)

    def multiply_by_locgrad(path_value):
        # Remove padding.
        indices = tuple(
            slice(L, path_value.array.shape[i] - R) for i, (L, R) in enumerate(pad_width)
        )
        return getitem(path_value, indices)

    local_gradients = ((a, multiply_by_locgrad),)
    return Variable(value, local_gradients)


def reshape(a, shape):
    "Reshape `a` into shape `shape`."
    value = np.reshape(a.array, shape)
    local_gradients = ((a, lambda path_value: reshape(path_value, a.array.shape)),)
    return Variable(value, local_gradients)


def setat(a, indices, b):
    """Similar to NumPy's `setitem`. Set values of `a` at `indices` to `b`...
    BUT `a` is not modified, a new object is returned."""
    value = a.array.copy()
    value[indices] = b.array

    def multiply_by_locgrad_a(path_value):
        return setat(path_value, indices, Variable(np.array(0, a.array.dtype)))

    def multiply_by_locgrad_b(path_value):
        return getitem(path_value, indices)

    local_gradients = ((a, multiply_by_locgrad_a), (b, multiply_by_locgrad_b))
    return Variable(value, local_gradients)


def square(a):
    "Square each element of `a`"
    value = np.square(a.array)

    def multiply_by_locgrad(path_value):
        return mul(path_value, mul(Variable(np.array(2, a.array.dtype)), a))

    local_gradients = ((a, multiply_by_locgrad,),)
    return Variable(value, local_gradients)


def sub(a, b):
    "Elementwise subtraction."
    value = a.array - b.array
    a_, b_ = enable_broadcast(a, b)
    local_gradients = (
        (a_, lambda path_value: path_value),
        (b_, lambda path_value: neg(path_value)),
    )
    return Variable(value, local_gradients)


def sum(a, axis=None):
    "Sum elements of `a`, along axes specified in `axis`."
    value = np.sum(a.array, axis)

    def multiply_by_locgrad(path_value):
        result = Variable(np.zeros(a.array.shape, a.array.dtype))
        if axis:
            # Expand dims so they can be broadcast.
            path_value = expand_dims(path_value, axis)
        return result + path_value

    local_gradients = ((a, multiply_by_locgrad),)
    return Variable(value, local_gradients)


def where(condition, a, b):
    "Condition is a boolean NumPy array, a and b are Variables."
    value = np.where(condition, a.array, b.array)

    a_, b_ = enable_broadcast(a, b)

    def multiply_by_locgrad_a(path_value):
        return where(
            condition,
            path_value,
            Variable(np.zeros(path_value.array.shape, dtype=a.array.dtype)),
        )

    def multiply_by_locgrad_b(path_value):
        return where(
            condition,
            Variable(np.zeros(path_value.array.shape, dtype=a.array.dtype)),
            path_value,
        )

    local_gradients = (
        (a_, multiply_by_locgrad_a),
        (b_, multiply_by_locgrad_b),
    )

    return Variable(value, local_gradients)


# ---------------- HIGHER OPS
# Operations written in terms of SmallPebble operations.


def conv2d(images, kernels, padding="SAME", strides=[1, 1]):
    """2D convolution, with same api as tf.nn.conv2d [1].

    Args:
        images: A `Variable` of shape [n_images, imheight, imwidth, n_channels].
        kernels: A `Variable` of shape [kernheight, kernwidth, channels_in, channels_out].
        `channels_in` must be the same as the images' `n_channels`.
        Note the plural, `kernels`, because there are actually n=channels_out kernels,
        where a single kernel has dimension [kernheight, kernwidth, channels_in].
        padding: Can be "SAME" or "VALID". Matches the padding used by tf.nn.conv2d.
        "SAME" results in the output images being of [imheight, imwidth],
        this is achieved by zero padding the input images.
        "VALID" is the result of convolution when there is no 0 padding.
        strides: Specifies the vertical and horizontal strides:
        strides = [vertical stride, horizontal stride]
    Returns:
        A `Variable` of dimensions [n_images, outheight, outwidth, channels_out].

    This implementation:
    - Indexes images to extract patches.
    - Reshapes patches and kernels, then matmuls them.
    - Reshapes the result back to the expected shape.
    (A diagram is helpful for understanding this..)

    [1] https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    """
    n_images, imheight, imwidth, _ = images.array.shape
    kernheight, kernwidth, channels_in, channels_out = kernels.array.shape
    stride_y, stride_x = strides
    images, imheight, imwidth = padding2d(
        images, padding, imheight, imwidth, stride_y, stride_x, kernheight, kernwidth
    )
    # Index to get the image patches:
    index_of_patches, outheight, outwidth, n_patches = patches_index(
        imheight, imwidth, kernheight, kernwidth, stride_y, stride_x
    )
    #  Extract patches, and reshape so can matrix multiply.
    patches = getitem(
        images, (slice(None), index_of_patches[0], index_of_patches[1], slice(None)),
    )
    patches_as_matrix = reshape(
        patches, [n_images * n_patches, kernheight * kernwidth * channels_in]
    )
    kernels_as_matrix = reshape(
        kernels, [kernheight * kernwidth * channels_in, channels_out]
    )
    result = matmul(patches_as_matrix, kernels_as_matrix)
    return reshape(result, [n_images, outheight, outwidth, channels_out])


def lrelu(a, alpha=0.02):
    return where(a.array > 0, a, mul(a, Variable(np.array(alpha, a.array.dtype))))


def maxpool2d(images, kernheight, kernwidth, padding="SAME", strides=[1, 1]):
    """Maxpooling on a `Variable` of shape [n_images, imheight, imwidth, n_channels]."""
    n_images, imheight, imwidth, n_channels = images.array.shape
    stride_y, stride_x = strides
    images, imheight, imwidth = padding2d(
        images, padding, imheight, imwidth, stride_y, stride_x, kernheight, kernwidth
    )
    index_of_patches, outheight, outwidth, n_patches = patches_index(
        imheight, imwidth, kernheight, kernwidth, stride_y, stride_x
    )
    patches = getitem(
        images, (slice(None), index_of_patches[0], index_of_patches[1], slice(None)),
    )
    patches_max = maxax(patches, axis=2)
    return reshape(patches_max, [n_images, outheight, outwidth, n_channels])


def padding2d(
    images, padding, imheight, imwidth, stride_y, stride_x, kernheight, kernwidth
):
    """Pad `images` for conv2d, maxpool2d."""
    if padding == "SAME":
        pad_top, pad_bottom = pad_amounts(imheight, stride_y, kernheight)
        pad_left, pad_right = pad_amounts(imwidth, stride_x, kernwidth)
        images = pad(
            images,
            pad_width=[(0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0),],
        )
        _, imheight, imwidth, _ = images.array.shape
    elif padding == "VALID":
        pass
    else:
        raise ValueError("padding must be 'SAME' or 'VALID'.")
    return images, imheight, imwidth


# ---------------- UTIL


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

    b_repeatdims = (b_shape_ == 1) & (a_shape_ > 1)
    b_repeatdims[:add_ndims_to_b] = True
    b_repeatdims = np.where(b_repeatdims == True)[0]

    return tuple(a_repeatdims), tuple(b_repeatdims)


def pad_amounts(array_length, stride, kern_length):
    """Amount of padding to be applied to a 1D array.
    Matches TensorFlow's conv2d padding, when padding='SAME'.
    """
    num_patches = math.ceil(array_length / stride)
    target_length = (num_patches - 1) * stride + kern_length
    padding_amount = max(target_length - array_length, 0)
    pad_front = padding_amount // 2
    pad_end = padding_amount - pad_front
    return pad_front, pad_end


def patches_index(imheight, imwidth, kernheight, kernwidth, stride_y, stride_x):
    "Index to get image patches, e.g. for 2d convolution."
    max_y_idx = imheight - kernheight + 1
    max_x_idx = imwidth - kernwidth + 1
    row_major_index = np.arange(imheight * imwidth).reshape([imheight, imwidth])
    patch_corners = row_major_index[0:max_y_idx:stride_y, 0:max_x_idx:stride_x]
    elements_relative = row_major_index[0:kernheight, 0:kernwidth]
    index_of_patches = patch_corners.reshape([-1, 1]) + elements_relative.reshape([1, -1])
    index_of_patches = np.unravel_index(index_of_patches, shape=[imheight, imwidth])
    outheight, outwidth = patch_corners.shape
    n_patches = outheight * outwidth
    return index_of_patches, outheight, outwidth, n_patches


# ---------------- AUGMENTING `VARIABLE`
# Add methods/properties to the Variable class for convenience/ux.
# This is not essential (was put down here to not distract the reader from autodiff).

# Enable the use of +, -, *, /...
Variable.__add__ = add
Variable.__mul__ = mul
Variable.__sub__ = sub
Variable.__truediv__ = div

# Indexing (no __setitem__, use `setat` instead):
Variable.__getitem__ = getitem

# Useful attributes:
Variable.dtype = property(lambda self: self.array.dtype)
Variable.ndim = property(lambda self: self.array.ndim)
Variable.shape = property(lambda self: self.array.shape)


# ----------------
# ---------------- DEPRECATED
# ----------------
# May or may not be removed at some point. Moved here to reduce noise.


def bincount(idx, a, size):
    """Faster than `np.add.at`. `idx`: flat NumPy array , `a` flat Variable, `size` int.
    
    Deprecation reason:
    add.at is the function we want really.
    Although bincount is faster when using numpy, it probably isn't worth the obfuscation.
    Also, CuPy doesn't benefit from the bincount version of add.at,
    since cupy.scatter_add is faster.
    """
    value = np.bincount(idx, a.array, minlength=size)
    local_gradients = ((a, lambda path_value: getitemflat(path_value, idx)),)
    return Variable(value, local_gradients)


def getitemflat(a, idx):
    """`a` is flat, `idx` is row major (and flat).
    
    Deprecation reason:
    This is the sister function of `sp.bincount`, and is deprecated for the same reason.
    """
    value = a.array[idx]
    local_gradients = ((a, lambda path_value: bincount(idx, path_value, a.array.size)),)
    return Variable(value, local_gradients)


def sliding_window_view(a, window_shape):
    value = np.lib.stride_tricks.sliding_window_view(a.array, window_shape)

    def multiply_by_locgrad(path_value):
        # this is generally quicker than add.at ..
        idx = np.arange(a.array.size).reshape(a.array.shape)
        idx = np.lib.stride_tricks.sliding_window_view(idx, window_shape)
        idx = idx.reshape(-1)

        path_value = reshape(path_value, (-1))
        return reshape(bincount(idx, path_value, a.array.size), a.array.shape)

    local_gradients = ((a, multiply_by_locgrad),)

    return Variable(value, local_gradients)
