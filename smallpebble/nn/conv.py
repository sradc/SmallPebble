import smallpebble.core as core
from collections import defaultdict
import math
import numpy
import smallpebble.array_library as np

def conv2d(images, kernels, padding="SAME", strides=[1, 1]) -> core.Variable:
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

    [1] https://www.tensorflow.org/api_docs/python/tf/nn/conv2d
    """
    n_images, imheight, imwidth, _ = images.array.shape
    kernheight, kernwidth, channels_in, channels_out = kernels.array.shape
    stride_y, stride_x = strides
    images, imheight, imwidth = padding2d(
        images, padding, imheight, imwidth, stride_y, stride_x, kernheight, kernwidth
    )
    window_shape = (1, kernheight, kernwidth, channels_in)
    image_patches = strided_sliding_view(
        images, window_shape, (1, stride_y, stride_x, 1))
    outh, outw = image_patches.shape[1], image_patches.shape[2]
    patches_as_matrix = core.reshape(
        image_patches, [n_images * outh * outw,
                        kernheight * kernwidth * channels_in]
    )
    kernels_as_matrix = core.reshape(
        kernels, [kernheight * kernwidth * channels_in, channels_out]
    )
    result = core.matmul(patches_as_matrix, kernels_as_matrix)
    return core.reshape(result, [n_images, outh, outw, channels_out])


def maxpool2d(images, kernheight, kernwidth, padding="SAME", strides=[1, 1]):
    "Maxpooling on a `variable` of shape [n_images, imheight, imwidth, n_channels]."
    n_images, imheight, imwidth, n_channels = images.array.shape
    stride_y, stride_x = strides
    images, imheight, imwidth = padding2d(
        images, padding, imheight, imwidth, stride_y, stride_x, kernheight, kernwidth
    )
    window_shape = (1, kernheight, kernwidth, 1)
    image_patches = strided_sliding_view(
        images, window_shape, [1, stride_y, stride_x, 1])
    flat_patches_shape = image_patches.array.shape[:4] + (-1,)
    image_patches = core.reshape(image_patches, shape=flat_patches_shape)
    result = core.maxax(image_patches, axis=-1)
    return core.reshape(result, result.array.shape[:-1])


def pad(a, pad_width):
    "Zero pad `a`, where pad_width[dim] = (left width, right width)."
    value = np.pad(a.array, pad_width)

    def multiply_by_locgrad(path_value):
        indices = tuple(
            slice(L, path_value.shape[i] - R) for i, (L, R) in enumerate(pad_width)
        )
        return path_value[indices]

    local_gradients = [(a, multiply_by_locgrad)]
    return core.Variable(value, local_gradients)


def strided_sliding_view(a, window_shape, strides):
    "Sliding window view with strides (the unit of strides is index, not bytes)."
    value = core.np_strided_sliding_view(a.array, window_shape, strides)

    def multiply_by_locgrad(path_value):  # TODO: a faster method
        result = np.zeros(a.shape, a.dtype)
        core.np_add_at(core.np_strided_sliding_view(
            result, window_shape, strides), None, path_value)
        return result

    local_gradients = [(a, multiply_by_locgrad)]
    return core.Variable(value, local_gradients)


def padding2d(
    images, padding, imheight, imwidth, stride_y, stride_x, kernheight, kernwidth
):
    "Pad `images` for conv2d, maxpool2d."
    if padding == "SAME":
        pad_top, pad_bottom = pad_amounts(imheight, stride_y, kernheight)
        pad_left, pad_right = pad_amounts(imwidth, stride_x, kernwidth)
        images = pad(
            images,
            pad_width=[
                (0, 0),
                (pad_top, pad_bottom),
                (pad_left, pad_right),
                (0, 0),
            ],
        )
        _, imheight, imwidth, _ = images.array.shape
    elif padding == "VALID":
        pass
    else:
        raise ValueError("padding must be 'SAME' or 'VALID'.")
    return images, imheight, imwidth


def patches_index(imheight, imwidth, kernheight, kernwidth, stride_y, stride_x):
    "Index to get image patches, e.g. for 2d convolution."
    max_y_idx = imheight - kernheight + 1
    max_x_idx = imwidth - kernwidth + 1
    row_major_index = np.arange(
        imheight * imwidth).reshape([imheight, imwidth])
    patch_corners = row_major_index[0:max_y_idx:stride_y, 0:max_x_idx:stride_x]
    elements_relative = row_major_index[0:kernheight, 0:kernwidth]
    index_of_patches = patch_corners.reshape(
        [-1, 1]) + elements_relative.reshape([1, -1])
    index_of_patches = np.unravel_index(
        index_of_patches, shape=[imheight, imwidth])
    outheight, outwidth = patch_corners.shape
    n_patches = outheight * outwidth
    return index_of_patches, outheight, outwidth, n_patches


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
