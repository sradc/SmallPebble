"""Tests for SmallPebble.

These tests will generally compare the first and second 
derivatives from SmallPebble's autodiff against numerical derivatives.
"""
import numpy as np
import numpy.lib.stride_tricks as stride_tricks
import tensorflow as tf
import smallpebble as sp


EPS = 1e-7


def test_add():
    "Tests add, 1st and 2nd derivatives. Also implicitly: broadcasting, __add__."
    np.random.seed(0)

    a = sp.Variable(np.random.random([4, 2, 3]))
    b = sp.Variable(np.random.random([3, 4, 1, 2, 3]))
    y = a + b
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return a + b

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_add2():
    "Tests add, 1st and 2nd derivatives. Also implicitly: broadcasting, __add__."
    np.random.seed(0)

    a = sp.Variable(np.array(4.0))
    b = sp.Variable(np.random.random([3, 4, 1, 2, 3]))
    y = a + b
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return a + b

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1e-3)
    num_grads2 = numgrads(func, args, n=2, delta=1e-3)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_add_at():
    np.random.seed(0)

    a = sp.Variable(np.random.random([5]))
    indices = np.random.randint(0, len(a.array), size=[10])
    b = sp.Variable(np.random.random([10]))
    y = sp.add_at(a, indices, b)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        result = a.copy()
        np.add.at(result, indices, b)
        return result

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_bincount():
    np.random.seed(0)

    size = 34
    a = sp.Variable(np.random.random(size))
    idx = np.arange(size)
    y = sp.bincount(idx, a, size)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.bincount(idx, a, minlength=size)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_div():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]) * 1000)
    b = sp.Variable(np.random.random([4, 3, 1, 5]) * 1000)
    y = a / b
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return a / b

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1e-3)
    num_grads2 = numgrads(func, args, n=2, delta=1e-3)  # TODO work out why this fails
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_exp():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    y = sp.exp(a)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.exp(a)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1e-4)
    num_grads2 = numgrads(func, args, n=2, delta=1e-4)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_expand_dims():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    axis = [3]
    y = sp.expand_dims(a, axis=axis)

    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.expand_dims(a, axis=axis)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_getitem():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    indices = (slice(None), slice(1), (0, 2))
    y = sp.getitem(a, indices)

    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return a[indices]

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_getitemflat():
    np.random.seed(0)

    size = 34
    a = sp.Variable(np.random.random(size))
    idx = np.arange(size)
    y = sp.getitemflat(a, idx)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return a[idx]

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_matmul():
    np.random.seed(0)

    a = sp.Variable(np.random.random([2, 5]))
    b = sp.Variable(np.random.random([5, 9]))
    y = sp.matmul(a, b)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return np.matmul(a, b)

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_matmul2():
    np.random.seed(0)

    a = sp.Variable(np.random.random([2, 4, 3, 2, 5]))
    b = sp.Variable(np.random.random([1, 3, 5, 9]))
    y = sp.matmul(a, b)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return np.matmul(a, b)

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_matrix_tranpose():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    y = sp.matrix_transpose(a)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.swapaxes(a, -2, -1)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_maxax():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5, 4]))
    axis = -2
    y = sp.maxax(a, axis)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.expand_dims(np.max(a, axis), axis)

    y_np = func(a.array)
    args = [a.array]
    # Careful, delta could change the max value.
    num_grads = numgrads(func, args, n=1, delta=1e-5)
    num_grads2 = numgrads(func, args, n=2, delta=1e-5)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_mul1():
    np.random.seed(0)

    a = sp.Variable(np.random.random([4, 1, 3, 2, 5]))
    b = sp.Variable(np.random.random([1, 5, 3, 1, 5]))
    y = a * b
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return a * b

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1e-3)
    num_grads2 = numgrads(func, args, n=2, delta=1e-3)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_mul2():
    np.random.seed(0)

    a = sp.Variable(np.array(3.2))
    b = sp.Variable(np.random.random([3, 1, 2]))

    y = a * b
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return a * b

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)

        assert error < EPS, f"rmse = {error}"


def test_mul3():
    np.random.seed(0)

    a = sp.Variable(np.array([2.0, 3.0]))
    b = sp.Variable(np.random.random([3, 1, 2]))

    y = a * b
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return a * b

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)

        assert error < EPS, f"rmse = {error}"


def test_pad():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    pad_width = ((2, 1), (4, 4), (0, 1))
    y = sp.pad(a, pad_width)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.pad(a, pad_width)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_reshape():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    shape = (6, 5, 1)
    y = sp.reshape(a, shape)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.reshape(a, shape)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_setat():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    indices = (slice(None), slice(1), (0, 2))
    y = sp.setat(a, indices, sp.Variable(np.array(5.0)))

    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        result = a.copy()
        result[indices] = 5
        return result

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_sliding_window_view():

    a = sp.Variable(np.random.random([2, 5, 5, 2]))
    window_shape = (1, 2, 2, 2)
    y = sp.sliding_window_view(a, window_shape)

    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return stride_tricks.sliding_window_view(a, window_shape)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_square():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    y = sp.square(a)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return a ** 2

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1e-3)
    num_grads2 = numgrads(func, args, n=2, delta=1e-3)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_sub():
    "Tests add, 1st and 2nd derivatives. Also implicitly, broadcasting, __add__."
    np.random.seed(0)

    a = sp.Variable(np.random.random([4, 2, 3]))
    b = sp.Variable(np.random.random([4, 1, 2, 3]))
    y = a - b
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return a - b

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_sum():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    y = sp.sum(a)
    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.sum(a)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_grads2 = numgrads(func, args, n=2, delta=1)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


def test_where():
    np.random.seed(0)

    a = sp.Variable(np.random.random([4, 2, 3]))
    b = sp.Variable(np.random.random([3, 4, 1, 2, 3]))
    y = sp.where(a.array > b.array, a, b)

    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    grad_b_2nd = sp.get_gradients(grads[b])[b]
    sp_results = [y, grads[a], grads[b], grad_a_2nd, grad_b_2nd]

    def func(a, b):
        return np.where(a > b, a, b)

    y_np = func(a.array, b.array)
    args = [a.array, b.array]
    num_grads = numgrads(func, args, n=1, delta=1e-3)
    num_grads2 = numgrads(func, args, n=2, delta=1e-3)
    num_results = [y_np, num_grads[0], num_grads[1], num_grads2[0], num_grads2[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


# ---------------- HIGHER OPS


def generate_images_and_kernels(imagedims, kerndims):
    np.random.seed(0)
    n_images, imheight, imwidth, _ = imagedims
    kernheight, kernwidth, channels_in, channels_out = kerndims
    images = np.random.random([n_images, imheight, imwidth, channels_in]).astype(np.float64)
    kernels = np.random.random([kernheight, kernwidth, channels_in, channels_out])
    kernels = kernels.astype(np.float64)
    return images, kernels


def test_conv2d_result():
    "Check that rd.conv2d gets the same results as tf.conv2d."
    images, kernels = generate_images_and_kernels([3, 55, 66, 5], [13, 22, 5, 7])

    for stride in range(1, 11, 3):
        for padding in ["SAME", "VALID"]:
            strides = [stride, stride]
            result_sp = sp.conv2d(
                sp.Variable(images), sp.Variable(kernels), padding=padding, strides=strides
            )
            result_tf = tf.nn.conv2d(images, kernels, padding=padding, strides=strides)
            result_mean_error = np.mean(np.abs(result_sp.array - result_tf))
            assert result_mean_error < EPS, f"Mean error = {result_mean_error}"


def test_conv2d_grads():
    "Compare TensorFlow derivatives with SmallPebble."
    images, kernels = generate_images_and_kernels([5, 16, 12, 2], [2, 4, 2, 3])

    images_sp = sp.Variable(images)
    kernels_sp = sp.Variable(kernels)
    images_tf = tf.Variable(images)
    kernels_tf = tf.Variable(kernels)

    for stride in range(1, 11, 3):
        strides = [stride, stride]
        for padding in ["SAME", "VALID"]:

            # Calculate convolution and gradients with revdiff:
            result_sp = sp.conv2d(images_sp, kernels_sp, padding=padding, strides=strides)

            gradients_sp = sp.get_gradients(result_sp)
            grad_images_sp = gradients_sp[images_sp]
            grad_kernels_sp = gradients_sp[kernels_sp]

            # Calculate convolution and gradients with tensorflow:
            with tf.GradientTape() as tape:
                result_tf = tf.nn.conv2d(
                    images_tf, kernels_tf, padding=padding, strides=strides
                )
            grad_images_tf, grad_kernels_tf = tape.gradient(
                result_tf, [images_tf, kernels_tf]
            )

            # Compare the gradients:
            imgrad_error = np.mean(np.abs(grad_images_sp.array - grad_images_tf))
            assert imgrad_error < EPS, f"Image gradient error = {imgrad_error}"

            kerngrad_error = np.mean(np.abs(grad_kernels_sp.array - grad_kernels_tf))
            assert kerngrad_error < EPS, f"Kernel gradient error = {kerngrad_error}"


def test_conv2d_2nd_grads():
    "Currently only checks that no error is thrown when computing."
    images, kernels = generate_images_and_kernels([2, 33, 22, 5], [4, 7, 5, 2])
    strides = [2, 3]

    images_sp = sp.Variable(images)
    kernels_sp = sp.Variable(kernels)
    result_sp = sp.conv2d(images_sp, kernels_sp, strides=strides, padding="SAME")

    first_derivatives = sp.get_gradients(result_sp)

    images_sp_deriv1 = first_derivatives[images_sp]
    images_sp_deriv2 = sp.get_gradients(images_sp_deriv1)[images_sp]

    kernels_sp_deriv1 = first_derivatives[kernels_sp]
    kernels_sp_deriv2 = sp.get_gradients(kernels_sp_deriv1)[kernels_sp]

    assert np.sum(images_sp_deriv2.array) == 0
    assert np.sum(kernels_sp_deriv2.array) == 0


def test_maxpool2d():
    np.random.seed(0)
    images = np.random.random([4, 12, 14, 3])
    images_sp = sp.Variable(images)
    images_tf = tf.Variable(images)

    for stride in range(1, 11, 3):
        strides = [stride, stride]
        for padding in ["SAME", "VALID"]:
            strides = [stride, stride]

            result_sp = sp.maxpool2d(images_sp, 2, 2, padding, strides)

            gradients_sp = sp.get_gradients(result_sp)
            grad_images_sp = gradients_sp[images_sp]

            with tf.GradientTape() as tape:
                result_tf = tf.nn.max_pool2d(images_tf, [1, 2, 2, 1], strides, padding)

            grad_images_tf = tape.gradient(result_tf, [images_tf])[0]

            # Compare results:
            results_error = rmsq(result_sp.array, result_tf)
            assert results_error < EPS, f"Results error = {results_error}"

            # Compare gradients:
            imgrad_error = np.mean(np.abs(grad_images_sp.array - grad_images_tf))
            assert imgrad_error < EPS, f"Gradient error = {imgrad_error}"


def test_lrelu():
    np.random.seed(0)

    a = sp.Variable(np.random.random([3, 2, 5]))
    alpha = 0.1
    y = sp.lrelu(a, alpha)

    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return np.maximum(a, a * alpha)

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1e-6)
    num_grads2 = numgrads(func, args, n=2, delta=1e-6)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


# ---------------- MAGIC METHODS


def test_magic_getitem():
    a = sp.Variable(np.random.random([4, 5, 3]))
    y = a[0, :, 0:2]

    grads = sp.get_gradients(y)
    grad_a_2nd = sp.get_gradients(grads[a])[a]
    sp_results = [y, grads[a], grad_a_2nd]

    def func(a):
        return a[0, :, 0:2]

    y_np = func(a.array)
    args = [a.array]
    num_grads = numgrads(func, args, n=1, delta=1e-6)
    num_grads2 = numgrads(func, args, n=2, delta=1e-6)
    num_results = [y_np, num_grads[0], num_grads2[0]]

    for spval, numval in zip(sp_results, num_results):
        error = rmsq(spval.array, numval)
        assert error < EPS, f"rmse = {error}"


# ---------------- TEST UTILS


def numgrads(func, args, n=1, delta=1e-6):
    "Numerical nth derivatives of func w.r.t. args."
    gradients = []
    for i, arg in enumerate(args):

        def func_i(a):
            new_args = [x for x in args]
            new_args[i] = a
            return func(*new_args)

        gradfunc = lambda a: numgrad(func_i, a, delta)

        for _ in range(1, n):
            prev_gradfunc = gradfunc
            gradfunc = lambda a: numgrad(prev_gradfunc, a, delta)

        gradients.append(gradfunc(arg))
    return gradients


def numgrad(func, a, delta=1e-6):
    "Numerical gradient of func(a) at `a`."
    grad = np.zeros(a.shape, a.dtype)
    for index, _ in np.ndenumerate(grad):
        delta_array = np.zeros(a.shape, a.dtype)
        delta_array[index] = delta / 2
        grad[index] = np.sum((func(a + delta_array) - func(a - delta_array)) / delta)
    return grad


def rmsq(a: np.ndarray, b: np.ndarray):
    "Root mean square error."
    return np.sqrt(np.mean((a - b) ** 2))
