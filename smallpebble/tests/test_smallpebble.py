"""Tests for SmallPebble.
Check results, and derivatives against numerical derivatives.
"""
import pytest
import smallpebble as sp
import tensorflow as tf

np = sp.np

EPS = 1e-6


@pytest.fixture(autouse=True)
def set_np_seed():
    np.random.seed(0)
    yield


class NumericalError(Exception):
    pass


def compare_results(args, sp_func, np_func, delta=1, eps=EPS):
    """Compares:
    - SmallPebble function output against NumPy function output.
    - SmallPebble gradient against numerical gradient.

    Notes: 
    `delta` can be 1 for linear functions,
    but should otherwise be a very small number.

    `eps` may need to be adjusted, in the case of
    inaccurate numerical approximations.
    """
    # Compute SmallPebble results
    args_sp = [sp.Variable(a) for a in args]
    y_sp = sp_func(*args_sp)
    grads_sp = sp.get_gradients(y_sp)
    grads_sp = [grads_sp[var] for var in args_sp]

    # Compute numerical results
    y_np = np_func(*args)
    grads_np = numgrads(np_func, args, n=1, delta=delta)

    # Compare output values
    error = rmse(y_sp.array, y_np)
    if error > eps:
        raise NumericalError("function output rmse:", error)

    # Compare gradient values
    for i, (spval, npval) in enumerate(zip(grads_sp, grads_np)):
        if error > eps:
            raise NumericalError(f"arg[{i}] gradient rmse:", error)


# ---------------- TEST OPS


def test_add():
    args = [
        np.random.random([20, 20]),
        np.random.random([20, 20]),
    ]
    compare_results(args, sp.add, np.add)


def test_add_broadcast():
    args = [
        np.random.random([4, 2, 3]),
        np.random.random([3, 4, 1, 2, 3]),
    ]
    compare_results(args, sp.add, np.add)


def test_add_broadcast_scalar():
    args = [
        np.array(4.0),
        np.random.random([3, 4, 1, 2, 3]),
    ]
    compare_results(args, sp.add, np.add)


def test_add_operator():
    args = [
        np.random.random([20, 20]),
        np.random.random([20, 20]),
    ]
    func = lambda a, b: a + b
    compare_results(args, func, func)


def test_add_at():
    len_a, len_b = 5, 10
    args = [np.random.random([len_a]), np.random.random([len_b])]
    indices = np.random.randint(0, len_a, size=[len_b])
    sp_func = lambda a, b: sp.add_at(a, indices, b)

    def np_func(a, b):
        result = a.copy()
        np.add.at(result, indices, b)
        return result

    compare_results(args, sp_func, np_func)


def test_div_broadcast_operator():
    args = [np.random.random([3, 2, 5]) * 1000, np.random.random([4, 3, 1, 5]) * 2 + 1]
    func = lambda a, b: a / b
    compare_results(args, func, func, delta=1e-5)


def test_exponential():
    args = [np.random.random([3, 2, 5])]
    compare_results(args, sp.exp, np.exp, delta=1e-5)


def test_expand_dims():
    args = [np.random.random([3, 2, 5])]
    axis = [3]
    sp_func = lambda a: sp.expand_dims(a, axis)
    np_func = lambda a: np.expand_dims(a, axis)
    compare_results(args, sp_func, np_func)


def test_getitem():
    args = [np.random.random([3, 2, 5])]
    indices = (slice(None), slice(1), (0, 2))
    sp_func = lambda a: sp.getitem(a, indices)
    np_func = lambda a: a[indices]
    compare_results(args, sp_func, np_func)


def test_getitem_dunder():
    args = [np.random.random([4, 5, 3])]
    func = lambda a: a[0, :, 0:2]
    compare_results(args, func, func)


def test_log():
    args = [
        np.random.random([3, 2, 5]) * 100 + 1,
    ]
    compare_results(args, sp.log, np.log, delta=1e-5)


def test_leaky_relu():
    args = [np.random.random([8, 8, 8]) * 10 - 5]
    alpha = 0.1
    sp_func = lambda a: sp.leaky_relu(a, alpha)
    np_func = lambda a: np.maximum(a, a * alpha)
    compare_results(args, sp_func, np_func, delta=1e-5)


def test_matmul():
    args = [np.random.random([2, 5]), np.random.random([5, 9])]
    compare_results(args, sp.matmul, np.matmul)


def test_matmul_broadcast():
    args = [np.random.random([2, 4, 3, 2, 5]), np.random.random([1, 3, 5, 9])]
    compare_results(args, sp.matmul, np.matmul)


def test_matrix_transpose():
    args = [np.random.random([3, 2, 5])]
    np_func = lambda a: np.swapaxes(a, -2, -1)
    compare_results(args, sp.matrix_transpose, np_func)


def test_maxax():
    args = [np.random.random([3, 2, 5, 4])]
    axis = -2
    sp_func = lambda a: sp.maxax(a, axis)
    np_func = lambda a: np.max(a, axis, keepdims=True)
    compare_results(args, sp_func, np_func, delta=1e-5)


def test_mul_broadcast_operator():
    args = [np.random.random([4, 1, 3, 2, 5]), np.random.random([5, 3, 1, 5])]
    func = lambda a, b: a * b
    compare_results(args, func, func)


def test_neg_operator():
    args = [np.random.random([5, 3, 1, 5])]
    func = lambda a: -a
    compare_results(args, func, func)


def test_pad():
    args = [np.random.random([3, 2, 5])]
    pad_width = ((2, 1), (4, 4), (0, 1))
    sp_func = lambda a: sp.pad(a, pad_width)
    np_func = lambda a: np.pad(a, pad_width)
    compare_results(args, sp_func, np_func)


def test_reshape():
    args = [np.random.random([3, 2, 5])]
    shape = (6, 5, 1)
    sp_func = lambda a: sp.reshape(a, shape)
    np_func = lambda a: a.reshape(shape)
    compare_results(args, sp_func, np_func)


def test_setat():
    args = [
        np.random.random([3, 2, 5]),
        np.array(5),
    ]
    indices = (slice(None), slice(1), (0, 2))
    sp_func = lambda a, b: sp.setat(a, indices, b)

    def np_func(a, b):
        result = a.copy()
        result[indices] = b
        return result

    compare_results(args, sp_func, np_func)


def test_softmax():
    args = [np.random.random([100, 10])]
    np_func = lambda a: np.exp(a) / np.sum(np.exp(a), axis=-1, keepdims=True)
    compare_results(args, sp.softmax, np_func, delta=1e-6, eps=1e-9)


def test_square():
    args = [np.random.random([3, 2, 5])]
    compare_results(args, sp.square, np.square)


def test_sub_broadcast_operator():
    args = [np.random.random([4, 2, 3]), np.random.random([4, 1, 2, 3])]
    func = lambda a, b: a - b
    compare_results(args, func, func)


def test_sum():
    args = [np.random.random([4, 3, 2, 5])]
    axis = (1, 3)
    sp_func = lambda a: sp.sum(a, axis)
    np_func = lambda a: np.sum(a, axis)
    compare_results(args, sp_func, np_func)


def test_where():
    args = [
        np.random.random([4, 2, 3]),
        np.random.random([3, 4, 1, 2, 3]),
    ]
    condition = args[0] > args[1]
    sp_func = lambda a, b: sp.where(condition, a, b)
    np_func = lambda a, b: np.where(condition, a, b)
    compare_results(args, sp_func, np_func)


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
                sp.Variable(images), sp.Variable(kernels), padding=padding, strides=strides,
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
            print("\n\n")
            print(stride)
            print(padding)

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
            imgrad_error = np.mean(np.abs(grad_images_sp - grad_images_tf))
            assert imgrad_error < EPS, f"Image gradient error = {imgrad_error}"

            kerngrad_error = np.mean(np.abs(grad_kernels_sp - grad_kernels_tf))
            assert kerngrad_error < EPS, f"Kernel gradient error = {kerngrad_error}"


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
            results_error = rmse(result_sp.array, result_tf)
            assert results_error < EPS, f"Results error = {results_error}"

            # Compare gradients:
            imgrad_error = np.mean(np.abs(grad_images_sp - grad_images_tf))
            assert imgrad_error < EPS, f"Gradient error = {imgrad_error}"


# ---------------- TEST DELAYED GRAPH EXECUTION


def test_operation_and_placeholder_basic():
    x_in = sp.Placeholder()
    x = sp.Lazy(lambda x: x ** 2)(x_in)
    x_in.assign_value(3)
    assert x.run() == 9, "Expecting a value of 9"


def test_operation_and_placeholder_result():
    a_array = np.ones([10, 5])
    b_array = np.ones([5, 10])

    a = sp.Placeholder()
    b = sp.Placeholder()
    c = sp.Lazy(sp.matmul)(a, b)
    d = sp.Lazy(sp.mul)(c, sp.Variable(np.array(5)))

    a.assign_value(sp.Variable(a_array))
    b.assign_value(sp.Variable(b_array))

    result = d.run()  # runs the graph

    assert np.all(result.array - np.matmul(a_array, b_array) * 5) == 0, "invalid result"


def test_operation_and_placeholder_gradients():
    a_array = np.ones([10, 5])
    b_array = np.ones([5, 10])
    a_sp = sp.Variable(a_array)
    b_sp = sp.Variable(b_array)

    a = sp.Placeholder()
    b = sp.Placeholder()
    y = sp.Lazy(sp.matmul)(a, b)

    a.assign_value(a_sp)
    b.assign_value(b_sp)
    result = y.run()

    grads = sp.get_gradients(result)
    sp_results = [result.array, grads[a_sp], grads[b_sp]]

    def func(a, b):
        return np.matmul(a, b)

    y_np = func(a_array, b_array)
    args = [a_array, b_array]
    num_grads = numgrads(func, args, n=1, delta=1)
    num_results = [y_np, num_grads[0], num_grads[1]]

    for spval, numval in zip(sp_results, num_results):
        error = rmse(spval, numval)
        assert error < EPS, f"rmse = {error}"


# ---------------- TRAINING - HELPER FUNCTIONS


def test_learnable_and_get_learnables():
    """Test that sp.get_learnables correctly obtains 
    sp.Variables that have been flagged by sp.learnable().
    """
    param_1 = sp.learnable(sp.Variable(np.array(5)))
    param_2 = sp.learnable(sp.Variable(np.array(10)))

    a = sp.Placeholder()
    b = sp.Placeholder()
    y = sp.Lazy(sp.matmul)(a, b)
    y = sp.Lazy(sp.mul)(y, param_1)
    y = sp.Lazy(sp.add)(y, param_2)

    params = sp.get_learnables(y)

    assert params == [param_1, param_2], "Did not get params."


# ---------------- NEURAL NETWORKS

def test_adam():
    a = sp.Variable(np.random.random([3, 5]))
    b = sp.Variable(np.random.random([5, 3]))
    y_true = sp.Variable(np.ones([3, 3]))

    adam = sp.Adam()
    losses = []
    for _ in range(100):
        y_pred = sp.matmul(a, b)
        loss = sp.sum(sp.square(y_true - y_pred))
        losses.append(loss.array)
        
        gradients = sp.get_gradients(loss)
        adam.training_step([a, b], gradients)
    
    assert np.all(np.diff(losses) < 0), 'Not converging.'



def test_sgd_step():
    a = sp.Variable(np.random.random([3, 5]))
    b = sp.Variable(np.random.random([5, 3]))
    y_true = sp.Variable(np.ones([3, 3]))

    losses = []
    for _ in range(100):
        y_pred = sp.matmul(a, b)
        loss = sp.sum(sp.square(y_true - y_pred))
        losses.append(loss.array)
        
        gradients = sp.get_gradients(loss)
        sp.sgd_step([a, b], gradients)
    
    assert np.all(np.diff(losses) < 0), 'Not converging.'




# ---------------- UTIL


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


def rmse(a: np.ndarray, b: np.ndarray):
    "Root mean square error."
    return np.sqrt(np.mean((a - b) ** 2))

