# Copyright 2022-2026 The SmallPebble Authors, Sidney Radcliffe
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
"""Numerical gradients, used for debugging and tests.
Computed using finite differences.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

import smallpebble as sp

EPS = 1e-6


def rmse(a: np.ndarray, b: np.ndarray):
    "Root mean square error."
    return np.sqrt(np.mean((a - b) ** 2))


def numgrads(
    func: Callable, args: list[np.ndarray], n: int = 1, delta: float = 1e-6
) -> list[np.ndarray]:
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


def numgrad(func: Callable, a: np.ndarray, delta: float = 1e-6) -> np.ndarray:
    "Numerical gradient of func(a) at `a`."
    grad = np.zeros(a.shape, a.dtype)
    for index, _ in np.ndenumerate(grad):
        delta_array = np.zeros(a.shape, a.dtype)
        delta_array[index] = delta / 2
        grad[index] = np.sum((func(a + delta_array) - func(a - delta_array)) / delta)
    return grad


class NumericalError(Exception):
    pass


def check_grad(
    args: list[np.ndarray],
    sp_func: Callable,
    np_func: Callable,
    delta: int = 1,
    eps: float = EPS,
) -> None:
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
