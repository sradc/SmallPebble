# Copyright 2022 The SmallPebble Authors, Sidney Radcliffe
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
"""Allows SmallPebble to dynamically switch between using NumPy/CuPy,
as its ndarray library. The default is NumPy.

To SmallPebble devs:
Watch out for cases where NumPy and CuPy differ, 
e.g. np.add.at is cupy.scatter_add.
"""
from types import ModuleType

import numpy

library = numpy  # numpy or cupy


def use(array_library: ModuleType) -> None:
    """Set the array library that SmallPebble will use.

    Parameters
    ----------
    array_library : ModuleType
        Either NumPy (the SmallPebble default) or CuPy (for GPU acceleration).

    Example:
    ```python
    # Switch array library to CuPy.
    import cupy
    import smallpebble as sp
    sp.use(cupy)

    # To switch back to NumPy:
    import numpy
    sp.use(numpy)
    ```
    """
    global library
    library = array_library


def __getattr__(name: str):
    """Make this module act as a proxy, for NumPy/CuPy.

    Here's an example:
    ```python
    import smallpebble.array_library as array_library

    x = array_library.array([1, 2, 3]) ** 2  # <- a NumPy array
    ```
    In this example, a NumPy array is created,
    because `array_library.array` results in this function
    being called, which then calls `getattr(numpy, "array")`,
    which is NumPy's function for creating arrays.
    (The above assumes that `library == numpy`, which is the
    default. The same thing would happen but with CuPy arrays,
    if `library == cupy`.)
    """
    return getattr(library, name)
