"""Allows SmallPebble to dynamically switch between NumPy/CuPy,
for its ndarray computations. The default is NumPy.

Note to SmallPebble devs:
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
