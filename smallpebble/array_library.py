"""This module acts as a proxy, allowing NumPy/CuPy to be switched dynamically.
The default library is NumPy.

To switch to CuPy:
>> import smallpebble as sp
>> import cupy
>> sp.array_library.library = cupy

To switch back to NumPy:
>> import numpy
>> sp.array_library.library = numpy

Watch out for cases where NumPy and CuPy differ, 
e.g. np.add.at is cupy.scatter_add.
"""
import numpy

library = numpy  # numpy or cupy


def use(array_library):
    """Set array library to be NumPy or CuPy.

    E.g.
    >> import cupy
    >> import smallpebble as sp
    >> sp.use(cupy)

    To switch back to NumPy:
    >> import numpy
    >> sp.use(numpy)
    """
    global library
    library = array_library


def __getattr__(name):
    return getattr(library, name)
