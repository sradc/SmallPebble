"""This module acts as a proxy, allowing NumPy/CuPy to be switched dynamically.
The default library is NumPy.

To switch to CuPy:
>> import smallpebble as sp
>> import cupy
>> sp.array_library.library = cupy

To switch back to NumPy:
>> import numpy
>> sp.array_library.library = cupy

Watch out for cases where NumPy and CuPy differ, 
e.g. np.add.at is cupy.scatter_add.
"""
import numpy

library = numpy  # numpy or cupy


def __getattr__(name):
    return getattr(library, name)
