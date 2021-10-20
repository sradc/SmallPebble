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
