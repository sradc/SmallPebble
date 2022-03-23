# Copyright 2022 The SmallPebble authors
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
import pathlib
import pytest
import smallpebble as sp
import smallpebble.tests.test_smallpebble


def run_tests(arg_strings=[]):
    """Run SmallPebble's tests using pytest.main().

    arg_strings: a list of pytest arguments,
    eg. arg_strings=['-x'] to stop after the first failed test.
    """
    smallpebble_dir = str(pathlib.Path(sp.__file__).parent)
    exit_code = pytest.main([smallpebble_dir] + arg_strings)
    return exit_code


def run_tests_with(array_library):
    """Set array_library, and then run tests (manually,
    by finding `test_` functions in `test_smallpebble.py").
    If any test fails, an error is raised.
    """
    original_array_library = sp.array_library
    try:
        sp.array_library = array_library
        tests = (
            f for f in dir(smallpebble.tests.test_smallpebble) if f.startswith("test_")
        )
        for test in tqdm_wrap(tests):
            getattr(smallpebble.tests.test_smallpebble, test)()
    finally:
        sp.array_library = original_array_library


def tqdm_wrap(iterable):
    """Use tqdm if it is installed."""
    try:
        from tqdm import tqdm

        return tqdm(iterable)
    except ModuleNotFoundError:
        return iterable
