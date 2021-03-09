import pathlib
import pytest
import smallpebble as sp


def run_tests(arg_strings=[]):
    """Run SmallPebble's tests using pytest.
    
    arg_strings: a list of pytest arguments, 
    eg. arg_strings=['-x'] to stop after the first failed test.

    """
    smallpebble_dir = str(pathlib.Path(sp.__file__).parent)
    exit_code = pytest.main([smallpebble_dir] + arg_strings)
    return exit_code
