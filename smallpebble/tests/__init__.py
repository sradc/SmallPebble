import pathlib
import pytest
import smallpebble as sp


def run_tests():
    "Run SmallPebble's tests using pytest."
    smallpebble_dir = pathlib.Path(sp.__file__).parent
    smallpebble_dir = str(smallpebble_dir)
    pytest.main([smallpebble_dir])
