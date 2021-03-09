"""Build for distribution."""
import pathlib
import shutil
import subprocess
import pytest
import smallpebble.tests

root = pathlib.Path(__file__).parent

# Check tests pass:
exit_code = smallpebble.tests.run_tests(["-x"])
assert exit_code != pytest.ExitCode.TESTS_FAILED, "Aborting build, due to failed test."

# Delete old 'dist' folder, where distribution is located.
dist_dir = root / "dist"
if dist_dir.is_dir():
    shutil.rmtree(dist_dir)
assert not dist_dir.is_dir(), "Failed to delete old distribution directory."

# Build dist.
subprocess.run(["python", "setup.py", "sdist", "bdist_wheel"], cwd=root)

# Finally, submit to pypi with: twine upload dist/*
