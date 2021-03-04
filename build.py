"""Build for distribution."""
import pathlib
import shutil
import subprocess

root = pathlib.Path(__file__).parent

# Delete old 'dist' folder, where distribution is located.
dist_dir = root / "dist"
if dist_dir.is_dir():
    shutil.rmtree(dist_dir)
assert not dist_dir.is_dir(), 'Failed to delete old distribution directory.'

# Build to distribution.
subprocess.run(
    ["python", "setup.py", "sdist", "bdist_wheel"], cwd=root
)

# Submit to pypi with: twine upload dist/*
