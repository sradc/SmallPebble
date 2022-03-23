import pathlib
import setuptools
from smallpebble.version import version

setuptools.setup(
    name="smallpebble",
    version=version,
    author="Sidney Radcliffe",
    author_email="sidneyradcliffe@gmail.com",
    description="Minimal automatic differentiation implementation in Python, NumPy.",
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/sradc/smallpebble",
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    install_requires=["numpy"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
