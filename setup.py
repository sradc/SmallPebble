import pathlib
import setuptools
import smallpebble

setuptools.setup(
    name="smallpebble",
    version=smallpebble.__version__,
    author="Sidney Radcliffe",
    author_email="sidneyradcliffe@gmail.com",
    description="Minimal automatic differentiation implementation in Python, NumPy.",
    long_description=pathlib.Path('README.md').read_text(),
    long_description_content_type="text/markdown",
    url="https://github.com/sradc/smallpebble",
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    install_requires=["numpy>=1.20.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
