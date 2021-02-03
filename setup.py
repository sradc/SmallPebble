import setuptools

with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()

setuptools.setup(
    name="smallpebble",
    version="0.1.0",
    author="Sidney Radcliffe",
    author_email="sidneyradcliffe@gmail.com",
    description="Minimal automatic differentiation implementation in Python, NumPy.",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/sradc/smallpebble",
    license="Apache License 2.0",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
