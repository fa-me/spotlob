from setuptools import setup, find_packages
import spotlob

setup(
    name=spotlob.__name__,
    version=spotlob.__version__,
    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    author=spotlob.__version__,
    description=spotlob.__summary__,
    license="BSD 3-clause"
)
