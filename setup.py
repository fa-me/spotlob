from setuptools import setup, find_packages
import spotlob

setup(
    name=spotlob.__name__,
    version=spotlob.__version__,
    packages=["spotlob", "spotlob.tests"],
    author=spotlob.__version__,
    author_email=spotlob.__email__,
    description=spotlob.__summary__,
    license="BSD 3-clause",
    url=spotlob.__url__
)
