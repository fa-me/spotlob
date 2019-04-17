from setuptools import setup
import spotlob


with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name=spotlob.__name__,
    version=spotlob.__version__,
    author=spotlob.__author__,
    author_email=spotlob.__email__,
    description=spotlob.__summary__,
    license="BSD 3-clause",
    url=spotlob.__url__,
    classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: BSD License',
            'Natural Language :: English',
            'Programming Language :: Python :: 3 :: Only',
            'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    long_description=readme,
    long_description_content_type='text/markdown'
)
