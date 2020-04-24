from setuptools import setup
import spotlob.version as spotlobversion


with open('README.md') as readme_file:
    readme = readme_file.read()

setup(
    name=spotlobversion.__name__,
    version=spotlobversion.__version__,
    author=spotlobversion.__author__,
    author_email=spotlobversion.__email__,
    description=spotlobversion.__summary__,
    license="BSD 3-clause",
    url=spotlobversion.__url__,
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
