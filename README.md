# Spotlob

Spotlob is a package to provide a simple, yet flexible
and fast workflow to measure properties of features in
images for scientific purposes.

It provides implementations for some use cases but can
be easily tuned and be extended towards specific purposes.
Jupyter notebook widgets can be used to quickly find a
set of algorithms and parameters that work for a given
image and should also work for similar images.

The set of parameters and algorithms are stored as a
pipeline which can be restored and distributed and 
then be applied to a possibly large set of images.
This way, standard routines for repetitive and comparable
measurements for a defined type of images can be forged
into a small and portable file.

## Why is it helpful


## What it's not

Spotlob is not a complete feature detection library and
it does try to solve an arbitrary detection problem.
It is not an alternative to opencv or scikit-image, but
rather builds on top of it.

At the moment it covers only a tiny fraction of what is possible
with these libraries, but it tries to make it easy for the
reader to use these within the spotlob workflow, if he or
she already knows what works for them.

## Getting started

### Installation

#### with pip
```
pip install spotlob
```

### Usage example

```
from spotlob.spim import Spim
from spotlob.defaults import default_pipeline()

my_spim = Spim.from_file("path/to/image.jpg", cached=True)
my_pipe = default_pipeline()

result_spim = my_pipe.apply_all_steps(my_spim)

print(result_spim.get_data())
```
