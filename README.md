# Spotlob

Spotlob is a package to provide a simple, yet flexible
and fast workflow to measure properties of features in
images for scientific purposes.

It provides implementations for some use cases but can
be easily tuned and be extended towards specific applications.
Jupyter notebook widgets can be used to quickly find a
set of algorithms and parameters that work for a given
image and should also work for similar images.

The set of parameters and algorithms are stored as a
pipeline which can be restored and distributed and 
then be applied to a possibly large set of images.
This way, standard routines for repetitive and comparable
measurements for a defined type of images can be forged
into a small and portable file.

## When it's helpful

It is meant to be used in a scenario where a detection method
has to be applied repetetively onto a large set of similar images,
but the exact parameters are not clear.
If your set of tasks can be done by a collection of opencv-function
calls, that need tweaking and you wish to have a GUI to do that, 
but without to lose scripting options, spotlob is for you.

If you already have a couple of working python algorithms and
want to have a GUI for them to play around, use spotlob.

If you need to evaluate some images and you don't know which
of the thousand parameters of an algorithm work best, you might be
able to find the right ones faster with spotlob.

![Spotlob jupyter widget](/demo.gif)

### Usage example

```python
from spotlob.spim import Spim
from spotlob.defaults import default_pipeline

my_spim = Spim.from_file("image.jpg", cached=True)
my_pipe = default_pipeline()

result_spim = my_pipe.apply_all_steps(my_spim)

print(result_spim.get_data())
```

## What it's not

Spotlob is not a complete feature detection library and it does
not solve a detection problem, that has not already been solved
elsewhere.
It is not an alternative to opencv or scikit-image, but
rather builds on top of it.
At the moment it covers only a tiny fraction of what is possible
with these libraries, but it tries to make it easy for the
reader to use these (or any other python image processing library) 
within the spotlob workflow.

Although it might work with machine learning algorithms, it is
not tuned towards this usage and it is not designed with this
application in mind.

## Installation

Install with pip

```
pip install spotlob
```

