# from https://gist.github.com/brikeats/687fd260e4c4a867ea1d7eb2c6532e89#file-image_parameter_tweak-py

import matplotlib.pyplot as plt
import inspect
from ipywidgets import *
from IPython.display import display
# %matplotlib inline
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'none'

# http://ipywidgets.readthedocs.io/en/latest/examples/Using%20Interact.html


def tweak_parameters(ims, image_processing_function, param_dict):
    """
    Interactively manipulate image processing parameters and see their effect on 
    a set of images. This function is appropriate for running in a jupyter notebook.

    Args:
        ims (list of arrays): the images or videos that you wish to analyze & display
        image_processing_function (callable): a function that accepts elements of
            `ims` as input, and returns images (i.e., 2- or 3D arrays). The first 
            argument is the input image/image sequence, and the other arguments must
            be specified in `param_dict` and/or have a default value.
        param_dict (dict): string as keys, length-4 tuples as values. Each value 
            must be of the form (minval, maxval, step, default). You can fix a 
            parameter by using functools.partial, or by defining a default value.

    Example:

    >> # A function with several arguments, the first of which is an image
    >> # It also returns an image (a 2D array in this case.)
    >> 
    >> def image_analysis_function(im, sigma, low_thresh=2, high_thresh=40):
    >> ....gray = img_as_ubyte(rgb2gray(im))
    >> ....return canny(gray, sigma=sigma, low_threshold=low_thresh, high_threshold=high_thresh)
    >>
    >> # Parameter dict
    >> param_dict = {'low_thresh': (0, 255, 1, 2),
    >>               'high_thresh': (0, 255, 1, 40),
    >>               'sigma': (0.5, 15., 0.1, 1.0)}
    >> from skimage import data
    >> tweak_parameters([data.astronaut(), data.rocket()], image_analysis_function, param_dict)
    """
    image_num_label = 'Image Number'
    default_kwargs = {argname: argspec[-1]
                      for argname, argspec in param_dict.iteritems()}
    widget_args = {argname: argspec[:-1]
                   for argname, argspec in param_dict.iteritems()}
    widget_dict = {}
    widget_dict[image_num_label] = IntSlider(min=1, max=len(ims),
                                             step=1, value=1,
                                             description='Image number',
                                             continuous_update=False)
    for argname, argspec in param_dict.iteritems():
        min_, max_, step, val = argspec
        if isinstance(argspec[0], int):
            WidgetClass = FloatSlider
        elif isinstance(argspec[0], float):
            WidgetClass = IntSlider
        else:
            raise NotImplementedError
        widget_dict[argname] = WidgetClass(min=min_, max=max_,
                                           step=step, value=val,
                                           description=argname,
                                           continuous_update=False)

    def analyze_and_plot(**kwargs):
        im_num = int(kwargs[image_num_label])
        # delete it so you don't pass it to image_processing_function
        del kwargs[image_num_label]
        image = ims[im_num-1]
        processed = image_processing_function(image, **kwargs)
        plt.figure(figsize=(10, 10))
        plt.imshow(processed)

        image = ims[0]
    interact(analyze_and_plot, **widget_dict)


##### Example Usage #####
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.feature import canny
from skimage.data import astronaut, rocket


# Your function. Must accept an numpy array as the first argument and return an image
def image_analysis_function(im, sigma, low_thresh=2, high_thresh=40):
    gray = img_as_ubyte(rgb2gray(im))
    edges = canny(gray, sigma=sigma, low_threshold=low_thresh,
                  high_threshold=high_thresh)
    return edges


# Dictionary of non-image parameters for the function. Values should be length-4 tuples
# (minval, maxval, step, default)
param_dict = {'low_thresh': (0, 255, 1, 2),
              'high_thresh': (0, 255, 1, 40),
              'sigma': (0.5, 15., 0.1, 1.0)}

tweak_parameters([astronaut(), rocket()], image_analysis_function, param_dict)
