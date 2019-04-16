import cv2
import pandas as pd
import numpy as np
import os

from .process_steps import Reader, Converter, Preprocessor,\
    Binarization, Postprocessor, FeatureFinder, FeatureFilter
from .parameters import SpotlobParameterSet, EnumParameter,\
    BoolParameter, NumericRangeParameter


class SimpleReader(Reader):
    """Reads an image from a file as an RGB file.
    Standard image formats, such as `png`, `jpg`, `tif` are supported.
    It uses `cv2.imread`.
    """
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(SimpleReader, self).__init__(self.fn_read, pars)

    def fn_read(self, filepath):
        # load as color, convert from BGR to RGB
        if os.path.exists(filepath):
            return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB), {}
        else:
            raise IOError("File %s not found" % filepath)


class GreyscaleConverter(Converter):
    """Converts a color image to a greyscale image, by selecting one channel
    or by converting it to another color space and then selecting one channel.

    The supported options are given by the `conversion` parameter, which must be
    one of the following strings
    - `Hue`, `Saturation` or `Value` channel
    - `Red`, `Blue` or `Green` color channel
    - normal `Greyscale` conversion

    It uses the `cv2.cvtColor` function.
    Additionally the dark an bright parts can be switched using `invert=True`
    """
    def __init__(self):
        self.hsv_str_list = ["Hue", "Saturation", "Value"]
        self.rgb_str_list = ["Red channel", "Blue channel", "Green channel"]
        converter_options = ["Grey"] + self.hsv_str_list + self.rgb_str_list

        pars = SpotlobParameterSet(
            [EnumParameter("conversion",
                           converter_options[0],
                           converter_options),
             BoolParameter("invert", False)])
        super(GreyscaleConverter, self).__init__(self.convert, pars)

    def convert(self, rgb_image, conversion, invert):
        if conversion == "Grey":
            out = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
        else:
            if conversion in self.hsv_str_list:
                code = cv2.COLOR_RGB2HSV
                ch_nr = self.hsv_str_list.index(conversion)
                out = cv2.cvtColor(rgb_image, code)
            elif conversion in self.rgb_str_list:
                ch_nr = self.rgb_str_list.index(conversion)
                out = rgb_image

            out = out[:, :, ch_nr].astype(np.uint8)

        if invert:
            return cv2.bitwise_not(out)
        else:
            return out


class GaussianPreprocess(Preprocessor):
    """Blur the image with a gaussian blur with kernel size given 
    by `ksize`. It uses the `cv2.filter2D` function
    """
    def __init__(self, ksize):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("kernelsize", ksize, 1, 47, step=2)])
        super(GaussianPreprocess, self).__init__(self.filter_fn, pars)

    def filter_fn(self, grey_image, kernelsize):
        if kernelsize > 1:
            kernel = np.ones((kernelsize, kernelsize), np.float32)
            kernel /= kernelsize**2
            return cv2.filter2D(grey_image, -1, kernel)
        else: 
            return grey_image


class BinaryThreshold(Binarization):
    """Converts the image to a binary one, where the parts above
    the given threshold are set to 255 and the parts below it to 0.
    The sole parameter is the threshold value.
    It uses the `cv2.threshold` function.    
    """

    def __init__(self, threshold):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("threshold", threshold, 0, 255)])
        super(BinaryThreshold, self).__init__(self.threshold_fn, pars)

    def threshold_fn(self, grey_image, threshold):
        _, im = cv2.threshold(
            grey_image, threshold, 255, cv2.THRESH_BINARY)
        return im


class OtsuThreshold(Binarization):
    """Performs a binarization based on Otsu's algorithm.
    It uses the `cv2.threshold` function.    
    """
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(OtsuThreshold, self).__init__(self.threshold_fn, pars)

    def threshold_fn(self, grey_image):
        _, im = cv2.threshold(grey_image,
                              0,
                              255,
                              cv2.THRESH_OTSU)
        return im


class PostprocessNothing(Postprocessor):
    """This process is used as a placeholder for a postprocessing step
    and does not modify the image at all"""
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(PostprocessNothing, self).__init__(self.postprocess_fn, pars)

    def postprocess_fn(self, im):
        return im


class ContourFinderSimple(FeatureFinder):
    """Finds contours, i.e. lists of points that enclose connected areas of
    the same value. It is based on the `cv2.findContours` function
    """

    def __init__(self):
        pars = SpotlobParameterSet([])
        super(ContourFinderSimple, self).__init__(self.finder_fn, pars)

    def finder_fn(self, bin_im):
        cont_ret = cv2.findContours(bin_im,
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        # cont_ret is
        # contours, hierarchy for opencv >4.0
        # im, contours, hierarchy for opencv <=3.4

        contours = cont_ret[-2]
        return contours


class FeatureFormFilter(FeatureFilter):
    """It analyzes the contours and filters them using given criteria:
    
    - the enclosed area must be smaller (i.e. contain fewer pixels) than
      `minimal_area`
    - it solidity, i.e. the ratio of the area of the contour and its convex
      hull must be below a given value
    - if `remove_on_edge` is `True`, contours that touch the border of the
      image are filtered out

    """
    def __init__(self, size, solidity, remove_on_edge):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("minimal_area", size, 0, 10000),
             NumericRangeParameter("solidity_limit",
                                   solidity, 0, 1, step=0.01, type_=float),
             BoolParameter("remove_on_edge", remove_on_edge)])
        super(FeatureFormFilter, self).__init__(self.filter_fn, pars)

    def solidity(self, c):
        try:
            return cv2.contourArea(c)/cv2.contourArea(cv2.convexHull(c))
        except ZeroDivisionError:
            return 0

    def filter_fn(self,
                  contours,
                  image_shape,
                  minimal_area,
                  solidity_limit,
                  remove_on_edge):

        def valid_contour(c):
            valid_area = cv2.contourArea(c) > minimal_area
            valid_solidity = self.solidity(c) > solidity_limit
            valid_on_edge = not remove_on_edge or \
                self.is_off_border(c, image_shape)
            return (valid_area and
                    valid_solidity and
                    valid_on_edge)

        return [c for c in contours if valid_contour(c)]

    def is_off_border(self, contour, image_shape):
        """
        this function checks if a contour is touching the border
        of an image shaped like image_shape
        """
        bb_x, bb_y, bb_w, bb_h = cv2.boundingRect(contour)
        maxh = image_shape[0]
        maxw = image_shape[1]

        xMin = 0
        yMin = 0
        xMax = maxw - 1
        yMax = maxh - 1

        if any([bb_x <= xMin,
                bb_y <= yMin,
                bb_x+bb_w >= xMax,
                bb_y+bb_h >= yMax]):
            return False
        else:
            return True

    def draw_contours(self, image, contours):
        background = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        return draw_contours(background, contours)


def draw_contours(color_image, contours, color=(0, 255, 0)):
    return cv2.drawContours(color_image, contours, -1, color, 3)
