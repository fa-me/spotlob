import cv2
import pandas as pd
import numpy as np

import process
from parameters import *


class SimpleReader(process.Reader):
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(SimpleReader, self).__init__(self.fn_read, pars)

    def fn_read(self, filepath):
        # load as color, convert from BGR to RGB
        return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB), {}


class GreyscaleConverter(process.Converter):
    def __init__(self):
        self.hsv_str_list = ["Hue", "Saturation", "Value"]
        self.rgb_str_list = ["Red channel", "Blue channel", "Green channel"]
        converter_options = ["Grey"] + self.hsv_str_list + self.rgb_str_list

        pars = SpotlobParameterSet(
            [EnumParameter("conversion", converter_options[0], converter_options),
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


class GaussianPreprocess(process.Preprocessor):
    def __init__(self, ksize):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("kernelsize", ksize, 1, 47, step=2)])
        super(GaussianPreprocess, self).__init__(self.filter_fn, pars)

    def filter_fn(self, grey_image, kernelsize):
        kernel = np.ones((kernelsize, kernelsize), np.float32)/kernelsize**2
        return cv2.filter2D(grey_image, -1, kernel)


class BinaryThreshold(process.Binarisation):
    def __init__(self, threshold):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("threshold", threshold, 0, 255)])
        super(BinaryThreshold, self).__init__(self.threshold_fn, pars)

    def threshold_fn(self, grey_image, threshold):
        thresh, im = cv2.threshold(
            grey_image, threshold, 255, cv2.THRESH_BINARY)
        return im


class PostprocessNothing(process.Postprocessor):
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(PostprocessNothing, self).__init__(self.postprocess_fn, pars)

    def postprocess_fn(self, im):
        return im


class ContourFinderSimple(process.FeatureFinder):
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(ContourFinderSimple, self).__init__(self.finder_fn, pars)

    def finder_fn(self, bin_im):
        cim, contours, _ = cv2.findContours(
            bin_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


class FeatureFormFilter(process.FeatureFilter):
    def __init__(self, size, solidity):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("minimal_area", size, 0, 10000),
             NumericRangeParameter(
                 "solidity_limit", solidity, 0, 1, step=0.01, type_=float)])
        super(FeatureFormFilter, self).__init__(self.filter_fn, pars)

    def solidity(self, c):
        try:
            return cv2.contourArea(c)/cv2.contourArea(cv2.convexHull(c))
        except ZeroDivisionError:
            return 0

    def filter_fn(self, contours, minimal_area, solidity_limit):
        filtered_contours = [c for c in contours if
                             (cv2.contourArea(c) > minimal_area and
                              self.solidity(c) > solidity_limit)]

        # TODO: filter the ones that touch the border
        return filtered_contours

    def draw_contours(self, image, contours):
        background = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        return cv2.drawContours(background, contours, -1, (0, 255, 0), 3)
