import cv2
from process import *
import pandas as pd
import numpy as np


class SimpleReader(Reader):
    def __init__(self, default_filepath):
        pars = SpotlobParameterSet(
            [FilepathParameter("filepath", default_filepath)])
        super(SimpleReader, self).__init__(self.fn_read, pars)

    def fn_read(self, filepath):
        return cv2.imread(filepath), {"filepath": filepath}


class GreyscaleConverter(Converter):
    def __init__(self):
        converter_options = ["BGR to Grey", "RGB to Grey"]

        pars = SpotlobParameterSet(
            [EnumParameter("conversion", converter_options[0], converter_options)])
        super(GreyscaleConverter, self).__init__(self.convert, pars)

    def convert(self, image, conversion):
        if conversion == "BGR to Grey":
            code = cv2.COLOR_BGR2GRAY
        elif conversion == "RGB to Grey":
            code = cv2.COLOR_RGB2GRAY

        return cv2.cvtColor(image, code).astype(np.uint8)


class GaussianPreprocess(Preprocessor):
    def __init__(self, ksize):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("kernelsize", ksize, 1, 47, step=2)])
        super(GaussianPreprocess, self).__init__(self.filter_fn, pars)

    def filter_fn(self, grey_image, kernelsize):
        kernel = np.ones((kernelsize, kernelsize), np.float32)/kernelsize**2
        return cv2.filter2D(grey_image, -1, kernel)


class BinaryThreshold(Binarisation):
    def __init__(self, threshold):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("threshold", threshold, 0, 255)])
        super(BinaryThreshold, self).__init__(self.threshold_fn, pars)

    def threshold_fn(self, grey_image, threshold):
        thresh, im = cv2.threshold(
            grey_image, threshold, 255, cv2.THRESH_BINARY)
        return im


class PostprocessNothing(Postprocessor):
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(PostprocessNothing, self).__init__(self.postprocess_fn, pars)

    def postprocess_fn(self, im):
        return im


class ContourFinderSimple(FeatureFinder):
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(ContourFinderSimple, self).__init__(self.finder_fn, pars)

    def finder_fn(self, bin_im):
        cim, contours, _ = cv2.findContours(
            bin_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours


class FeatureFormFilter(FeatureFilter):
    def __init__(self, size, solidity, exclude_edge):
        pars = SpotlobParameterSet(
            [NumericRangeParameter("minimal_area", size, 0, 10000),
             NumericRangeParameter(
                 "solidity_limit", solidity, 0, 1, step=0.01, type_=float),
             BoolParameter("exclude_edge_touching", exclude_edge)
             ])
        super(FeatureFormFilter, self).__init__(self.filter_fn, pars)

    def solidity(self, c):
        try:
            return cv2.contourArea(c)/cv2.contourArea(cv2.convexHull(c))
        except ZeroDivisionError:
            return 0

    def filter_fn(self, contours, minimal_area, solidity_limit, exclude_edge_touching):
        contours = filter(lambda c: cv2.contourArea(c)
                          > minimal_area, contours)
        filtered_contours = filter(
            lambda c: self.solidity(c) > solidity_limit, contours)

        # TODO: filter the ones that touch the border
        return filtered_contours

    def draw_contours(self, image, contours):
        bg = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2RGB)
        return cv2.drawContours(bg, contours, -1, (0, 255, 0), 3)


class CircleAnalysis(Analysis):
    def __init__(self):
        pars = SpotlobParameterSet([])
        super(CircleAnalysis, self).__init__(self.analyse, pars)

    def analyse(self, contours):
        areas = []
        ellipses_positions = []
        ellipses_majorAxes = []
        ellipses_minorAxes = []
        ellipses_angles = []

        for c in contours:
            # AREA
            areas += [cv2.contourArea(c)]

            # ELLIPSE
            try:
                xy, (MA, ma), angle = cv2.fitEllipse(c)
            except:
                xy, (MA, ma), angle = np.nan, (np.nan, np.nan), np.nan

            ellipses_positions += [xy]
            ellipses_majorAxes += [MA]
            ellipses_minorAxes += [ma]
            ellipses_angles += [angle]

        # ADD METADATA
        return pd.DataFrame({"area": areas,
                             "ellipse_position": ellipses_positions,
                             "ellipse_majorAxis": ellipses_majorAxes,
                             "ellipse_minorAxis": ellipses_minorAxes,
                             "ellipse_angle": ellipses_angles})
