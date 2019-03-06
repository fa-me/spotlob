import json

import cv2
import pandas as pd
import numpy as np

from .process_steps import Analysis
from .parameters import SpotlobParameterSet
from .calculation import points_within_contours


class LineAnalysis(Analysis):
    def __init__(self, calibration=None):
        pars = SpotlobParameterSet([])
        self.calibration = calibration
        super(LineAnalysis, self).__init__(self.analyse, pars)

    def analyse(self, contours):
        inner_points = points_within_contours(contours)
        result = pd.DataFrame({"area_px": len(inner_points)})

        if not self.calibration:
            return result
        else:
            return self.calibration.calibrate(result)

    def draw_results(self, image, dataframe):
        # TODO: draw line(s) from dataframe onto image
        return image
