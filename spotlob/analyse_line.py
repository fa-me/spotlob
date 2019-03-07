import json

import cv2
import pandas as pd
import numpy as np

from .process_steps import Analysis
from .parameters import SpotlobParameterSet
from .calculation import points_within_contours, max_extends,\
    distance_point_to_line


class LineAnalysis(Analysis):
    def __init__(self, calibration=None, linewidth_percentile=95):
        pars = SpotlobParameterSet([])
        self.calibration = calibration
        self.linewidth_percentile = linewidth_percentile
        super(LineAnalysis, self).__init__(self.analyse, pars)

    def analyse(self, contours):
        inner_points = points_within_contours(contours)

        [vx, vy, x, y] = cv2.fitLine(inner_points,
                                     cv2.DIST_FAIR, 0, 0.01, 0.01)

        # calculate crossings with right/left border
        # TODO: what happens with perfectly vertical line
        _, extend = max_extends(contours)

        left_y = int((-x*vy/vx) + y)
        righty_y = int(((extend-x)*vy/vx)+y)
        p1 = (0, left_y)
        p2 = (extend, righty_y)

        distances = distance_point_to_line(inner_points[:, 0],
                                           inner_points[:, 1],
                                           p1,
                                           p2)

        linewidth = np.percentile(distances, self.linewidth_percentile)*2

        result = pd.DataFrame({"area_px": len(inner_points),
                               "linewidth_px": linewidth}, index=[0])

        if not self.calibration:
            return result
        else:
            return self.calibration.calibrate(result)

    def draw_results(self, image, dataframe):
        # TODO: draw line(s) from dataframe onto image
        return image
