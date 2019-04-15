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
        super(LineAnalysis, self).__init__(self.analyze, pars)

    def analyze(self, contours):
        if len(contours) == 0:
            empty_df = pd.DataFrame([], columns=["area_px2",
                                                 "linewidth_px",
                                                 "linewidth2_px",
                                                 "bb_width_px",
                                                 "bb_height_px",
                                                 "bb_angle",
                                                 "distances_hist",
                                                 "distances_bin_edges_px"])
            return empty_df
        elif len(contours) > 1:
            raise NotImplementedError("Currently only single contour is\
                                       supported for lines")

        inner_points = points_within_contours(contours)

        area = len(inner_points)

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

        hist, bin_edges = np.histogram(distances, bins="auto")

        linewidth_perc = np.percentile(distances, self.linewidth_percentile)*2

        # TODO: use convex hull to combine contours
        (bb_cx, bb_cy), (bb_w, bb_h), bb_angle = cv2.minAreaRect(contours[0])

        linewidth_area = area/bb_h

        result = pd.DataFrame({"area_px2": area,
                               "linewidth_px": linewidth_perc,
                               "linewidth2_px": linewidth_area,
                               "bb_width_px": bb_w,
                               "bb_height_px": bb_h,
                               "bb_angle": bb_angle}, index=[0])
                            #    "distances_hist": hist.tolist(),
                            #    "distances_bin_edges_px": bin_edges.tolist()

        if not self.calibration:
            return result
        else:
            return self.calibration.calibrate(result)

    def draw_results(self, image, dataframe):
        # TODO: draw line(s) from dataframe onto image
        return image
