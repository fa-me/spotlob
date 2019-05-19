import json

import cv2
import pandas as pd
import numpy as np

from .process_steps import Analysis
from .parameters import SpotlobParameterSet
from .calculation import points_within_contours, max_extends,\
    distance_point_to_line


class LineAnalysis(Analysis):
    def __init__(self, calibration=None,
                 linewidth_percentile=95,
                 extended_output=True):
        pars = SpotlobParameterSet([])
        self.calibration = calibration
        self.linewidth_percentile = linewidth_percentile
        super(LineAnalysis, self).__init__(
            self.analyze, pars, extended_output=extended_output)

    def analyze(self, metadata):
        contours = metadata['contours']

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

        elif len(contours) == 1:
            inner_points = points_within_contours(contours)
        else:
            inner_points = np.vstack([points_within_contours([ctr])
                                      for ctr in contours])

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

        linewidth_perc = np.percentile(distances, self.linewidth_percentile)*2

        # TODO: use convex hull to combine contours
        # (bb_cx, bb_cy), (bb_w, bb_h), bb_angle = cv2.minAreaRect(contours[0])

        # linewidth_area = area/bb_h

        res_dict = {"area_px2": area,
                    "linewidth_px": linewidth_perc,
                    #  "linewidth2_px": linewidth_area,
                    "line_params": [np.array([x, y, vx, vy])]}

        if self.extended_output:
            hist, bin_edges = np.histogram(distances, bins="auto")
            res_dict.update({"distances_hist": [hist],
                             "distances_bin_edges_px": [bin_edges]})

            #  "bb_width_px": bb_w,
            #  "bb_height_px": bb_h,
            #  "bb_angle": bb_angle,

        result = pd.DataFrame(res_dict,
                              index=[0])

        if not self.calibration:
            return result
        else:
            return self.calibration.calibrate(result)

    def draw_results(self, image, dataframe):
        assert len(dataframe) == 1
        x0, y0, vx, vy = dataframe.loc[0, "line_params"]
        lw1h = dataframe.loc[0, "linewidth_px"]/2.0

        m = 1000

        cstart = np.array([x0[0]-m*vx[0], y0[0]-m*vy[0]]).astype(int)
        cstop = np.array([x0[0]+m*vx[0], y0[0]+m*vy[0]]).astype(int)

        # center line
        cv2.line(image, tuple(cstart), tuple(cstop),
                 (255, 0, 0), 1, lineType=cv2.LINE_AA)

        # calculate border lines
        orthogonal_v = (np.array([-vy[0], vx[0]])*lw1h).astype(int)

        lower_start = tuple(np.subtract(cstart, orthogonal_v))
        lower_stop = tuple(np.subtract(cstop, orthogonal_v))
        upper_start = tuple(np.add(cstart, orthogonal_v))
        upper_stop = tuple(np.add(cstop, orthogonal_v))

        cv2.line(image, lower_start, lower_stop,
                 (200, 0, 0), 2, lineType=cv2.LINE_AA)
        cv2.line(image, upper_start, upper_stop,
                 (200, 0, 0), 2, lineType=cv2.LINE_AA)

        return image
