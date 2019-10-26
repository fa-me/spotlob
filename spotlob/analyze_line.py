import json

import cv2
import pandas as pd
import numpy as np

from .process_steps import Analysis
from .parameters import SpotlobParameterSet
from .calculation import points_within_contours, max_extends,\
    distance_point_to_line, straight_line_rectangle_collision, perp
from .process_opencv import draw_contours


class LineAnalysis(Analysis):
    def __init__(self, calibration=None,
                 linewidth_percentile=95,
                 extended_output=True):
        self.calibration = calibration
        self.linewidth_percentile = linewidth_percentile
        super(LineAnalysis, self).__init__(
            self.analyze, [], extended_output=extended_output)

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

        vx, vy, x0, y0 = cv2.fitLine(inner_points,
                                     cv2.DIST_FAIR, 0, 0.01, 0.01)
        # x0, y0, vx, vy
        line_parameters = x0[0], y0[0], vx[0], vy[0]

        # calculate crossings with borders
        width, height = max_extends(contours)

        image_rectangle = 0, 0, width, height
        p1, p2 = straight_line_rectangle_collision(line_parameters,
                                                   image_rectangle)

        distances = distance_point_to_line(inner_points[:, 0],
                                           inner_points[:, 1],
                                           p1,
                                           p2)

        linewidth_perc = np.percentile(distances, self.linewidth_percentile)*2

        line_length = np.linalg.norm(np.array(p2)-np.array(p1))
        linewidth_shading = area/line_length

        res_dict = {"area_px2": area,
                    "linewidth_px": linewidth_perc,
                    "linewidth_shading_px": linewidth_shading,
                    "line_params": [np.array(line_parameters)],
                    "line_start": [np.array(p1)],
                    "line_end": [np.array(p2)]}

        if self.extended_output:
            hist, bin_edges = np.histogram(distances, bins="auto")
            res_dict.update({"distances_hist": [hist],
                             "distances_bin_edges_px": [bin_edges],
                             "contours": [contours]})

        result = pd.DataFrame(res_dict,
                              index=[0])

        if not self.calibration:
            return result
        else:
            return self.calibration.calibrate(result)

    def draw_results(self, image, dataframe, crop_to_contours=False):
        if len(dataframe) == 1:
            row = dataframe.iloc[0]

            x0, y0, vx, vy = row["line_params"]
            linewidth = row["linewidth_px"]
            linewidth_shading = row["linewidth_shading_px"]

            if "contours" in dataframe:
                contour = row["contours"]
                draw_contours(image, contour)

            cstart = np.round(row["line_start"]).astype(int)
            cstop = np.round(row["line_end"]).astype(int)

            # center line
            cv2.line(image, tuple(cstart), tuple(cstop),
                     (255, 0, 0), 1, lineType=cv2.LINE_AA)
            cv2.circle(image, tuple(cstart), 4, (255, 0, 0), -1)
            cv2.circle(image, tuple(cstop), 4, (255, 0, 0), -1)

            # border lines

            # percentile linewidth
            self._draw_line_borders(image,
                                    linewidth,
                                    (255, 0, 0),
                                    cstart,
                                    cstop,
                                    (vx, vy))

            # # shading linewidth
            self._draw_line_borders(image,
                                    linewidth_shading,
                                    (0, 200, 200),
                                    cstart,
                                    cstop,
                                    (vx, vy))

        return image

    def _draw_line_borders(self,
                           image,
                           linewidth,
                           color,
                           center_start,
                           center_stop,
                           vector):

        # orthogonal vector to line vector with length linewidth
        # assuming vector is normalized to length 1
        orthogonal_v = np.round(perp(vector)*linewidth/2.0).astype(int)

        lower_start = tuple(np.subtract(center_start, orthogonal_v))
        lower_stop = tuple(np.subtract(center_stop, orthogonal_v))
        upper_start = tuple(np.add(center_start, orthogonal_v))
        upper_stop = tuple(np.add(center_stop, orthogonal_v))

        cv2.line(image, lower_start, lower_stop,
                 color, 2, lineType=cv2.LINE_AA)
        cv2.line(image, upper_start, upper_stop,
                 color, 2, lineType=cv2.LINE_AA)
