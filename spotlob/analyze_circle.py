import json

import cv2
import pandas as pd
import numpy as np

from .process_steps import Analysis
from .parameters import SpotlobParameterSet
from .process_opencv import draw_contours, crop_to_contour


class CircleAnalysis(Analysis):
    def __init__(self, calibration=None, extended_output=True):
        self.calibration = calibration
        super(CircleAnalysis, self).__init__(
            self.analyze, [], extended_output=extended_output)

    def analyze(self, metadata):
        contours = metadata['contours']
        areas = []
        ellipses_positions = []
        ellipses_major_axes = []
        ellipses_minor_axes = []
        ellipses_angles = []

        for cont in contours:
            # AREA
            areas += [cv2.contourArea(cont)]

            # ELLIPSE
            try:
                e_pos, (e_major_ax, e_minor_ax), angle = cv2.fitEllipse(cont)
            except cv2.error:
                e_pos, (e_major_ax,
                        e_minor_ax), angle = np.nan, (np.nan, np.nan), np.nan

            ellipses_positions += [np.array(e_pos)]
            ellipses_major_axes += [e_major_ax]
            ellipses_minor_axes += [e_minor_ax]
            ellipses_angles += [angle]

        res_dict = {"area_px2": areas,
                    "ellipse_position_px": ellipses_positions,
                    "ellipse_majorAxis_px": ellipses_major_axes,
                    "ellipse_minorAxis_px": ellipses_minor_axes,
                    "ellipse_angle": ellipses_angles}

        if self.extended_output:
            res_dict.update({"contours": contours})

        result = pd.DataFrame(res_dict)

        if not self.calibration:
            return result
        else:
            return self.calibration.calibrate(result)

    def draw_results(self, image, dataframe, crop_to_contours=False):
        for _, row in dataframe.iterrows():
            pos_x, pos_y = row["ellipse_position_px"]
            e_major = row["ellipse_majorAxis_px"]
            e_minor = row["ellipse_minorAxis_px"]
            angle = row["ellipse_angle"]

            e_pos = (int(pos_x), int(pos_y))
            e_size = (int(e_major/2.0), int(e_minor/2.0))

            pen_color = [255, 0, 0]

            if "contours" in row.keys():
                contour = row["contours"]
                draw_contours(image, contour)

            cv2.circle(image, e_pos, 3, pen_color, -1)
            cv2.ellipse(image, e_pos, e_size, angle, 0, 360, pen_color, 1)

            if crop_to_contours:
                if "contours" in row.keys():
                    contour = row["contours"]
                    image = crop_to_contour(image, contour)

        return image
