import json

import cv2
import pandas as pd
import numpy as np

from .process_steps import Analysis
from .parameters import SpotlobParameterSet


class CircleAnalysis(Analysis):
    def __init__(self, calibration=None):
        pars = SpotlobParameterSet([])
        self.calibration = calibration
        super(CircleAnalysis, self).__init__(self.analyse, pars)

    def analyse(self, contours):
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

            ellipses_positions += [e_pos]
            ellipses_major_axes += [e_major_ax]
            ellipses_minor_axes += [e_minor_ax]
            ellipses_angles += [angle]

        result = pd.DataFrame({"area_px2": areas,
                               "ellipse_position_px": ellipses_positions,
                               "ellipse_majorAxis_px": ellipses_major_axes,
                               "ellipse_minorAxis_px": ellipses_minor_axes,
                               "ellipse_angle": ellipses_angles})

        return self.calibrate(result)

    def draw_results(self, image, dataframe):
        for _, row in dataframe.iterrows():
            pos_x, pos_y = row["ellipse_position_px"]
            e_major = row["ellipse_majorAxis_px"]
            e_minor = row["ellipse_minorAxis_px"]
            angle = row["ellipse_angle"]

            e_pos = (int(pos_x), int(pos_y))
            e_size = (int(e_major/2.0), int(e_minor/2.0))

            pen_color = [255, 0, 0]

            cv2.circle(image, e_pos, 10, pen_color, -1)
            cv2.ellipse(image, e_pos, e_size, angle, 0, 360, pen_color, 3)
        return image

    def calibrate(self, dataframe):
        """if there is a calibration, get all columns that include the suffix _px and _px2 \
         and calculate additional columns with the micron / micron^2 values"""
        if not self.calibration:
            return dataframe
        else:
            for col in dataframe.columns:
                if col.endswith("_px"):
                    new_col_name = col[:-3] + "_um"
                    dataframe[new_col_name] = self.calibration.pixel_to_micron(
                        dataframe[col])
                elif col.endswith("_px2"):
                    new_col_name = col[:-4] + "_um2"
                    dataframe[new_col_name] = self.calibration.squarepixel_to_squaremicron(
                        dataframe[col])
            return dataframe
