import json

import cv2

import process
import parameters


class CircleAnalysis(process.Analysis):
    def __init__(self, calibration=None):
        pars = parameters.SpotlobParameterSet([])
        self.calibration = calibration
        super(CircleAnalysis, self).__init__(self.analyse, pars)

    def analyse(self, contours):
        areas = []
        ellipses_positions = []
        ellipses_major_axes = []
        ellipses_minor_axes = []
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
            ellipses_major_axes += [MA]
            ellipses_minor_axes += [ma]
            ellipses_angles += [angle]

        return self.calibrate(pd.DataFrame({"area_px2": areas,
                                            "ellipse_position_px": ellipses_positions,
                                            "ellipse_majorAxis_px": ellipses_major_axes,
                                            "ellipse_minorAxis_px": ellipses_minor_axes,
                                            "ellipse_angle": ellipses_angles}))

    def draw_results(self, image, dataframe):
        for index, row in dataframe.iterrows():
            x, y = row["ellipse_position_px"]
            MA = row["ellipse_majorAxis_px"]
            ma = row["ellipse_minorAxis_px"]
            angle = row["ellipse_angle"]

            xy = (int(x), int(y))
            e_size = (int(MA/2.0), int(ma/2.0))

            pen_color = [255, 0, 0]

            cv2.circle(image, xy, 10, pen_color, -1)
            cv2.ellipse(image, xy, e_size, angle, 0, 360, pen_color, 3)
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
