import json

import cv2

from process import Analysis


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

        return pd.DataFrame({"area": areas,
                             "ellipse_position": ellipses_positions,
                             "ellipse_majorAxis": ellipses_majorAxes,
                             "ellipse_minorAxis": ellipses_minorAxes,
                             "ellipse_angle": ellipses_angles})

    def draw_results(self, image, dataframe):
        for index, row in dataframe.iterrows():
            x, y = row["ellipse_position"]
            MA = row["ellipse_majorAxis"]
            ma = row["ellipse_minorAxis"]
            angle = row["ellipse_angle"]

            xy = (int(x), int(y))
            e_size = (int(MA/2.0), int(ma/2.0))

            pen_color = [255, 0, 0]

            cv2.circle(image, xy, 10, pen_color, -1)
            cv2.ellipse(image, xy, e_size, angle, 0, 360, pen_color, 3)
        return image
