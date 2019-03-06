import numpy as np


def distance_point_to_line(x0, y0, pt_on_line1, pt_on_line2):
    """
    Distance from point (x0, y0) to a line defined by two points
    pt_on_line1 and pt_on_line2
    see https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points # noqa
    """

    x1, y1 = pt_on_line1
    x2, y2 = pt_on_line2

    numerator = np.abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
    denom = np.sqrt((y2-y1)**2 + (x2-x1)**2)

    return numerator / denom
