import numpy as np
import cv2


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


def max_extends(contours):
    max_extends = np.array([c.max(axis=0) for c in contours])
    return max_extends.max(axis=0)[0]


def points_within_contours(contours):
    """
    gives the indices of points within the given contours
    using the drawContours function
    """

    extend_x, extend_y = max_extends(contours)

    # get maximum points from contours to create mask image
    # of that shape that just covers all

    mask = np.zeros((extend_y, extend_x))

    # fill mask image
    cv2.drawContours(mask, contours, -1, 1, thickness=cv2.FILLED)

    mask = mask.astype(bool)

    yi, xi = np.indices((extend_y, extend_x))
    return np.vstack([xi[mask], yi[mask]]).T


def straight_line_rectangle_collision(line_parameters, rectangle):
    """Calculate the position of the two points where a line crosses the
    boundaries of a rectangle

    Parameters
    ----------
    line_parameters : list of float
        vx,vy, x0,y0
    rectangle : list of float
        corner_x, corner_y, width, height
    """
    x0, y0, vx, vy = line_parameters
    l1, l2 = np.array([x0, y0]), np.array([x0+vx, y0+vy])
    rx0, ry0, w, h = rectangle

    left_border = np.array([rx0, ry0]), np.array([rx0, ry0+h])
    right_border = np.array([rx0+w, ry0]), np.array([rx0+w, ry0+h])
    upper_border = np.array([rx0, ry0+h]), np.array([rx0+w, ry0+h])
    lower_border = np.array([rx0, ry0]), np.array([rx0+w, ry0])

    borders = [left_border, right_border, upper_border, lower_border]

    valid_intersects = []
    for b1, b2 in borders:
        x, y = get_intersection(l1, l2, b1, b2)

        if (rx0 <= x <= rx0+w) and (ry0 <= y <= ry0+h):
            valid_intersects.append(np.array([x, y]))

    assert len(valid_intersects) == 2

    return valid_intersects[0], valid_intersects[1]


def perp(a):
    """perpendicual 2d vector to given vector a"""
    b = np.empty_like(a)
    b[0] = -a[1]
    b[1] = a[0]
    return b


def get_intersection(a1, a2, b1, b2):
    """calculates the intersection point of two line segments

    Parameters
    ----------
    a1 : np.array
        endpoint 1 of segment 1
    a2 : np.array
        endpoint 2 of segment 1
    b1 : np.array
        endpoints 1 of segment 2
    b2 : np.array
        endpoints 2 of segment 2

    Returns
    -------
    np.array
        intersection point
    """
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2

    # line segment intersection using vectors
    # see Computer Graphics by F.S. Hill

    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = perp(da)
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float))*db + b1
