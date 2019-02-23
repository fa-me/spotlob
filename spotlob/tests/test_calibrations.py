from __future__ import absolute_import

import unittest
import os
import sys

from pkg_resources import resource_filename

import pandas as pd
import numpy

from ..analyse_opencv import CircleAnalysis
from ..calibration import Calibration


class TestCalibration(unittest.TestCase):
    def test_create_numeric(self):
        mycal = Calibration(3.0)
        mycal.pixel_to_micron(1)

    def test_create_fromFile(self):
        mycal = Calibration(microscope="zeiss", objective="20x")

        mycal.pixel_to_micron(1)

    def test_create_from_custom_file(self):
        calibration_filename = resource_filename(
            "spotlob.tests", "resources/test_calibrations.json")
        mycal = Calibration(
            microscope="zeiss", objective="20x",
            calibration_file=calibration_filename)

        self.assertTrue(mycal.micron_to_pixel(1) == 99)

    def test_micron_to_pixel(self):
        calibration_filename = resource_filename(
            "spotlob.tests", "resources/test_calibrations.json")
        mycal = Calibration(
            microscope="zeiss", objective="20x",
            calibration_file=calibration_filename)

        self.assertTrue(mycal.micron_to_pixel(1) == 99)
        self.assertTrue(mycal.micron_to_pixel(2) == 2*99)

        mycal2 = Calibration(
            microscope="zeiss", objective="50x",
            calibration_file=calibration_filename)

        self.assertTrue(mycal2.micron_to_pixel(1) == 100)
        self.assertTrue(mycal2.micron_to_pixel(2) == 200)

    def test_pixel_to_microns(self):
        calibration_filename = resource_filename(
            "spotlob.tests", "resources/test_calibrations.json")
        mycal = Calibration(
            microscope="zeiss", objective="50x",
            calibration_file=calibration_filename)

        self.assertTrue(mycal.pixel_to_micron(1) == 0.01)
        self.assertTrue(mycal.pixel_to_micron(2) == 0.02)

    def test_analyse_dataframe(self):
        data = pd.DataFrame({"area_px2": [100, 10000], "radius_px": [10, 20]})
        cal = Calibration(10)

        analysis = CircleAnalysis(cal)
        df_cal = analysis.calibrate(data)

        numpy.testing.assert_array_equal(
            df_cal["area_um2"], numpy.array([1, 100]))
        numpy.testing.assert_array_equal(
            df_cal["radius_um"], numpy.array([1, 2]))
