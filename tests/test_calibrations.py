import unittest
import os
import sys

sys.path.append("../")

import numpy
import pandas as pd

import calibration
import analyse_opencv


class TestCalibration(unittest.TestCase):
    def test_create_numeric(self):
        mycal = calibration.Calibration(3.0)
        mycal.pixel_to_micron(1)

    def test_create_fromFile(self):
        mycal = calibration.Calibration(microscope="zeiss", objective="20x")

        mycal.pixel_to_micron(1)

    def test_create_fromCustomFile(self):
        mycal = calibration.Calibration(
            microscope="zeiss", objective="20x", calibration_file="tests/test_calibration.json")

        self.assertTrue(mycal.micron_to_pixel(1) == 99)

    def test_micron_to_pixel(self):
        mycal = calibration.Calibration(
            microscope="zeiss", objective="20x", calibration_file="tests/test_calibration.json")

        self.assertTrue(mycal.micron_to_pixel(1) == 99)
        self.assertTrue(mycal.micron_to_pixel(2) == 2*99)

        mycal2 = calibration.Calibration(
            microscope="zeiss", objective="50x", calibration_file="tests/test_calibration.json")

        self.assertTrue(mycal2.micron_to_pixel(1) == 100)
        self.assertTrue(mycal2.micron_to_pixel(2) == 200)

    def test_pixel_to_microns(self):
        mycal = calibration.Calibration(
            microscope="zeiss", objective="50x", calibration_file="tests/test_calibration.json")

        self.assertTrue(mycal.pixel_to_micron(1) == 0.01)
        self.assertTrue(mycal.pixel_to_micron(2) == 0.02)

    def test_analyse_dataframe(self):
        df = pd.DataFrame({"area_px2": [100, 10000], "radius_px": [10, 20]})
        cal = calibration.Calibration(10)

        analysis = analyse_opencv.CircleAnalysis(cal)
        df_cal = analysis.calibrate(df)

        numpy.testing.assert_array_equal(
            df_cal["area_um2"], numpy.array([1, 100]))
        numpy.testing.assert_array_equal(
            df_cal["radius_um"], numpy.array([1, 2]))
