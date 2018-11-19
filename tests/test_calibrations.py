
import sys
sys.path.append("../")
import os
import unittest

import numpy as np

import calibration


class TestCalibration(unittest.TestCase):
    def test_create_numeric(self):
        mycal = calibration.Calibration(3.0)

        mycal.pixel2micron(1)

    def test_create_fromFile(self):
        mycal = calibration.Calibration(microscope="zeiss", objective="20x")

        mycal.pixel2micron(1)

    def test_create_fromCustomFile(self):
        mycal = calibration.Calibration(
            microscope="zeiss", objective="20x", calibration_file="tests/test_calibration.json")

        self.assertTrue(mycal.micron2pixel(1) == 99)

    def test_micron2pixel(self):
        mycal = calibration.Calibration(
            microscope="zeiss", objective="20x", calibration_file="tests/test_calibration.json")

        self.assertTrue(mycal.micron2pixel(1) == 99)
        self.assertTrue(mycal.micron2pixel(2) == 2*99)

        mycal2 = calibration.Calibration(
            microscope="zeiss", objective="50x", calibration_file="tests/test_calibration.json")

        self.assertTrue(mycal2.micron2pixel(1) == 100)
        self.assertTrue(mycal2.micron2pixel(2) == 200)

    def test_pixel2microns(self):
        mycal = calibration.Calibration(
            microscope="zeiss", objective="50x", calibration_file="tests/test_calibration.json")

        self.assertTrue(mycal.pixel2micron(1) == 0.01)
        self.assertTrue(mycal.pixel2micron(2) == 0.02)
