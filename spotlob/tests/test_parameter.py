
import sys
sys.path.append("../")
import os

import unittest

import numpy as np

from spotlob.parameters import *


class TestParameter(unittest.TestCase):
    def test_float_parameter_from_spec(self):
        """
        float range: ("parameter_name", (float_value, float_min_value, float_max_value))
        """
        spec1 = ("parameter_name", (0, 10, 0.5))
        resultpar = parameter_from_spec(spec1)
        expectedPar = NumericRangeParameter(
            "parameter_name", 0.5, 0., 10., float)

        self.assertTrue(type(resultpar) == type(expectedPar))
        self.assertTrue(expectedPar.name == resultpar.name)
        self.assertTrue(expectedPar.value == resultpar.value)
        self.assertTrue(expectedPar.name == resultpar.name)
        self.assertTrue(expectedPar.name == resultpar.name)

    def test_int_parameter_from_spec(self):
        """
        int range: ("parameter_name", (int_value, int_min_value, int_max_value))
        """
        spec1 = ("parameter_name", (0, 10, 1))
        resultpar = parameter_from_spec(spec1)
        expectedPar = NumericRangeParameter(
            "parameter_name", 1, 0, 10, int)

        self.assertTrue(type(resultpar) == type(expectedPar))
        self.assertTrue(expectedPar.name == resultpar.name)
        self.assertTrue(expectedPar.value is resultpar.value)
        self.assertTrue(expectedPar.name == resultpar.name)
        self.assertTrue(expectedPar.name == resultpar.name)

    def test_bool_parameter_from_spec(self):
        """
        bool: ("parameter_name", bool)
        """
        spec1 = ("parameter_name", True)
        resultpar = parameter_from_spec(spec1)
        expectedPar = BoolParameter("parameter_name", True)

        self.assertTrue(type(resultpar) == type(expectedPar))
        self.assertTrue(expectedPar.name == resultpar.name)
        self.assertTrue(expectedPar.value == resultpar.value)

    def test_enum_parameter_from_spec(self):
        """
        enum: ("parameter_name", ["option1", ..., "optionN"])
        """
        choices = ["choice1", "choice2", "choice3"]

        spec1 = ("parameter_name", choices)
        resultpar = parameter_from_spec(spec1)
        expectedPar = EnumParameter("parameter_name", choices[0], choices)

        self.assertTrue(type(resultpar) == type(expectedPar))
        self.assertTrue(expectedPar.name == resultpar.name)
        self.assertTrue(expectedPar.value == resultpar.value)
        self.assertTrue(expectedPar.options == resultpar.options)
