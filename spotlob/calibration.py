import json
import os.path

from pkg_resources import resource_filename


class Calibration(object):

    def __init__(self, *args, **kwargs):
        """
        Calibration(0.34) - 0.34 pixel per micrometer OR
        Calibration(microscope = "Zeiss", objective = "20x")
        """
        try:
            self.pxPerMicron = float(args[0])
        except ValueError:
            raise ValueError(
                "invalid argument %s: need to give numeric value for pxPerMicron" % args[0])
        except IndexError:
            microscope_name = kwargs.get("microscope")
            objective = kwargs.get("objective")

            if "calibration_file" in kwargs:
                calibration_file = kwargs.get("calibration_file")
            else:
                calibration_file = resource_filename(
                    "spotlob", "resources/calibrations.json")

            assert os.path.exists(calibration_file)

            caldict = Calibration.read_calibration_file(calibration_file)

            self.pxPerMicron = float(caldict[microscope_name][objective])

    def pixel_to_micron(self, pixels):
        return pixels / self.pxPerMicron

    def squarepixel_to_squaremicron(self, pixels_squared):
        return pixels_squared / self.pxPerMicron**2

    def micron_to_pixel(self, microns):
        return microns * self.pxPerMicron

    def squaremicron_to_squarepixel(self, microns_squared):
        return microns_squared * self.pxPerMicron**2

    @classmethod
    def read_calibration_file(cls, calibration_file="calibrations.json"):
        with open(calibration_file, "rb") as f_:
            caldict = json.load(f_)
        return caldict

    @classmethod
    def microscope_in_file(cls, calibration_file="calibrations.json"):
        caldict = Calibration.read_calibration_file(calibration_file)
        return caldict.keys()

    @classmethod
    def lenses_in_file(cls, microscope, calibration_file="calibrations.json"):
        caldict = Calibration.read_calibration_file(calibration_file)
        return caldict[microscope].keys()
