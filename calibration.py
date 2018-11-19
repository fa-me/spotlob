import json


class Calibration(object):

    def __init__(self, *args, **kwargs):
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
                calibration_file = "calibrations.json"

            with open(calibration_file, "rb") as f_:
                self.caldict = json.load(f_)

            self.pxPerMicron = float(self.caldict[microscope_name][objective])

    def pixel2micron(self, pixels):
        return pixels / self.pxPerMicron

    def micron2pixel(self, microns):
        return microns * self.pxPerMicron
