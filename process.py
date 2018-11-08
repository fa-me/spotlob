from parameters import *


class SpotlobProcessStep(object):
    def __init__(self, function, parameters):
        self.function = function
        self.parameters = SpotlobParameterSet(parameters)


class Reader(SpotlobProcessStep):
    def apply(self):
        return self.function(**self.parameters.to_dict())

    @classmethod
    def from_function(cls, reader_function):
        argspecs = inspect.getargspec(reader_function)

        try:
            assert len(argspecs.args) == 1
            assert type(argspecs.defaults[0]) == str
        except AssertionError:
            raise Exception(
                "Could not register function. Invalid signature for reader function %s" % reader_function.__name__)

        fpp = FilepathParameter("filepath", argspecs.defaults[0])
        return Reader(reader_function, [fpp])


class Converter(SpotlobProcessStep):
    def apply(self, image):
        """returns grey image"""
        return self.function(image, **self.parameters.to_dict())


class Preprocessor(SpotlobProcessStep):
    def apply(self, grey_image):
        """returns grey image"""
        return self.function(grey_image, **self.parameters.to_dict())


class Binarisation(SpotlobProcessStep):
    def apply(self, grey_image):
        """returns binary image"""
        return self.function(grey_image, **self.parameters.to_dict())


class Postprocessor(SpotlobProcessStep):
    def apply(self, bin_im):
        """returns binary image"""
        return self.function(bin_im, **self.parameters.to_dict())


class FeatureFinder(SpotlobProcessStep):
    def apply(self, bin_im):
        """returns contours"""
        return self.function(bin_im, **self.parameters.to_dict())


class FeatureFilter(SpotlobProcessStep):

    def apply(self, contours):
        """returns contours"""
        return self.function(contours, **self.parameters.to_dict())


class Analysis(SpotlobProcessStep):

    def apply(self, contours):
        """returns pandas DataFrame"""
        return self.function(contours, **self.parameters.to_dict())
