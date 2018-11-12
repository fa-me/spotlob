from parameters import *
from spim import SpimStage
import numpy as np


class SpotlobProcessStep(object):
    input_stage = None

    """a process is outdated, if it has not been applied since the parameters changed"""
    outdated = True

    def __init__(self, function, parameters):
        self.function = function
        self.parameters = SpotlobParameterSet(parameters)

    def preview(self, spim):
        """This function takes spim at an undefined stage and draws the effect of the process on top, to provide a preview for the user on how the funciton will work. No storage or sideeffects should take place.
        In contrast to the apply_to function it must always return an image"""
        raise NotImplementedError("abstract: to be implemented by subclass")

    def apply(self, input):
        output = self.function(input, **self.parameters.to_dict())
        self.outdated = False
        return output


class Reader(SpotlobProcessStep):
    input_stage = SpimStage.new

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
    """apply returns grey image"""
    input_stage = SpimStage.loaded

    def preview(self, spim):
        input = spim.get_at_stage(self.input_stage).image
        return self.apply(input)


class Preprocessor(SpotlobProcessStep):
    """apply returns grey image"""
    input_stage = SpimStage.converted

    def preview(self, spim):
        input = spim.get_at_stage(self.input_stage).image
        return self.apply(input)


class Binarisation(SpotlobProcessStep):
    """apply returns binary image"""
    input_stage = SpimStage.preprocessed

    def preview(self, spim):
        input = spim.get_at_stage(self.input_stage).image
        return self.apply(input)


class Postprocessor(SpotlobProcessStep):
    """apply returns binary image"""
    input_stage = SpimStage.binarized

    def preview(self, spim):
        input = spim.get_at_stage(self.input_stage).image
        return self.apply(input)


class FeatureFinder(SpotlobProcessStep):
    """apply returns contours"""
    input_stage = SpimStage.postprocessed


class FeatureFilter(SpotlobProcessStep):
    """apply returns contours"""
    input_stage = SpimStage.features_extracted

    def preview(self, spim):
        input_ = spim.get_at_stage(self.input_stage)
        new_contours = self.apply(input_.metadata["contours"])
        return self.draw_contours(spim.image, new_contours)

    def draw_contours(self, image, contours):
        raise NotImplementedError("abstract: to be implemented by subclass")


class Analysis(SpotlobProcessStep):
    """apply returns pandas DataFrame"""
    input_stage = SpimStage.features_filtered
