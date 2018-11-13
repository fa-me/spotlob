from parameters import *
from spim import SpimStage
import numpy as np
import register


class SpotlobProcessStep(object):
    """A super class for process steps"""

    # stage that a spim should have before this Process can be applied
    input_stage = None

    # a process is outdated, if it has not been applied since the parameters changed
    outdated = True

    def __init__(self, function, parameters, add_to_register=True):
        self.function = function
        self.parameters = SpotlobParameterSet(parameters)

        if add_to_register:
            register.ProcessRegister.register_process(self)

    def preview(self, spim):
        """This function takes spim at an undefined stage and draws the effect of the process on top, to provide a preview for the user on how the funciton will work. No storage or sideeffects should take place.
        In contrast to the apply function it must always return an image"""
        raise NotImplementedError("abstract: to be implemented by subclass")

    def apply(self, input):
        output = self.function(input, **self.parameters.to_dict())
        self.outdated = False
        return output


class Reader(SpotlobProcessStep):
    input_stage = SpimStage.new

    def apply(self):
        return self.function(**self.parameters.to_dict())

    # def preview(self, spim):
    #     # preview does not make sense here
    #     pass


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
        binim = self.apply(input)
        #bin_masked = ~np.ma.masked_array(binim, binim == 0)
        # return bin_masked
        return binim


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

        background = spim.get_at_stage(SpimStage.converted).image
        return self.draw_contours(background, new_contours)

    def draw_contours(self, image, contours):
        raise NotImplementedError("abstract: to be implemented by subclass")


class Analysis(SpotlobProcessStep):
    """apply returns pandas DataFrame"""
    input_stage = SpimStage.features_filtered

    def preview(self, spim):
        contours = spim.metadata["contours"]
        df = self.apply(contours)

        im = spim.get_at_stage(SpimStage.loaded).image
        return self.draw_results(im, df)

    def draw_results(self, image, dataframe):
        raise NotImplementedError("abstract: to be implemented by subclass")
