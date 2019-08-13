"""For every function of `Spim` that returns another `Spim` at
a further stage, there is a subclass of `SpotlobProcessStep`,
that can be used as super class for an concrete implementation
of that step. The return type of a `SpotlobProcessStep.apply`
call is different depending on the type of process. The Spim
internally passes the modified data to the new Spim created
through the process.
"""

from .spim import Spim, SpimStage
from .process import SpotlobProcessStep


class Reader(SpotlobProcessStep):
    """A reader loads the image data from storage into memory.
    `apply` returns an image"""
    input_stage = SpimStage.new


class Converter(SpotlobProcessStep):
    """A converter converts a color image to a greyscale image.
    `apply` returns a greyscale image
    """
    input_stage = SpimStage.loaded

    def preview(self, spim):
        input = spim.get_at_stage(self.input_stage).image
        return self.apply(input)


class Preprocessor(SpotlobProcessStep):
    """Preprocessing is applied onto a grey image to prepare for
    binarization, for example by cleaning the image from unwanted
    features.
    `apply` returns a greyscale image
    """
    input_stage = SpimStage.converted

    def preview(self, spim):
        input = spim.get_at_stage(self.input_stage).image
        return self.apply(input)


class Binarization(SpotlobProcessStep):
    """Turns a greyscale image into a black-and-white or binary image.
    `apply` returns a greyscale image
    """
    input_stage = SpimStage.preprocessed

    def preview(self, spim):
        input = spim.get_at_stage(self.input_stage).image
        binim = self.apply(input)
        # bin_masked = ~np.ma.masked_array(binim, binim == 0)
        # return bin_masked
        return binim


class Postprocessor(SpotlobProcessStep):
    """Postprocessing is done on a binary image to facilitate the
    detection.
    `apply` returns a greyscale image
    """
    input_stage = SpimStage.binarized

    def preview(self, spim):
        input = spim.get_at_stage(self.input_stage).image
        return self.apply(input)


class FeatureFinder(SpotlobProcessStep):
    """The FeatureFinder tries to find contours in a binary image.

    `apply` returns contours
    """
    input_stage = SpimStage.postprocessed

    def preview(self, spim):
        # do nothing, get at stage before feature find
        return spim.get_at_stage(self.input_stage).image


class FeatureFilter(SpotlobProcessStep):
    """The FeatureFilter can reduce the number of detected features
    by analyzing them.

    `apply` returns contours
    """
    input_stage = SpimStage.features_extracted

    def preview(self, spim):
        input_ = spim.get_at_stage(self.input_stage)
        new_contours = self.apply(input_.metadata["contours"],
                                  input_.metadata["image_shape"])

        background = spim.get_at_stage(SpimStage.binarized).image
        return self.draw_contours(background, new_contours)

    def draw_contours(self, image, contours):
        raise NotImplementedError("abstract: to be implemented by subclass")


class Analysis(SpotlobProcessStep):
    """An Analysis class evaluates the metadata (including contours) and 
    yields its results as a dataframe.

    `apply` returns :class:`~pandas.DataFrame`"""
    input_stage = SpimStage.features_filtered

    def __init__(self,
                 function,
                 parameters,
                 add_to_register=True,
                 extended_output=True):
        """Abstract analysis process step. 

        Parameters
        ----------
        function : callable
            function that performs the analysis. As first argument it must
            take the metadata dict and further arguments must match the
            Parameters of the SpotlobParameterSet.
            Must return a pandas.Dataframe
        parameters : SpotlobParameterSet
            Parameters of this process steps
        add_to_register : bool, optional
            wether this should automatically be added to the list of known
            processes, by default True
        extended_output : bool, optional
            if the output of this Analysis should contain as much data as
            possible. If False, only most important results are returned. 
            By default True
        """
        self.extended_output = extended_output
        super(Analysis, self).__init__(function, parameters, add_to_register)

    def preview(self, spim):
        df = self.apply(spim.metadata)

        im = spim.get_at_stage(SpimStage.loaded).image
        return self.draw_results(im, df)

    def draw_results(self, image, dataframe, crop_to_contours=False):
        raise NotImplementedError("abstract: to be implemented by subclass")
