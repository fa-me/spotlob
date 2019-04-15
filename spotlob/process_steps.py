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
    """An Analysis class evaluates the contours and yields its results
    as a dataframe.

    `apply` returns :class:`~pandas.DataFrame`"""
    input_stage = SpimStage.features_filtered

    def preview(self, spim):
        contours = spim.metadata["contours"]
        df = self.apply(contours)

        im = spim.get_at_stage(SpimStage.loaded).image
        return self.draw_results(im, df)

    def draw_results(self, image, dataframe):
        raise NotImplementedError("abstract: to be implemented by subclass")
