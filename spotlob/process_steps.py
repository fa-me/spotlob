from .spim import Spim, SpimStage
from .process import SpotlobProcessStep


class Reader(SpotlobProcessStep):
    input_stage = SpimStage.new


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
        # bin_masked = ~np.ma.masked_array(binim, binim == 0)
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
        new_contours = self.apply(input_.metadata["contours"],
                                  input_.metadata["image_shape"])

        background = spim.get_at_stage(SpimStage.binarized).image
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
