class SpimStage:
    new = 0
    loaded = 1
    converted = 2
    preprocessed = 3
    binarized = 4
    postprocessed = 5
    features_extracted = 6
    features_filtered = 7
    analyzed = 8
    exported = 9


class Spim:
    """Spotlob image item"""

    def __init__(self, image=None, metadata=dict(), stage=SpimStage.new):
        self.image = image
        self.metadata = metadata
        self.stage = stage

    def read(self, reader):
        im, metadata = reader.apply()
        metadata.update(self.metadata)
        return Spim(im, metadata, SpimStage.loaded)

    def convert(self, converter):
        im = converter.apply(self.image)
        return Spim(im, self.metadata.copy(), SpimStage.converted)

    def preprocess(self, preprocessor):
        im = preprocessor.apply(self.image)
        return Spim(im, self.metadata.copy(), SpimStage.preprocessed)

    def binarize(self, binarizer):
        im = binarizer.apply(self.image)
        return Spim(im, self.metadata.copy(), SpimStage.binarized)

    def postprocess(self, postprocessor):
        im = postprocessor.apply(self.image)
        return Spim(im, self.metadata.copy(), SpimStage.postprocessed)

    def extract_features(self, feature_extractor):
        contours = feature_extractor.apply(self.image)
        metadata = {"contours": contours}
        metadata.update(self.metadata)
        return Spim(None, metadata, SpimStage.features_extracted)

    def filter_features(self, feature_filter):
        filtered_contours = feature_filter.apply(self.metadata["contours"])
        metadata = self.metadata.copy()
        metadata["contours"] = filtered_contours
        return Spim(None, metadata, SpimStage.features_filtered)

    def analyse(self, analysis):
        results = analysis.apply(self.metadata["contours"])
        metadata = self.metadata.copy()
        metadata["results"] = results
        return Spim(None, metadata, SpimStage.analyzed)
