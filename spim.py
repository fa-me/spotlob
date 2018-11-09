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


class Spim:
    """Spotlob image item"""

    def __init__(self, image=None, metadata=dict(), stage=SpimStage.new, cached=False, predecessors=[]):
        self.image = image
        self.metadata = metadata
        self.stage = stage
        self.cached = cached
        self.predecessors = predecessors

    def read(self, reader):
        im, metadata = reader.apply()
        metadata.update(self.metadata)
        return Spim(im, metadata, SpimStage.loaded, self.cached, self._list_predecessors())

    def convert(self, converter):
        im = converter.apply(self.image)
        return Spim(im, self.metadata.copy(), SpimStage.converted, self.cached, self._list_predecessors())

    def preprocess(self, preprocessor):
        im = preprocessor.apply(self.image)
        return Spim(im, self.metadata.copy(), SpimStage.preprocessed, self.cached, self._list_predecessors())

    def binarize(self, binarizer):
        im = binarizer.apply(self.image)
        return Spim(im, self.metadata.copy(), SpimStage.binarized, self.cached, self._list_predecessors())

    def postprocess(self, postprocessor):
        im = postprocessor.apply(self.image)
        return Spim(im, self.metadata.copy(), SpimStage.postprocessed, self.cached, self._list_predecessors())

    def extract_features(self, feature_extractor):
        contours = feature_extractor.apply(self.image)
        metadata = {"contours": contours}
        metadata.update(self.metadata)
        return Spim(None, metadata, SpimStage.features_extracted, self.cached, self._list_predecessors())

    def filter_features(self, feature_filter):
        filtered_contours = feature_filter.apply(self.metadata["contours"])
        metadata = self.metadata.copy()
        metadata["contours"] = filtered_contours
        return Spim(None, metadata, SpimStage.features_filtered, self.cached, self._list_predecessors())

    def analyse(self, analysis):
        results = analysis.apply(self.metadata["contours"])
        metadata = self.metadata.copy()
        metadata["results"] = results
        return Spim(None, metadata, SpimStage.analyzed, self.cached, self._list_predecessors())

    def _list_predecessors(self):
        if self.cached:
            return self.predecessors + [self]
        else:
            return []

    def get_at_stage(self, spimstage):
        if not self.cached:
            raise Exception("Predecessors have not been cached")

        predecessors = self._list_predecessors()
        stages = [spim.stage for spim in predecessors]
        try:
            return predecessors[stages.index(spimstage)]
        except ValueError:
            raise IndexError("There is no predesessor at this stage")
