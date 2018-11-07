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

    def read(self, reader, reader_kwargs):
        im, metadata = reader(**reader_kwargs)
        metadata.update(self.metadata)
        return Spim(im, metadata, SpimStage.loaded)

    def convert(self, converter, converter_kwargs):
        im = converter(self.image, **converter_kwargs)
        return Spim(im, self.metadata.copy(), SpimStage.converted)

    def preprocess(self, preprocessor, preprocessor_kwargs):
        im = preprocessor(self.image, **preprocessor_kwargs)
        return Spim(im, self.metadata.copy(), SpimStage.preprocessed)

    def binarize(self, binarizer, binarizer_kwargs):
        im = binarizer(self.image, **binarizer_kwargs)
        return Spim(im, self.metadata.copy(), SpimStage.binarized)

    def postprocess(self, postprocessor, postprocessor_kwargs):
        im = postprocessor(self.image, **postprocessor_kwargs)
        return Spim(im, self.metadata.copy(), SpimStage.postprocessed)

    def extract_features(self, feature_extractor, feature_extractor_kwargs):
        contours = feature_extractor(self.image, **feature_extractor_kwargs)
        metadata = {"contours": contours}
        metadata.update(self.metadata)
        return Spim(None, metadata, SpimStage.features_extracted)

    def filter_features(self, feature_filter, feature_filter_kwargs):
        filtered_contours = feature_filter(
            self.metadata["contours"], **feature_filter_kwargs)
        metadata = self.metadata.copy()
        metadata["contours"] = filtered_contours
        return Spim(im, metadata, SpimStage.features_filtered)

    def analyse(self, analysis, analysis_kwargs):
        results = analysis(self.metadata["contours"], **analysis_kwargs)
        metadata = self.metadata.copy()
        metadata["results"] = results
        return Spim(im, metadata, SpimStage.analyzed)

    def export(self, exporter, export_kwargs):
        exporter(self.metadata, **export_kwargs)
        return Spim(im, self.metadata, SpimStage.exported)
