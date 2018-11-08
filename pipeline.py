class Pipeline:
    reader = None
    converter = None
    preprocessor = None
    binarization = None
    postprocessor = None
    feature_extractor = None
    feature_filter = None
    analysis = None

    def __init__(self):
        pass

    @property
    def processes(self):
        return [self.reader,
                self.converter,
                self.preprocessor,
                self.binarization,
                self.postprocessor,
                self.feature_extractor,
                self.feature_filter,
                self.analysis]

    def apply_to(self, spim):
        return spim.read(self.reader) \
            .convert(self.converter) \
            .preprocess(self.preprocessor) \
            .binarize(self.binarization) \
            .postprocess(self.postprocessor)\
            .extract_features(self.feature_extractor)\
            .filter_features(self.feature_filter) \
            .analyse(self.analysis)
