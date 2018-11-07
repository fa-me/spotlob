class Pipeline:
    reader = None
    reader_kwargs = None
    converter = None
    converter_kwargs = None
    preprocessor = None
    preprocessor_kwargs = None
    binarization = None
    binarization_kwargs = None
    postprocessor = None
    postprocessor_kwargs = None
    feature_extractor = None
    feature_extractor_kwargs = None
    feature_filter = None
    feature_filter_kwargs = None
    analysis = None
    analysis_kwargs = None
    exporter = None
    exporter_kwargs = None

    def __init__(self):
        pass

    def apply_to(self, spim):
        pass
