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

    def apply_to(self, spim):
        pass
