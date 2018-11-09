from memory_profiler import profile

from process_opencv import *
from spim import *


@profile
def run_simple_batch():
    reader = SimpleReader("test/testim.png")
    converter = GreyscaleConverter()
    preprocessor = GaussianPreprocess(3)
    binarization = BinaryThreshold(100)
    postprocessor = PostprocessNothing()
    feature_extractor = ContourFinderSimple()
    feature_filter = FeatureFormFilter(4000, 0.98)
    analysis = CircleAnalysis()

    spim = Spim(cached=True)

    return spim.read(self.reader) \
        .convert(self.converter) \
        .preprocess(self.preprocessor) \
        .binarize(self.binarization) \
        .postprocess(self.postprocessor)\
        .extract_features(self.feature_extractor)\
        .filter_features(self.feature_filter) \
        .analyse(self.analysis)

# run with
# python -m memory-profiler memprofile.py


if __name__ == '__main__':
    run_simple_batch()
