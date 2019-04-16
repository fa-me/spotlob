Default Pipeline
================

Spotlob comes with a predefined pipeline, which has some standard routines
built-in. It consists of the following process steps

1. :class:`~spotlob.process_opencv.SimpleReader`
2. :class:`~spotlob.process_opencv.GreyscaleConverter`
3. :class:`~spotlob.process_opencv.GaussianPreprocess`
4. :class:`~spotlob.process_opencv.OtsuThreshold` or
   :class:`~spotlob.process_opencv.BinaryThreshold`
5. :class:`~spotlob.process_opencv.PostprocessNothing`
6. :class:`~spotlob.process_opencv.ContourFinderSimple`
7. :class:`~spotlob.process_opencv.FeatureFormFilter`
8. :class:`~spotlob.process_opencv.CircleAnalysis` or
   :class:`~spotlob.process_opencv.LineAnalysis`

The default pipeline can be configured with parameters of returned by
the following function

.. autofunction:: spotlob.defaults.default_pipeline

