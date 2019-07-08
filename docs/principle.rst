Priciple of operation
=====================

The Spim and its stages
-----------------------

.. automodule:: spotlob.spim
    :noindex:

The processes
-------------

.. automodule:: spotlob.process
    :noindex:

Any process step corresponds to one input stage and
one method of Spim to use it with

+--------------------+------------------+---------------------+
| input stage        | Spim method      | SpotlobProcessStep  |
|                    |                  | subclass            |
+====================+==================+=====================+
| new                | read             | Reader              |
+--------------------+------------------+---------------------+
| loaded             | convert          | Converter           |
+--------------------+------------------+---------------------+
| converted          | preprocess       | Preprocessor        |
+--------------------+------------------+---------------------+
| preprocessed       | binarize         | Binarization        |
+--------------------+------------------+---------------------+
| binarized          | postprocess      | Postprocessor       |
+--------------------+------------------+---------------------+
| postprocessed      | extract_features | FeatureExtractor    |
+--------------------+------------------+---------------------+
| features_extracted | filter_features  | FeatureFilter       |
+--------------------+------------------+---------------------+
| features_filtered  | analyze          | Analysis            |
+--------------------+------------------+---------------------+
| analyzed           | store            | Writer              |
+--------------------+------------------+---------------------+
| stored             |                  |                     |
+--------------------+------------------+---------------------+

.. automodule:: spotlob.process_steps
    :noindex:
.. autoclass:: spotlob.process_steps.Reader
    :noindex:
.. autoclass:: spotlob.process_steps.Converter
    :noindex:
.. autoclass:: spotlob.process_steps.Preprocessor
    :noindex:
.. autoclass:: spotlob.process_steps.Binarization
    :noindex:
.. autoclass:: spotlob.process_steps.Postprocessor
    :noindex:
.. autoclass:: spotlob.process_steps.FeatureFinder
    :noindex:
.. autoclass:: spotlob.process_steps.FeatureFilter
    :noindex:
.. autoclass:: spotlob.process_steps.Analysis
    :noindex:

The spotlob pipeline
--------------------

.. automodule:: spotlob.pipeline
    :noindex:
.. autoclass:: spotlob.pipeline.Pipeline
    :noindex:
