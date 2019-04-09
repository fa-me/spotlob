# Spotlob Change Log

All notable changes to this project will be documented in this file.

## 0.9.0
This version is a complete re-development from scratch.

- introduces spotlob image item "spim", to be used in an immutable way
- introduces pipeline
- introduces output via pandas dataframes
- introduces tests
- introduces parallel processing
- uses pandas for data handling
- support Python 3.7

A lot of features of former versions are removed, some of them will
be re-implemented in future versions.

- GUI via wxPython is removed
- exe compilation via py2exe is removed
- pickers, to select values from the image, are removed
- detection using background color model is removed
- manual removal of blobs is removed
- grouping of blobs is removed
- adding meta info is removed
- Python 2 support is cancelled