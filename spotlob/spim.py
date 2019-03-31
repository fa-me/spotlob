import pandas


class SpimStage(object):
    """Enumeration of the stages that a Spim can go through"""

    new = 0
    loaded = 1
    converted = 2
    preprocessed = 3
    binarized = 4
    postprocessed = 5
    features_extracted = 6
    features_filtered = 7
    analyzed = 8
    stored = 9


class Spim(object):
    """Spotlob image item"""
    # TODO: describe nature of Spim, immutable concept

    def __init__(self, image, metadata, stage, cached, predecessors):
        """A Spim is a **Spotlob image item**, an object representing an image
        and the metadata that is collected along the process through a
        pipeline.

        Parameters
        ----------
        image : numpy array
            an image
        metadata : dict
            the data desribing the image and containing results
        stage : SpimStage
            the stage along the pipeline the image has passed
        cached : bool
            if this is true, a reference to predecessors of this Spim are
            stored and they are kept in memory. This is required if a process
            step is to be repeated
        predecessors : dict(SpimStage, Spim)
            a registry of predecessors of the current spim, stored alongside
            the stage they are in
        """

        self._image = image
        self.metadata = metadata
        self.stage = stage
        self.cached = cached
        self.predecessors = predecessors

        if image is not None:
            self.metadata.update({"image_shape": image.shape})

    @classmethod
    def from_file(cls, image_filepath, cached=False):
        """Create a Spim object from an image file. The path is stored in the
        Spim object, but the image is not yet loaded.

        Parameters
        ----------
        image_filepath : string
            Path to an image file. The image type must be understood by the
            reader that is given when the `read`-function is called. If an
            invalid image type is given at this stage, it will not be
            recognized
        cached : bool, optional
            If the spim is to be cached, a reference to predecessors will be
            kept and not be deleted by the garbage collector. This allows to
            go back to an earlier stage after applying processes, but is more
            memory consuming. (the default is False)

        Returns
        -------
        Spim
            An empty Spim at SpimStage.new, that does not contain any data
            except the filepath
        """

        md = {"filepath": image_filepath}
        return Spim(None,
                    md,
                    SpimStage.new,
                    cached=cached,
                    predecessors=dict())

    @property
    def image(self):
        """Gives the image contained in this Spim or in the latest
        predecessor, that has an image

        Raises
        ------
        Exception
            Exception is raised if no image is present, most likely
            because it has not been cached

        Returns
        -------
        numpy.array
            latest image
        """

        if not (self._image is None):
            return self._image
        elif self.cached:
            return self.predecessor_image()
        else:
            raise Exception("image not found, has not been cached")

    def predecessor_image(self):
        predecessor_stages = self.predecessors.keys()
        predecessor_stages = sorted(predecessor_stages)

        for i in predecessor_stages[::-1]:
            p = self.predecessors[i]
            if not (p.image is None):
                return p.image
        raise Exception("no image found")

    def read(self, reader):
        im, metadata = reader.apply(self.metadata["filepath"])
        metadata.update(self.metadata)
        metadata.update({"image_shape": im.shape})
        return Spim(im,
                    metadata,
                    SpimStage.loaded,
                    self.cached,
                    self._predecessors_and_self())

    def apply_process(self, process):
        assert self.stage == process.input_stage
        im = process.apply(self.image)
        return Spim(im,
                    self.metadata.copy(),
                    self.stage + 1,
                    self.cached,
                    self._predecessors_and_self())

    def convert(self, converter):
        return self.apply_process(converter)

    def preprocess(self, preprocessor):
        return self.apply_process(preprocessor)

    def binarize(self, binarizer):
        return self.apply_process(binarizer)

    def postprocess(self, postprocessor):
        return self.apply_process(postprocessor)

    def extract_features(self, feature_extractor):
        contours = feature_extractor.apply(self.image)
        new_metadata = self.metadata.copy()
        new_metadata.update({"contours": contours})
        newspim = Spim(None, new_metadata, SpimStage.features_extracted,
                       self.cached, self._predecessors_and_self())
        return newspim

    def filter_features(self, feature_filter):
        filtered_contours = feature_filter.apply(self.metadata["contours"],
                                                 self.metadata["image_shape"])
        metadata = self.metadata.copy()
        metadata["contours"] = filtered_contours
        return Spim(None,
                    metadata,
                    SpimStage.features_filtered,
                    self.cached,
                    self._predecessors_and_self())

    def analyse(self, analysis):
        results = analysis.apply(self.metadata["contours"])
        metadata = self.metadata.copy()
        metadata["results"] = results
        return Spim(None,
                    metadata,
                    SpimStage.analyzed,
                    self.cached,
                    self._predecessors_and_self())

    def store(self, writer):
        assert self.stage == SpimStage.analyzed

        metadata = self.metadata.copy()

        contours = metadata["contours"]

        fresh_image = self.get_at_stage(SpimStage.loaded).image
        image_path = writer.store_image(fresh_image, contours)
        data_path = writer.store_data(self.get_data())

        metadata["output_image_filepath"] = image_path
        metadata["output_data_path"] = data_path

        return Spim(None,
                    metadata,
                    SpimStage.stored,
                    self.cached,
                    self._predecessors_and_self())

    def func_at_stage(self, spimstage):
        """returns the function, that can be applied the given stage"""

        # TODO: the static map of functions should be defined elsewhere
        functions = [self.read,
                     self.convert,
                     self.preprocess,
                     self.binarize,
                     self.postprocess,
                     self.extract_features,
                     self.filter_features,
                     self.analyse,
                     self.store]
        return functions[spimstage]

    def do_process_at_stage(self, process):
        return self.func_at_stage(process.input_stage)(process)

    def _predecessors_and_self(self):
        if self.cached:
            outd = dict()
            for p_stage, p_spim in self.predecessors.items():
                if p_stage < self.stage:
                    outd.update({p_stage: p_spim})
            outd.update({self.stage: self})
            return outd
        else:
            return dict()

    def get_at_stage(self, spimstage):
        if spimstage == self.stage:
            return self
        else:
            try:
                return self.predecessors[spimstage]
            except KeyError:
                # TODO: check if cached = False, then predecessor cannot exist
                msg = "Spim has no predecessor at stage %s." % spimstage
                # TODO: more specific exception predecessor does not exist
                raise Exception(msg)

    def get_data(self):
        """get all metadata and results as flat metadata

        RETURNS:
        --------
            dict
                all metadata including collected results
        """
        if "results" in self.metadata.keys():
            results = self.metadata["results"]
            results["filename"] = self.metadata["filepath"]
            return results
        else:
            return pandas.DataFrame(self.metadata, index=[0])

    def __repr__(self):
        return "<Spim instance %s at stage %s>" % (id(self), self.stage)
