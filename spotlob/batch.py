import multiprocessing as mp
import warnings

import pandas as pd

from .spim import Spim
from .pipeline import Pipeline


def is_interactive():
    """checks wether called in an interactive environment"""
    import __main__ as main
    return not hasattr(main, '__file__')


def _process_job(job):
    # function cannot be pickled,
    # so load function with dill within function call
    pipeline_file, image_file = job
    pipeline = Pipeline.from_file(pipeline_file)
    myspim = Spim.from_file(image_file)
    processed_spim = pipeline.apply_at_stage(myspim)
    return processed_spim.get_data()


def batchprocess(pipeline_file, image_files, multiprocessing=False):
    """This function applies a pipeline from a file onto a stack of images.
    The results are collected in one :class:`pandas.Dataframe`.

    PARAMETERS
    ----------
    pipeline_file : str
        the filepath of a pickled pipeline
    image_files : list of str
        paths of the images
    multiprocessing : bool, optional
        if True, the processing will be done in parallel using multiple cpu
        cores at once.
    
    RETURNS
    -------
    pandas.Dataframe
        Flat Dataframe where one row corresponds to one detected feature
    """
    if multiprocessing:
        if is_interactive():
            warnings.warn(
                """It seems you are running in an interactive environment.
                Multiprocessing might not work.
                Consider using ipyparallel instead""")

        jobs = zip([pipeline_file]*len(image_files), image_files)
        no_cores = mp.cpu_count()
        pool = mp.Pool(processes=no_cores)
        try:
            res = pool.map(_process_job, jobs)
        finally:
            pool.close()
            pool.join()

    else:
        pipeline = Pipeline.from_file(pipeline_file)

        def process_file(fn):
            myspim = Spim.from_file(fn)
            respim = pipeline.apply_at_stage(myspim)
            return respim.get_data()

        res = map(process_file, image_files)
    #
    # make large dataframe out of list of dataframes
    return pd.concat(res)
