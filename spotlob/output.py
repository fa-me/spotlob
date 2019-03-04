"""
Store results and images from a detection into files
"""

from .spim import Spim, SpimStage
from .process_opencv import draw_contours


class Writer(object):
    def __init__(self, image_filepath, data_filepath):
        self.image_filepath = image_filepath
        self.data_filepath = data_filepath

    def store_image(self, image):
        pass

    def store_data(self, dataframe):
        pass


def draw_contour_and_save(spim):
    """
    Loads the image file and draws the contour of
    detected features as saved in the metadata of the spim
    """
    filename = spim.metadata["filename"]
    contours = spim.metadata["results"]["contours"]
