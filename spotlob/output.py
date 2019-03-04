"""
Store results and images from a detection into files
"""
import os.path

from PIL import Image

from .spim import Spim, SpimStage
from .process_opencv import draw_contours


class Writer(object):
    def __init__(self, image_filepath, data_filepath):
        self.image_filepath = image_filepath
        self.data_filepath = data_filepath

    def store_image(self, image):
        im = Image.fromarray(image)
        im.save(self.image_filepath)
        return self.image_filepath

    def store_data(self, dataframe):
        _, ext = os.path.splitext(self.data_filepath)

        if ext == ".csv":
            dataframe.to_csv(self.data_filepath)
        return self.data_filepath


def draw_contour_and_save(spim):
    """
    Loads the image file and draws the contour of
    detected features as saved in the metadata of the spim
    """
    filename = spim.metadata["filename"]
    contours = spim.metadata["results"]["contours"]
