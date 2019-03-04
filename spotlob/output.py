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

    def store_image(self, image, contours):
        image_w_contours = draw_contours(image, contours)
        im = Image.fromarray(image_w_contours)
        im.save(self.image_filepath)
        return self.image_filepath

    def store_data(self, dataframe):
        _, ext = os.path.splitext(self.data_filepath)

        if ext == ".csv":
            dataframe.to_csv(self.data_filepath)
        return self.data_filepath
