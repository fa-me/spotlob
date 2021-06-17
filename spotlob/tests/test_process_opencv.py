import numpy as np
import numpy.testing
import imageio as im
from PIL import Image

from ..process_opencv import TifReader
from ..spim import Spim

def test_partial_read_1():
    width = 1000
    height = 1000
    array = np.zeros([height, width, 3], dtype= np.uint8)
    array[:height//2,:width//2] = [255, 0, 0]
    array[:height//2,width//2:] = [0, 0, 255]   
    array[height//2:,:width//2] = [255, 100, 255] 
    array[height//2:,width//2:] = [100, 100, 255]
    compare_array = array[400:550, 400:600, :]
    filename = "testtif.tif"
    im.imwrite(filename, array)
    array = TifReader().partial_read(filename, 20, 15, 40, 40)[0]
    numpy.testing.assert_array_almost_equal(array, compare_array)

def test_partial_read_2():
    width = 6000
    height = 6000
    width_percent, height_percent, x0_percent, y0_percent = 50, 30, 10, 30
    array = np.random.randint(255, size= (width, height, 3), dtype= np.uint8)
    compare_array = array[height*y0_percent//100:height*(y0_percent+height_percent)//100, 
                          width*x0_percent//100:width*(x0_percent+width_percent)//100, :]
    filename = "testtif.tif"
    im.imwrite(filename, array)
    array = TifReader().partial_read(filename, width_percent, height_percent, x0_percent, y0_percent)[0]
    numpy.testing.assert_array_almost_equal(array, compare_array)