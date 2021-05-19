import numpy as np
import numpy.testing
from pkg_resources import to_filename
import imageio as im

from ..process_opencv import TifReader
from ..spim import Spim

def test_tifreader():
    width = 1000
    height = 1000
    array = np.zeros([height, width, 3], dtype= np.uint8)
    array[:height//2,:width//2] = [255, 0, 0]
    array[:height//2,width//2:] = [0, 0, 255]   
    array[height//2:,:width//2] = [255, 100, 255] 
    array[height//2:,width//2:] = [100, 100, 255]
    compare_array = array[400:600, 400:550, :]
    filename = "testtif.tif"
    im.imwrite(filename, array)
    array = TifReader(filename).partial_read(filename, 20, 15, (40, 40))[0]
    numpy.testing.assert_array_almost_equal(array, compare_array)