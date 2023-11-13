import numpy as np
import pytest
from qtpy import QtCore
from arcospx import make_sample_data, remove_background, track_events
import napari
from skimage.io import imread
from numpy.testing import assert_array_equal
from qtpy.QtCore import QTimer
from napari.layers.image import Image
from arcos4py.tools import remove_image_background, track_events_image

# def test_remove_background():
#     """
#     Test background removal on a simple image.
#     """
#     viewer = napari.Viewer()
#     test_img = imread('test_data/1_growing.tif')
#     viewer.add_image(test_img, name='test_img')
#     true_img = imread('test_data/1_growing_true.tif')
#     removed_bg_img, _, _ = remove_background(image=viewer.layers['test_img'], filter_type="gaussian", size_0=1,
#                                              size_1=1, size_2=1)
#     assert_array_equal(removed_bg_img, true_img)

def test_remove_background(make_napari_viewer):
    """
    Test background removal on a simple image.
    """
    viewer = make_napari_viewer()
    test_img = imread('test_data/1_growing.tif')
    true_img = imread('test_data/1_growing_true.tif')

    viewer.add_image(test_img, name='test_img')

    # Create the widget from the factory
    remove_background_widget = remove_background()

    # Set the parameters for the widget
    remove_background_widget.image.value = viewer.layers['test_img']
    remove_background_widget.filter_type.value = "gaussian"
    remove_background_widget.size_0.value = 1
    remove_background_widget.size_1.value = 1
    remove_background_widget.size_2.value = 1

    # Execute the widget's function and get the worker
    worker = remove_background_widget()

    # Prepare to capture the result
    result = None

    def on_returned(value):
        nonlocal result
        result = value
        assert_array_equal(result[0], true_img)
        pytest.exit("Test completed")

    # Connect the returned signal of the worker to the on_returned function
    worker.returned.connect(on_returned)

# def test_track_events():
#     """
#     Test tracking on a simple image.
#     """
#     viewer = napari.Viewer()
#     test_img = imread('test_data/test_data_track_events.tif')
#     viewer.add_image(test_img, name='test_img')
#     true_img = imread('test_data/test_track_events_true.tif')
#     tracked_img,_,_ = track_events(viewer.layers['test_img'], threshold=300, eps=10, epsPrev=50, minClSz=50, minSamples=2, nPrev=2)
#     assert_array_equal(tracked_img, true_img)

