import numpy as np
import pytest
from qtpy import QtCore
from arcospx import make_sample_data, remove_background, track_events
import napari
from skimage.io import imread
from numpy.testing import assert_array_equal
from napari.layers.image import Image

def test_remove_background():
    """Test background removal on a simple image."""
    test_img = imread('src/arcospx/_tests/test_data/1_growing.tif')
    test_img = Image(test_img)
    true_img = imread('src/arcospx/_tests/test_data/1_growing_true.tif')
    #test_img = np.where(test_img == 255, 0, 1)
    removed_bg_img = remove_background(test_img.data, filter_type="gaussian", size_0=1, size_1=1, size_2=1)
    assert_array_equal(removed_bg_img, true_img)


def test_track_events():
    """
    Test tracking on a simple image.
    """
    test_img = imread('test_data/test_data_track_events.tif')
    test_img = Image(test_img)
    true_img = imread('test_data/test_data_track_events_true.tif')
    #test_img = np.where(test_img == 255, 0, 1)
    tracked_img = track_events(test_img.data, threshold=300, eps=10, epsPrev=50, minClSz=50, nPrev=2)
    assert_array_equal(tracked_img, true_img)

# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
# def test_remove_background(make_napari_viewer, capsys):
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()
#     viewer.add_image(np.random.random((100, 100)))
#
#     # create our widget, passing in the viewer
#     my_widget = remove_background(viewer)
#
#     # call our widget method
#     my_widget._on_click()
#
#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == "napari has 1 layers\n"
#
#
# def test_example_magic_widget(make_napari_viewer, capsys):
#     viewer = make_napari_viewer()
#     layer = viewer.add_image(np.random.random((100, 100)))
#
#     # this time, our widget will be a MagicFactory or FunctionGui instance
#     my_widget = example_magic_widget()
#
#     # if we "call" this object, it'll execute our function
#     my_widget(viewer.layers[0])
#
#     # read captured output and check that it's as we expected
#     captured = capsys.readouterr()
#     assert captured.out == f"you have selected {layer}\n"
