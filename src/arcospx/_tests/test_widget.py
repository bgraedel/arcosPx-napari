import numpy as np
import unittest

from arcospx import make_sample_data, remove_background, track_events
import unittest
from unittest.mock import Mock, patch
from typing import Tuple


class TestRemoveBackground(unittest.TestCase):

    @patch('arcospx.remove_background')
    def test_remove_background(self, mock_remove_background):
        # Mocking the Image class
        mock_image = Mock()
        mock_image.data = make_sample_data()
        mock_image.name = "test_image"

        # Mocking the remove_background function to return a value
        mock_remove_background.return_value = "removed_background_data"

        result = remove_background(mock_image)

        # Expected layer properties
        expected_layer_properties = {
            "name": "test_image background removed",
            "metadata": {
                "filter_type": "gaussian",
                "size_0": 20,
                "size_1": 5,
                "size_2": 5,
                "filename": "test_image",
            },
        }

        # Asserting the results
        self.assertEqual(result[0], "removed_background_data")
        self.assertEqual(result[1], expected_layer_properties)
        self.assertEqual(result[2], "image")

        # Asserting the remove_background function was called with the right arguments
        mock_remove_background.assert_called_once_with("image_data", "gaussian", (20, 5, 5))

if __name__ == '__main__':
    unittest.main()


# make_napari_viewer is a pytest fixture that returns a napari viewer object
# capsys is a pytest fixture that captures stdout and stderr output streams
# def test_example_q_widget(make_napari_viewer, capsys):
#     # make viewer and add an image layer using our fixture
#     viewer = make_napari_viewer()
#     viewer.add_image(np.random.random((100, 100)))
#
#     # create our widget, passing in the viewer
#     my_widget = ExampleQWidget(viewer)
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
