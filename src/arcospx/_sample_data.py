"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations
from napari.layers import Image

import numpy

import numpy as np


def make_sample_data() -> Image:
    """Generates an image"""
    # Create a random 2D array with shape (height, width)
    data = np.random.rand(512, 512)

    # Return list of tuples
    # [(data1, add_image_kwargs1), (data2, add_image_kwargs2)]
    # Check the documentation for more information about the
    # add_image_kwargs
    # https://napari.org/stable/api/napari.Viewer.html#napari.Viewer.add_image

    # Specify the layer type as 'image'
    add_image_kwargs = {
        'name': 'Sample Image',
        'colormap': 'gray',
        'blending': 'opaque',
        'layer_type': 'image'
    }

    return (data, add_image_kwargs, "image")


