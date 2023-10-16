import pytest
from arcospx import make_sample_data
import numpy as np
from qtpy import QtCore


def test_make_sample_data():
    """Test make_sample_data."""
    data = make_sample_data()

    # Check the shape of the numpy array
    assert data[0][0].shape == (512, 512)

    # Check if the first element of the tuple is a numpy array
    assert isinstance(data[0][0], np.ndarray)

    # Check if the second element of the tuple is a dictionary
    assert isinstance(data[0][1], dict)

