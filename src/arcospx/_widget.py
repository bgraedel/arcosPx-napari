"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from arcos4py.tools import remove_image_background, track_events_image
from magicgui import magic_factory, widgets
from napari import viewer
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from napari.types import LayerDataTuple
from napari.layers import Image
from napari.qt.threading import thread_worker, FunctionWorker
from napari.utils import progress
from magicgui import magicgui


if TYPE_CHECKING:
    import napari

# A global flag for aborting the process
abort_flag = False

# Optional: Define a custom exception for aborting the process
class AbortException(Exception):
    pass

@magic_factory()
def remove_background(
        image: Image,
        filter_type: str = "gaussian",
        size_0: int = 20,
        size_1: int = 5,
        size_2: int = 5,
        dims: str = "TXY",
        crop_time_axis: bool = False
    ) -> FunctionWorker[LayerDataTuple]:
    size = (size_0, size_1, size_2)
    pbar = progress(total=0)
    @thread_worker(connect={'returned': pbar.close})
    def remove_image_background_2() -> LayerDataTuple:
        global abort_flag

        # Reset the abort flag at the start of each execution
        abort_flag = False

        removed_background = remove_image_background(image.data, filter_type, size, dims, crop_time_axis)

        if abort_flag:
            # Return an error message
            # return "Interrupt error: Operation aborted by user."
            # Or raise a custom exception
            pbar.close()
            raise AbortException("Operation aborted by user.")

        layer_properties = {
            "name": f"{image.name} background removed",
            "metadata": {
                "filter_type": filter_type,
                "size_0": size_0,
                "size_1": size_1,
                "size_2": size_2,
                "dims": dims,
                "crop_time_axis": crop_time_axis,
                "filename": image.name,}}


        return (removed_background, layer_properties, "image")

    return remove_image_background_2()

@magic_factory()
def abort_process():
    global abort_flag
    abort_flag = True


@magic_factory()
def track_events(
    image_selector: Image,
    threshold: int = 300,
    eps: int = 10,
    epsPrev: int = 50,
    minClSz: int = 50,
    minSamples: int = 2,
    nPrev: int = 2,
    dims: str = "TXY",
) -> FunctionWorker[LayerDataTuple]:
    t_filter_size = 20
    pbar = progress(total=0)

    @thread_worker(connect={'returned': pbar.close})
    def track_events_2() -> LayerDataTuple:
        selected_image = image_selector.data
        img_tracked = track_events_image(selected_image >= threshold, eps = eps, epsPrev = epsPrev, minClSz = minClSz, minSamples = minSamples, nPrev = nPrev, dims = dims)

        # Like this we create the layer as a layer-data-tuple object which will automatically be parsed by napari and added to the viewer
        # This is more flexible and does not require the function to know about the viewer directly
        # Additionally like this you can now set the metadata of the layer
        layer_properties = {
            "name": f"{image_selector.name} tracked",
            "metadata": {
                "threshold": threshold,
                "eps": eps,
                "epsPrev": epsPrev,
                "minClSz": minClSz,
                "nPrev": nPrev,
                "filename": image_selector.name,  ## eg. setting medatada to the name of the image layer
         },}
        return (img_tracked, layer_properties, "labels")

    # return the layer data tuple
    return track_events_2()





