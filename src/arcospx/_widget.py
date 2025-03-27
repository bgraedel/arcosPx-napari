"""
This module containst the magic factory widgets for arcospx
"""

from time import sleep
from typing import Literal, Union

import numpy as np
from arcos4py.tools import remove_image_background
from arcos4py.tools._detect_events import ImageTracker, Linker
from magicgui import magic_factory
from napari.layers import Image
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import LayerDataTuple
from napari.utils import progress
from napari.utils.notifications import show_info
from skimage.filters import threshold_otsu

from arcospx.utils import tracker_to_napari_tracks


def do_nothing_function():
    pass


@magic_factory()
def remove_background(
    image: Image,
    filter_type: Literal["gaussian", "median"] = "gaussian",
    size_0: int = 20,
    size_1: int = 5,
    size_2: int = 5,
    dims: str = "TXY",
    crop_time_axis: bool = False,
) -> FunctionWorker[LayerDataTuple]:
    size_tuple = (size_0, size_1, size_2)
    pbar = progress(total=0)

    @thread_worker(connect={"returned": pbar.close})
    def remove_image_background_2() -> LayerDataTuple:
        selected_image = image.data
        removed_background = remove_image_background(
            image=selected_image,
            filter_type=filter_type,
            size=size_tuple,
            dims=dims,
            crop_time_axis=crop_time_axis,
        )
        removed_background = np.clip(removed_background, 0, None)

        layer_properties = {"name": f"{image.name} background removed"}
        sleep(0.1)  # need this for the tests to pass ...
        return (removed_background, layer_properties, "image")

    return remove_image_background_2()


def _on_thresholder_init(widget):
    def update_slider(image: Image):
        min_value = np.min(image.data)
        max_value = np.max(image.data)
        widget.threshold.max = max_value
        widget.threshold.min = min_value
        widget.threshold.value = threshold_otsu(image.data)

    widget.image_selector.changed.connect(update_slider)
    if widget.image_selector.value is not None:
        update_slider(widget.image_selector.value)


@magic_factory(
    auto_call=True,
    threshold={"widget_type": "FloatSlider", "max": 1},
    widget_init=_on_thresholder_init,
)
def thresholder(
    image_selector: Image,
    threshold: float = 0.5,
) -> LayerDataTuple:
    binary_image = np.where(image_selector.data > threshold, 1, 0)
    return (binary_image, {"name": f"{image_selector.name} binary"}, "image")


def _on_track_events_init(widget):
    def _reset_callbutton_name():
        widget.call_button.text = "Run"

    def _set_widget_worker(funciton_worker):
        widget.arcos_worker.value = funciton_worker
        widget.call_button.text = "Abort"
        widget.arcos_worker.value.finished.connect(_reset_callbutton_name)

    widget.called.connect(_set_widget_worker)


@magic_factory(
    arcos_worker={"visible": False},
    widget_init=_on_track_events_init,
    call_button="Run",
    eps={
        "tooltip": "Clustering distance threshold (per frame). Adjusted for downscaling."
    },
    eps_prev={
        "tooltip": "Linking distance to previous frames. Set 0 to disable. Adjusted for downscaling."
    },
    min_clustersize={
        "tooltip": "Minimum cluster size (pixels). Adjusted for downscaling and dimensionality."
    },
    n_prev={"tooltip": "Number of previous frames to consider for linking."},
    split_merge_stability={
        "tooltip": "Minimum stable frames before allowing splits/merges (0=disable)."
    },
    downscale={
        "tooltip": "Downsampling factor for faster processing (>=1). Affects spatial parameters."
    },
    use_predictor={
        "tooltip": "Use motion prediction for better linking between frames."
    },
    remove_small_clusters={
        "tooltip": "Remove clusters smaller than min_clustersize after tracking."
    },
    dims={
        "tooltip": "Dimension order of input data. Spatial dimensions (X,Y,Z) affect parameter scaling."
    },
)
def track_events(
    image_selector: Image,
    arcos_worker: Union[FunctionWorker, None] = None,
    eps: float = 1.5,
    eps_prev: float = 0,
    min_clustersize: int = 9,
    n_prev: int = 1,
    split_merge_stability: int = 0,
    downscale: int = 1,
    use_predictor: bool = False,
    remove_small_clusters: bool = False,
    dims: Literal["TXY", "TYX", "TZXY", "ZTYX", "XY", "ZYX"] = "TXY",
) -> FunctionWorker[LayerDataTuple]:
    if arcos_worker is not None and arcos_worker.is_running:
        arcos_worker.quit()
        show_info("Operation aborted by user.")
        return FunctionWorker(do_nothing_function)

    # Validate parameters
    if downscale < 1:
        raise ValueError("Downscale must be ≥1")
    if min_clustersize < 1:
        raise ValueError("min_clustersize must be ≥1")

    pbar = progress(total=image_selector.data.shape[0])

    @thread_worker(connect={"finished": pbar.close, "yielded": pbar.update})
    def track_events_worker() -> list[LayerDataTuple]:
        try:
            selected_image = image_selector.data
            spatial_dims = sum(1 for c in dims.upper() if c in {"X", "Y", "Z"})

            # Adjust parameters for downscaling
            eps_adjusted = eps / downscale
            eps_prev_adjusted = eps_prev / downscale if eps_prev else None
            min_clustersize_adjusted = max(
                1, int(min_clustersize / (downscale**spatial_dims))
            )

            linker = Linker(
                eps=eps_adjusted,
                eps_prev=eps_prev_adjusted,
                min_clustersize=min_clustersize_adjusted,
                n_prev=n_prev,
                predictor=use_predictor,
                allow_merges=split_merge_stability > 0,
                allow_splits=split_merge_stability > 0,
                stability_threshold=split_merge_stability,
                remove_small_clusters=remove_small_clusters,
            )

            tracker = ImageTracker(linker, downscale)
            img_tracked = np.zeros_like(selected_image, dtype=np.uint16)

            for idx, timepoint in enumerate(
                tracker.track(selected_image, dims)
            ):
                img_tracked[idx] = timepoint
                yield 1

            # Prepare layer metadata
            meta = {
                "eps": eps,
                "eps_prev": eps_prev,
                "min_clustersize": min_clustersize,
                "n_prev": n_prev,
                "split_merge_stability": split_merge_stability,
                "downscale": downscale,
                "use_predictor": use_predictor,
                "remove_small_clusters": remove_small_clusters,
                "dims": dims,
                "spatial_dims": spatial_dims,
                "adjusted_eps": eps_adjusted,
                "adjusted_min_clustersize": min_clustersize_adjusted,
            }

            layers = [
                (
                    img_tracked,
                    {
                        "name": f"{image_selector.name} tracked",
                        "metadata": meta,
                    },
                    "labels",
                )
            ]

            # Always generate tracks if any events detected
            if np.any(img_tracked > 0):
                data, properties, graph = tracker_to_napari_tracks(
                    linker.lineage_tracker,
                    label_stack=img_tracked.astype(int),
                    spacing=(1.0, 1.0, 1.0),
                    time_axis=0,
                )
                layers.append(
                    (
                        data,
                        {
                            "name": f"{image_selector.name} tracks",
                            "properties": properties,
                            "graph": graph,
                            "metadata": meta,
                        },
                        "tracks",
                    )
                )

            return layers
        except Exception as e:
            show_info(f"Error during tracking: {str(e)}")
            raise

    worker = track_events_worker()

    # Add error handling
    def handle_error(exc):
        show_info(f"Tracking failed: {str(exc)}")

    worker.errored.connect(handle_error)
    return worker


if __name__ == "__main__":
    import napari
    from napari import Viewer

    viewer = Viewer()
    viewer.window.add_dock_widget(remove_background())
    viewer.window.add_dock_widget(thresholder())
    viewer.window.add_dock_widget(track_events())
    napari.run()
