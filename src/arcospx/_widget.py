"""
This module containst the magic factory widgets for arcospx
"""

from time import sleep
from typing import Literal, Union

import numpy as np
from arcos4py.tools import estimate_eps as estimate_eps_func
from arcos4py.tools import remove_image_background
from arcos4py.tools._detect_events import ImageTracker, Linker
from magicgui import magic_factory
from napari.layers import Image
from napari.qt.threading import FunctionWorker, thread_worker
from napari.types import LayerDataTuple
from napari.utils import progress
from napari.utils.notifications import show_info
from skimage.filters import threshold_otsu

from arcospx.utils import (
    remap_label_image_to_lineage,
    tracker_to_napari_tracks,
)


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

    def _set_widget_worker(function_worker: FunctionWorker):
        if function_worker:
            widget.call_button.text = "Abort"
            function_worker.finished.connect(_reset_callbutton_name)
            function_worker.errored.connect(_reset_callbutton_name)
            widget.arcos_worker.value = function_worker

    def _on_eps_changed(value):
        if widget.estimate_eps.value == "Manual":
            widget.eps.enabled = True
        else:
            widget.eps.enabled = False

    def _on_stability_threshold_changed(value):
        if value > 0:
            widget.create_lineage_map.enabled = True
        else:
            widget.create_lineage_map.enabled = False

    def set_combobox_color(widget, color):
        qcombobox = widget.native
        qcombobox.setStyleSheet(f"QComboBox {{ background-color: {color}; }}")

    def _set_dims_as_text_label(value):
        if widget.image.value is not None:
            image_dims = widget.image.value.data.shape
            selected_dims = widget.dims.value
            # label the dimensions according to the selected Dims
            if len(selected_dims) != len(image_dims):
                show_info(
                    "Selected dimensions do not match the image dimensions. Please check the input data."
                )
                set_combobox_color(widget.dims, "red")
                return
            label_string = ", ".join(
                [
                    f"{lab}: {dat}"
                    for lab, dat in zip(
                        selected_dims, image_dims, strict=False
                    )
                ]
            )
            widget.dims.label = f"Dims\n{label_string}"
        else:
            widget.dims.label = "Dims\n(None)"
            set_combobox_color(widget.dims, "red")
        set_combobox_color(widget.dims, "green")

    widget.called.connect(_set_widget_worker)
    widget.estimate_eps.changed.connect(_on_eps_changed)
    widget.split_merge_stability.changed.connect(
        _on_stability_threshold_changed
    )
    widget.create_lineage_map.enabled = False
    widget.image.changed.connect(_set_dims_as_text_label)
    widget.dims.changed.connect(_set_dims_as_text_label)


@magic_factory(
    arcos_worker={"visible": False},
    widget_init=_on_track_events_init,
    call_button="Run",
    estimate_eps={
        "widget_type": "Combobox",
        "choices": ["Manual", "Mean", "Kneepoint"],
        "tooltip": "Estimate eps automatically. Tip: Mean is faster and works well for most cases, kneepoint should be more accurate.",
    },
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
        "tooltip": "Dimension order of input data. Spatial dimensions (X,Y,Z) affect parameter scaling.",
        "label": "Dims\n(None)",
    },
)
def track_events(
    image: Image,
    arcos_worker: Union[FunctionWorker, None] = None,
    estimate_eps: Literal["Manual", "Mean", "Kneepoint"] = "Manual",
    linking_method: Literal[
        "nearest_neighbors",
        "Sinkhorn",
    ] = "nearest_neighbors",
    eps: float = 1.5,
    eps_prev: float = 0,
    min_clustersize: int = 9,
    n_prev: int = 1,
    split_merge_stability: int = 0,
    downscale: int = 1,
    use_predictor: bool = False,
    remove_small_clusters: bool = True,
    create_tracks_layer: bool = True,
    create_lineage_map: bool = False,
    dims: Literal["TXY", "TYX", "TZXY", "ZTYX", "XY", "ZYX"] = "TXY",
) -> FunctionWorker[list[LayerDataTuple]]:
    if arcos_worker is not None and arcos_worker.is_running:
        arcos_worker.quit()
        show_info("Operation aborted by user.")
        return FunctionWorker(do_nothing_function)

    if image is None:
        show_info("No image selected.")
        worker = FunctionWorker(
            do_nothing_function
        )  # somehow this is needed to properly reset the call button, for sure there is a better way but i guess it works for now....
        worker.start()
        return worker

    # if the image is not binary throw an error
    if np.unique(image.data).size > 2:
        show_info("Input image must be binary.")
        worker = FunctionWorker(do_nothing_function)
        worker.start()
        return worker

    # Validate parameters
    if downscale < 1:
        show_info("Downscale factor must be ≥1.")
        worker = FunctionWorker(do_nothing_function)
        worker.start()
        return worker

    if min_clustersize < 1:
        show_info("Minimum cluster size must be ≥1.")
        worker = FunctionWorker(do_nothing_function)
        worker.start()
        return worker

    if estimate_eps == "Mean":
        eps = estimate_eps_func(
            image=image.data,
            n_neighbors=min_clustersize,
            plot=False,
            binarize_threshold=0,
            method="mean",
        )
    elif estimate_eps == "Kneepoint":
        eps = estimate_eps_func(
            image=image.data,
            n_neighbors=min_clustersize,
            plot=False,
            S=7,
            binarize_threshold=0,
            method="kneepoint",
            max_samples=10000,
            direction="increasing",
            interp_method="polynomial",
            polynomial_degree=7,
            online=True,
        )

    track_events.eps.value = eps

    pbar = progress(total=image.data.shape[0])

    @thread_worker(connect={"finished": pbar.close, "yielded": pbar.update})
    def track_events_worker() -> list[LayerDataTuple]:
        try:
            selected_image = image.data
            spatial_dims = sum(1 for c in dims.upper() if c in {"X", "Y", "Z"})

            # Adjust parameters for downscaling
            eps_adjusted = eps / downscale
            eps_prev_adjusted = eps_prev / downscale if eps_prev else None
            min_clustersize_adjusted = max(
                1, int(min_clustersize / (downscale**spatial_dims))
            )
            linkin_method = (
                "nearest"
                if linking_method == "nearest_neighbors"
                else "transportation"
            )

            linker = Linker(
                eps=eps_adjusted,
                eps_prev=eps_prev_adjusted,
                min_clustersize=min_clustersize_adjusted,
                linking_method=linkin_method,
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
                        "name": f"{image.name} tracked",
                        "metadata": meta,
                    },
                    "labels",
                )
            ]

            # Always generate tracks if any events detected
            if np.any(img_tracked > 0) and create_tracks_layer:
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
                            "name": f"{image.name} tracks",
                            "properties": properties,
                            "graph": graph,
                            "metadata": meta,
                        },
                        "tracks",
                    )
                )

            if create_lineage_map and split_merge_stability > 0:
                lineage_map = remap_label_image_to_lineage(
                    img_tracked, linker.lineage_tracker
                )
                layers.append(
                    (
                        lineage_map,
                        {
                            "name": f"{image.name} lineage map",
                            "metadata": meta,
                        },
                        "labels",
                    )
                )

            return layers
        except Exception as e:  # noqa: BLE001
            show_info(f"Error during tracking: {str(e)}")
            return layers
        finally:
            # Ensure the progress bar is closed even if an error occurs
            pbar.close()

    worker = track_events_worker()

    # Add error handling
    def handle_error(exc):
        show_info(f"Tracking failed: {str(exc)}")

    worker.errored.connect(handle_error)
    return worker


if __name__ == "__main__":
    from napari import Viewer

    viewer = Viewer()
    # viewer.window.add_dock_widget(remove_background())
    # viewer.window.add_dock_widget(thresholder())
    viewer.window.add_dock_widget(track_events())
    viewer.show(block=True)
