import numpy as np
import pandas as pd
from skimage.measure import regionprops


def labels_to_positions_df(label_stack, spacing=(1.0, 1.0, 1.0), time_axis=0):
    """
    Converts a label image stack into a positions DataFrame required for napari tracks.

    Args:
        label_stack (numpy.ndarray): 3D or 4D numpy array where each frame is a labeled image.
            Shape can be (T, Z, Y, X) or (T, Y, X) depending on whether it's 3D or 2D data.
        spacing (tuple): Physical spacing between pixels in each dimension (Z, Y, X).
        time_axis (int): The axis corresponding to time in the label_stack.

    Returns:
        positions_df (pd.DataFrame): DataFrame with columns:
            - 'cluster_id'
            - 'frame'
            - 'z' (if applicable)
            - 'y'
            - 'x'
            - Other region properties as needed
    """
    # Initialize an empty list to collect data
    data = []

    # Move the time axis to the first position
    label_stack = np.moveaxis(label_stack, time_axis, 0)

    # Determine if data is 2D or 3D
    is_3d = label_stack.ndim == 4  # (T, Z, Y, X)

    # Iterate over frames
    for t, frame_labels in enumerate(label_stack):
        if is_3d:
            # For 3D data, frame_labels shape is (Z, Y, X)
            regions = regionprops(frame_labels)
            for region in regions:
                centroid = region.centroid  # (z, y, x)
                cluster_id = region.label
                data.append(
                    {
                        "cluster_id": cluster_id,
                        "frame": t,
                        "z": centroid[0] * spacing[0],
                        "y": centroid[1] * spacing[1],
                        "x": centroid[2] * spacing[2],
                        # Add other properties if needed
                    }
                )
        else:
            # For 2D data, frame_labels shape is (Y, X)
            regions = regionprops(frame_labels)
            for region in regions:
                centroid = region.centroid  # (y, x)
                cluster_id = region.label
                data.append(
                    {
                        "cluster_id": cluster_id,
                        "frame": t,
                        "y": centroid[0] * spacing[1],
                        "x": centroid[1] * spacing[2],
                        # Add other properties if needed
                    }
                )

    # Create DataFrame
    positions_df = pd.DataFrame(data)

    # Reorder columns
    columns = ["cluster_id", "frame"]
    if is_3d:
        columns += ["z", "y", "x"]
    else:
        columns += ["y", "x"]
    positions_df = positions_df[columns]

    return positions_df


def tracker_to_napari_tracks(
    tracker,
    positions_df=None,
    label_stack=None,
    spacing=(1.0, 1.0, 1.0),
    time_axis=0,
    position_cols=None,
):
    """
    Generates data for napari tracks layer from a LineageTracker instance.

    Args:
        tracker (LineageTracker): An instance of LineageTracker.
        positions_df (pd.DataFrame, optional): DataFrame containing positional data with columns:
            - 'cluster_id'
            - 'frame' (or 'time')
            - position columns (e.g., 'x', 'y', 'z')
        label_stack (numpy.ndarray, optional): Label image stack with cluster IDs as labels.
            Provide either positions_df or label_stack.
        spacing (tuple): Physical spacing between pixels in each dimension (Z, Y, X).
            Used if label_stack is provided.
        time_axis (int): The axis corresponding to time in the label_stack.
            Used if label_stack is provided.
        position_cols (list, optional): List of column names for spatial coordinates.
            If None, defaults to ['z', 'y', 'x'] for 3D or ['y', 'x'] for 2D data.

    Returns:
        tuple: (data, properties, graph)
            - data: NumPy array of shape (N, D+2)
            - properties: Dictionary mapping property names to arrays of shape (N,)
            - graph: Dictionary mapping track IDs to lists of child track IDs
    """
    # Check if positions_df is provided, else generate it from label_stack
    if positions_df is None:
        if label_stack is None:
            raise ValueError(
                "Either positions_df or label_stack must be provided."
            )
        positions_df = labels_to_positions_df(label_stack, spacing, time_axis)

    # Determine if data is 2D or 3D based on positions_df columns
    if position_cols is None:
        if "z" in positions_df.columns:
            position_cols = ["z", "y", "x"]
        else:
            position_cols = ["y", "x"]

    # Filter positions_df to include only clusters in the tracker
    valid_cluster_ids = set(tracker.nodes.keys())
    positions_df = positions_df[
        positions_df["cluster_id"].isin(valid_cluster_ids)
    ].copy()

    # Assign track IDs (cluster IDs)
    positions_df["track_id"] = positions_df["cluster_id"]

    # Collect data for napari tracks
    # The data array has columns: [track_id, frame, z, y, x] or [track_id, frame, y, x]
    data_cols = ["track_id", "frame"] + position_cols
    positions_df["frame"] = positions_df["frame"] - positions_df["frame"].min()
    data = positions_df[data_cols].to_numpy()

    # Collect properties
    properties = {}
    for col in positions_df.columns:
        if col not in data_cols:
            properties[col] = positions_df[col].to_numpy()

    # Build the graph
    graph = {}
    for node in tracker.nodes.values():
        parent_id = node.cluster_id
        child_ids = [child.cluster_id for child in node.children]
        if child_ids:
            graph[parent_id] = child_ids

    return data, properties, graph
