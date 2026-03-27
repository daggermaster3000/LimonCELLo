import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates

def assign_label_features(
    cilia_labels: np.ndarray,
    cilia_centroids: np.ndarray,
    cilia_ids: np.ndarray,
    dist_to_neurite: np.ndarray,
    nearest_skel_idx: np.ndarray,
    distance_map_neurites: np.ndarray,
    distance_map_nuclei: np.ndarray,
    map_ratio: np.ndarray,
    max_cilia_dist_cutoff_um: float,
    file: str,
    object_type: str = "cilia"
) -> pd.DataFrame:
    """
    Assign features to cilia objects based on centroids and distance maps.

    Parameters
    ----------
    cilia_centroids : np.ndarray
        Array of shape (n_objects, 3) with (z, y, x) coordinates.
    cilia_ids : np.ndarray
        Array of cilia labels corresponding to centroids.
    dist_to_neurite : np.ndarray
        Distance map to neurites (3D volume).
    nearest_skel_idx : np.ndarray
        Nearest skeleton indices array of shape (3, Z, Y, X).
    distance_map_neurites : np.ndarray
        Distance transform of neurites.
    distance_map_nuclei : np.ndarray
        Distance transform of nuclei.
    map_ratio : np.ndarray
        Ratio map.
    max_cilia_dist_cutoff_um : float
        Maximum distance allowed to consider assignment.
    file : str
        Filename for this set of cilia (used in DataFrame).
    cilia_labels_shape : tuple
        Shape of the cilia label volume (for clipping coordinates).
    object_type : str, optional
        Type of object, by default "cilia".

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per cilium that passed the distance threshold.
    """
    rows = []

    for cid, centroid in zip(cilia_ids, cilia_centroids):
        z, y, x = np.round(centroid).astype(int)

        # Clip to volume bounds
        z = np.clip(z, 0, cilia_labels.shape[0] - 1)
        y = np.clip(y, 0, cilia_labels.shape[1] - 1)
        x = np.clip(x, 0, cilia_labels.shape[2] - 1)

        # Distance to neurite at centroid
        d = map_coordinates(dist_to_neurite, np.array(centroid).reshape(3, 1), order=1)[0]

        if d > max_cilia_dist_cutoff_um:
            continue

        # Nearest skeleton voxel
        sz, sy, sx = nearest_skel_idx[:, z, y, x]

        dt_neurite = distance_map_neurites[sz, sy, sx]
        dt_nuclei = distance_map_nuclei[sz, sy, sx]
        ratio = map_ratio[sz, sy, sx]

        rows.append(
            {
                "filename": file,
                "cilia_id": cid,
                "coords": [centroid[0],centroid[1],centroid[2]], # z,y,x
                "distance_to_neurite_um": d,
                "ratio": ratio,
                "log_ratio": np.log(ratio),
                "dt_neurite": dt_neurite,
                "dt_nuclei": dt_nuclei,
                "log_dt_neurite": np.log(dt_neurite),
                "log_dt_nuclei": np.log(dt_nuclei),
                "object_type": object_type,
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        print(f"No valid {object_type} found in {file}")

    return df