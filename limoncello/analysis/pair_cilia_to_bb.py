# pair_cilia_to_bb.py

import numpy as np
import pandas as pd
import re

def parse_coords_string(series):
    """
    Convert a pandas series of coordinate values into a numeric array
    (n_objects, 3). Handles already-parsed lists/arrays as well as string
    reprs in either clean ('[10.5, 428.1, 11.57]') or numpy-wrapped
    ('[np.float64(10.52), ...]') form.
    """
    num_pattern = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")
    coords_list = []
    for s in series:
        if isinstance(s, (list, tuple, np.ndarray)):
            coords_list.append([float(x) for x in s])
            continue
        # Strip numpy wrappers then pull out every number
        numbers = num_pattern.findall(str(s))
        coords_list.append([float(x) for x in numbers[:3]])
    return np.array(coords_list, dtype=np.float64)


def pair_from_mixed_df(df, voxel_size=(1.0, 1.0, 1.0), max_pair_distance_um=None):
    """
    Pair cilia and basal bodies from a single dataframe containing both.

    Assumes `coords` column is a numeric (z, y, x) array (shape (3,)) per row,
    in voxel index units.

    Stringent **closest, strict 1:1** assignment: every (cilium, basal body)
    pair distance is computed in physical µm (via ``voxel_size``), sorted
    ascending, and assigned greedily so each cilium takes only its single
    nearest still-available basal body and each basal body is used at most
    once. Pairs farther apart than ``max_pair_distance_um`` are rejected.

    Parameters
    ----------
    voxel_size : tuple of float
        Physical spacing (vz, vy, vx) in µm used to scale centroid distances.
    max_pair_distance_um : float or None
        Reject any pair beyond this µm distance. None = no cutoff.

    Returns
    -------
    pd.DataFrame
        Original dataframe with added columns:
        - paired_id          (id of the partner object, NaN if unpaired)
        - pair_distance_um   (µm; NaN if unpaired)
        - pairing_status     ("paired" / "lonely")
        - validated          (bool; True iff paired)
    """
    df = df.copy()

    # -------------------------
    # Split objects
    # -------------------------
    df_cilia = df[df["object_type"] == "cilia"].copy()
    df_basal = df[df["object_type"] == "basal_body"].copy()

    # Initialize outputs
    df["paired_id"] = np.nan
    df["pair_distance_um"] = np.nan
    df["pairing_status"] = "lonely"
    df["validated"] = False

    if len(df_cilia) == 0 or len(df_basal) == 0:
        # No possible pairs
        return df

    # -------------------------
    # Build coordinate arrays (voxel units) and a µm distance matrix
    # -------------------------
    cilia_coords = parse_coords_string(df_cilia["coords"])  # (n_cilia, 3)
    basal_coords = parse_coords_string(df_basal["coords"])  # (n_basal, 3)

    vs = np.asarray(voxel_size, dtype=float)
    dist_matrix = np.linalg.norm(
        (cilia_coords[:, None, :] - basal_coords[None, :, :]) * vs,
        axis=2,
    )  # µm

    # -------------------------
    # Greedy nearest, strict 1:1 assignment (closest pairs first)
    # -------------------------
    n_cilia, n_basal = dist_matrix.shape
    order = np.argsort(dist_matrix, axis=None)        # flat indices, ascending
    cilia_taken = np.zeros(n_cilia, dtype=bool)
    basal_taken = np.zeros(n_basal, dtype=bool)

    for flat in order:
        c, b = divmod(int(flat), n_basal)
        dist = dist_matrix[c, b]
        if max_pair_distance_um is not None and dist > max_pair_distance_um:
            break                                     # all remaining are farther
        if cilia_taken[c] or basal_taken[b]:
            continue
        cilia_taken[c] = basal_taken[b] = True

        cilia_row_idx = df_cilia.index[c]
        basal_row_idx = df_basal.index[b]
        cilia_id = df_cilia.loc[cilia_row_idx, "cilia_id"]
        basal_id = df_basal.loc[basal_row_idx, "cilia_id"]

        df.loc[cilia_row_idx, ["paired_id", "pair_distance_um", "pairing_status", "validated"]] = [
            basal_id, dist, "paired", True
        ]
        df.loc[basal_row_idx, ["paired_id", "pair_distance_um", "pairing_status", "validated"]] = [
            cilia_id, dist, "paired", True
        ]

    return df