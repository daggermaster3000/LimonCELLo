# pair_cilia_to_bb.py

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
import re
import numpy as np

def parse_coords_string(series):
    """
    Convert a pandas series of strings like
    '[np.float64(10.52), np.float64(428.1), np.float64(11.57)]'
    into a proper numeric array (n_objects, 3)
    """
    coords_list = []
    float_pattern = re.compile(r"np\.float64\((.*?)\)")
    for s in series:
        # Find all numbers inside np.float64()
        numbers = float_pattern.findall(s)
        coords_list.append([float(x) for x in numbers])
    return np.array(coords_list, dtype=np.float64)


def pair_from_mixed_df(df):
    """
    Pair cilia and basal bodies from a single dataframe containing both.

    Assumes `coords` column is a numeric array (shape (3,)) for each row.

    Applies the Hungarian algorithm to the cost matrix containing euclidian distances between bb and cilia centroids
    i.e. creates a one to one mapping that minimizes the cost.

    Returns
    -------
    pd.DataFrame
        Original dataframe with:
        - paired_id
        - pairing_status ("paired" / "lonely")
        - pair_distance_um
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

    if len(df_cilia) == 0 or len(df_basal) == 0:
        # No possible pairs
        return df

    # -------------------------
    # Build coordinate arrays
    # -------------------------
    
    cilia_coords = parse_coords_string(df_cilia["coords"])  # shape (n_cilia, 3)
    basal_coords = parse_coords_string(df_basal["coords"])  # shape (n_basal, 3)

    # -------------------------
    # Distance matrix
    # -------------------------
    dist_matrix = np.linalg.norm(
        cilia_coords[:, None, :] - basal_coords[None, :, :],
        axis=2
    )
    
    dist_mask = dist_matrix.copy()


    # -------------------------
    # Hungarian assignment
    # -------------------------
    c_idx, b_idx = linear_sum_assignment(dist_mask)

    # -------------------------
    # Assign pairs
    # -------------------------
    for c, b in zip(c_idx, b_idx):
        if dist_mask[c, b] != np.inf:
            cilia_row_idx = df_cilia.index[c]
            basal_row_idx = df_basal.index[b]

            dist = dist_mask[c, b]
            cilia_id = df_cilia.loc[cilia_row_idx, "cilia_id"]
            basal_id = df_basal.loc[basal_row_idx, "cilia_id"]

            # Assign to both
            df.loc[cilia_row_idx, ["paired_id", "pair_distance_um", "pairing_status"]] = [
                basal_id, dist, "paired"
            ]
            df.loc[basal_row_idx, ["paired_id", "pair_distance_um", "pairing_status"]] = [
                cilia_id, dist, "paired"
            ]

    return df