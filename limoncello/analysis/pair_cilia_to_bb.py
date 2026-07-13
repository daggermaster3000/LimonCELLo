# pair_cilia_to_bb.py

import math

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


def pair_from_mixed_df_directional(
    df,
    cilia_labels,
    voxel_size=(1.0, 1.0, 1.0),
    max_pair_distance_um=None,
    search_dist_um=None,
    cone_angle_deg=45.0,
):
    """Axis-directed cilia↔basal-body pairing with nearest-neighbour fallback.

    A basal body sits at the *base* of a cilium, roughly on the cilium's long
    axis. For each cilium we take its major axis (PCA of its label voxels) and,
    seeding from each **extremity**, look for a basal body inside a cone
    (half-angle ``cone_angle_deg``, length ``search_dist_um``) pointing outward
    along the axis. The nearest such basal body is the cilium's directional
    candidate. Candidates are assigned closest-first, strict 1:1
    (``pairing_method = "axis"``).

    Any cilium left unpaired (no basal body along its axis, or its candidate was
    taken) falls back to the original closest, strict 1:1 nearest-neighbour
    assignment within ``max_pair_distance_um`` (``pairing_method = "nearest"``),
    so this never pairs *fewer* cilia than :func:`pair_from_mixed_df`.

    Parameters
    ----------
    cilia_labels : ndarray
        The 3-D cilia label volume (ids match ``cilia_id`` in ``df``).
    search_dist_um : float or None
        Cone length (µm) for the axis search. Defaults to ``max_pair_distance_um``.
    cone_angle_deg : float
        Cone half-angle (deg) around the axis a basal body may deviate.

    Returns
    -------
    pd.DataFrame
        Same columns as :func:`pair_from_mixed_df` plus ``pairing_method``
        ("axis" / "nearest" / "none").
    """
    from ..utils.shape_props import cilia_axis_endpoints

    df = df.copy()
    df["paired_id"] = np.nan
    df["pair_distance_um"] = np.nan
    df["pairing_status"] = "lonely"
    df["validated"] = False
    df["pairing_method"] = "none"

    df_cilia = df[df["object_type"] == "cilia"].copy()
    df_basal = df[df["object_type"] == "basal_body"].copy()
    if len(df_cilia) == 0 or len(df_basal) == 0:
        return df

    vs = np.asarray(voxel_size, dtype=float)
    basal_coords = parse_coords_string(df_basal["coords"])          # voxel idx
    basal_um = basal_coords * vs                                     # µm
    basal_rows = list(df_basal.index)

    L = search_dist_um if search_dist_um is not None else max_pair_distance_um
    if L is None:
        L = np.inf
    cos_thr = math.cos(math.radians(float(cone_angle_deg)))

    axes = cilia_axis_endpoints(cilia_labels, voxel_size)

    c_taken = {i: False for i in df_cilia.index}
    b_taken = {i: False for i in df_basal.index}

    def _assign(cidx, bidx, dist, method):
        c_taken[cidx] = b_taken[bidx] = True
        cid = df_cilia.loc[cidx, "cilia_id"]
        bid = df_basal.loc[bidx, "cilia_id"]
        df.loc[cidx, ["paired_id", "pair_distance_um", "pairing_status",
                      "validated", "pairing_method"]] = [bid, dist, "paired", True, method]
        df.loc[bidx, ["paired_id", "pair_distance_um", "pairing_status",
                      "validated", "pairing_method"]] = [cid, dist, "paired", True, method]

    # ── 1) Directional candidates: each cilium's best BB along its axis ──────────
    candidates = []            # (dist_um, cilium_idx, basal_row_idx)
    for cidx, crow in df_cilia.iterrows():
        ax = axes.get(int(crow["cilia_id"]))
        if ax is None:
            continue
        best = None            # (dist, basal_position_in_basal_um)
        for end_um, out_dir in ((ax["end_a_um"], -ax["axis"]),
                                (ax["end_b_um"], ax["axis"])):
            vec = basal_um - end_um                    # (n_basal, 3) µm
            dist = np.linalg.norm(vec, axis=1)
            with np.errstate(divide="ignore", invalid="ignore"):
                cosang = np.where(dist > 0, (vec @ out_dir) / dist, -1.0)
            ok = (dist > 0) & (dist <= L) & (cosang >= cos_thr)
            if not ok.any():
                continue
            j = int(np.argmin(np.where(ok, dist, np.inf)))
            if best is None or dist[j] < best[0]:
                best = (float(dist[j]), j)
        if best is not None:
            candidates.append((best[0], cidx, basal_rows[best[1]]))

    candidates.sort(key=lambda t: t[0])
    for dist, cidx, bidx in candidates:
        if c_taken[cidx] or b_taken[bidx]:
            continue
        _assign(cidx, bidx, dist, "axis")

    # ── 2) Nearest-neighbour fallback over whatever is still unpaired ────────────
    c_idx = [i for i in df_cilia.index if not c_taken[i]]
    b_idx = [i for i in df_basal.index if not b_taken[i]]
    if c_idx and b_idx:
        cc = parse_coords_string(df_cilia.loc[c_idx, "coords"])
        bb = parse_coords_string(df_basal.loc[b_idx, "coords"])
        dmat = np.linalg.norm((cc[:, None, :] - bb[None, :, :]) * vs, axis=2)
        n_b = len(b_idx)
        for flat in np.argsort(dmat, axis=None):
            a, bpos = divmod(int(flat), n_b)
            dist = dmat[a, bpos]
            if max_pair_distance_um is not None and dist > max_pair_distance_um:
                break
            ci, bi = c_idx[a], b_idx[bpos]
            if c_taken[ci] or b_taken[bi]:
                continue
            _assign(ci, bi, float(dist), "nearest")

    return df