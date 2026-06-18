import numpy as np
from skimage.transform import resize


def make_isotropic(volume, voxel_size, target_spacing=None, order=1):
    """
    Resample a (Z, Y, X) volume to isotropic voxels.

    By default the target spacing is the *coarsest* axis spacing
    (``max(voxel_size)``, typically Z), so the finer axes are **downsampled**
    to match it. This keeps the voxel count from exploding (no upsampling) at
    the cost of in-plane resolution.

    Parameters
    ----------
    volume : np.ndarray
        Image volume (Z, Y, X).
    voxel_size : tuple of float
        Physical spacing per axis (vz, vy, vx) in µm.
    target_spacing : float, optional
        Isotropic spacing to resample to (µm). Defaults to ``max(voxel_size)``.
    order : int
        Interpolation order (1 = linear for intensities, 0 = nearest for labels).

    Returns
    -------
    (np.ndarray, tuple)
        The resampled volume and its new isotropic ``voxel_size`` ``(s, s, s)``.
        If the input is already isotropic (to within rounding), the volume is
        returned unchanged.
    """
    vol = np.asarray(volume)
    vs = tuple(float(v) for v in voxel_size)
    s = float(target_spacing) if target_spacing else max(vs)

    if all(abs(v - s) < 1e-6 for v in vs):
        return vol, (s, s, s)

    out_shape = tuple(max(1, int(round(dim * v / s))) for dim, v in zip(vol.shape, vs))
    downsampling = any(v < s for v in vs)            # anti-alias only when shrinking
    out = resize(
        vol, out_shape, order=order,
        anti_aliasing=downsampling, preserve_range=True,
    ).astype(vol.dtype, copy=False)
    return out, (s, s, s)


def percentile_minmax_normalize(
    data,
    p_low=1,
    p_high=99,
    per_channel=False,
    eps=1e-8

):
    """
    Percentile-based min–max normalization.

    Parameters
    ----------
    data : np.ndarray
        Input array (e.g., ZYX or TCZYX).
    p_low : float
        Lower percentile.
    p_high : float
        Upper percentile.
    per_channel : bool
        If True, normalize each channel separately
        (assumes channel axis = 1 for TCZYX).
    eps : float
        Small number to avoid division by zero.

    Returns
    -------
    np.ndarray
        Normalized array in range [0, 1].
    """

    data = data.astype(np.float32)

    if per_channel and data.ndim >= 4:
        # Assume channel axis = 1 (TCZYX style)
        norm = np.empty_like(data, dtype=np.float32)

        for c in range(data.shape[1]):
            channel_data = data[:, c]
            p1 = np.percentile(channel_data, p_low)
            p99 = np.percentile(channel_data, p_high)

            channel_norm = (channel_data - p1) / (p99 - p1 + eps)
            norm[:, c] = np.clip(channel_norm, 0, 1)

        return norm

    else:
        # Global normalization
        p1 = np.percentile(data, p_low)
        p99 = np.percentile(data, p_high)

        norm = (data - p1) / (p99 - p1 + eps)
        return np.clip(norm, 0, 1)
    

def normalize_intensity(a,p_low=1,p_high=99):
    """
    Helper to run minmax percentile normalization
    """
    norm_data = {}
    for c in range(a.shape[1]):
        img = a[0,c]

        img_norm = percentile_minmax_normalize(img,per_channel=False,p_low=p_low,p_high = p_high)

        norm_data[0,c] = img_norm
    return norm_data