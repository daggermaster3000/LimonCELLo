import numpy as np
from skimage.transform import resize


def _isotropic_gpu(vol, vs, s, order):
    """GPU resample to isotropic voxels via pyclesperanto. Raises on any failure
    so the caller can fall back to the CPU path."""
    import pyclesperanto_prototype as cle

    # clesperanto uses x,y,z ordering; our volume is (Z, Y, X).
    # new_size = old_size * factor, so factor = old_spacing / target_spacing.
    fz, fy, fx = vs[0] / s, vs[1] / s, vs[2] / s
    linear = order != 0                              # linear for intensities, nearest for labels

    src = cle.push(np.asarray(vol, dtype=np.float32))
    # Anti-alias before *downsampling* intensities (nearest labels must not blur).
    if linear:
        sig = [max(0.0, (1.0 / f - 1.0) / 2.0) if f < 1.0 else 0.0
               for f in (fx, fy, fz)]
        if any(g > 0 for g in sig):
            src = cle.gaussian_blur(src, sigma_x=sig[0], sigma_y=sig[1], sigma_z=sig[2])
    out = cle.resample(src, factor_x=fx, factor_y=fy, factor_z=fz,
                       linear_interpolation=linear)
    return np.asarray(out)


def make_isotropic(volume, voxel_size, target_spacing=None, order=1, use_gpu=True):
    """
    Resample a (Z, Y, X) volume to isotropic voxels.

    By default the target spacing is the *coarsest* axis spacing
    (``max(voxel_size)``, typically Z), so the finer axes are **downsampled**
    to match it. This keeps the voxel count from exploding (no upsampling) at
    the cost of in-plane resolution.

    When ``use_gpu`` is set (default) the resampling runs on the GPU via
    pyclesperanto's ``resample`` (with a Gaussian anti-alias pre-blur when
    downsampling intensities); it transparently falls back to the CPU
    ``skimage.transform.resize`` if the GPU path is unavailable or errors.

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
    use_gpu : bool
        Try the GPU (clesperanto) path first. Falls back to CPU on failure.

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

    if use_gpu:
        try:
            out = _isotropic_gpu(vol, vs, s, order).astype(vol.dtype, copy=False)
            return out, (s, s, s)
        except Exception as exc:                     # noqa: BLE001
            print(f"[LC] GPU isotropic resample failed ({exc}); using CPU.")

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