import numpy as np

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