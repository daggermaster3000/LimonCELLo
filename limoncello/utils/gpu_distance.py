"""GPU-accelerated Euclidean distance transform.

Uses CuPy (``cupyx.scipy.ndimage``) when available for a large speed-up on the
NVIDIA GPU, and transparently falls back to SciPy on the CPU when CuPy is not
installed or the GPU runs out of memory. The public ``distance_transform_edt``
is a drop-in replacement for ``scipy.ndimage.distance_transform_edt`` and always
returns NumPy arrays.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt as _cpu_edt

try:
    import cupy as _cp
    from cupyx.scipy.ndimage import distance_transform_edt as _gpu_edt
    _HAVE_CUPY = True
except Exception:                                   # cupy missing / no CUDA
    _cp = None
    _gpu_edt = None
    _HAVE_CUPY = False


def have_cupy() -> bool:
    """True if a working CuPy + cupyx EDT is importable."""
    return _HAVE_CUPY


def _free_gpu() -> None:
    if _cp is not None:
        try:
            _cp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass


def distance_transform_edt(
    input,
    sampling=None,
    return_distances: bool = True,
    return_indices: bool = False,
):
    """Drop-in for ``scipy.ndimage.distance_transform_edt``.

    GPU-accelerated via CuPy when available; otherwise SciPy. On a GPU
    out-of-memory error the computation is retried on the CPU. Outputs are
    always NumPy arrays, matching SciPy's return contract:
      - return_distances and not return_indices → distances
      - return_indices and not return_distances → indices
      - both                                    → (distances, indices)
    """
    if _HAVE_CUPY:
        d_in = None
        try:
            d_in = _cp.asarray(input)
            res = _gpu_edt(
                d_in, sampling=sampling,
                return_distances=return_distances,
                return_indices=return_indices,
            )
            if return_distances and return_indices:
                out = (_cp.asnumpy(res[0]), _cp.asnumpy(res[1]))
            else:
                out = _cp.asnumpy(res)
            return out
        except Exception as exc:
            print(f"[LC] CuPy distance transform failed ({exc}); "
                  f"falling back to SciPy (CPU).")
        finally:
            del d_in
            _free_gpu()

    return _cpu_edt(
        input, sampling=sampling,
        return_distances=return_distances,
        return_indices=return_indices,
    )
