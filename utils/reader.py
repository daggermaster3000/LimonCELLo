# reader.py

from pathlib import Path
from imaris_ims_file_reader.ims import ims
import os
import re

from pathlib import Path
from imaris_ims_file_reader.ims import ims


def load_image(ims_path):
    """
    Load an Imaris .ims file and return the image object with metadata.

    Parameters
    ----------
    ims_path : str or Path
        Path to the .ims file

    Returns
    -------
    img : ims object
        Loaded image object
    metadata : dict
        Dictionary containing image metadata
    """

    # Normalize path for Windows/Mac/Linux
    ims_path = Path(ims_path).expanduser().resolve()

    print(f"Opening .ims file: {ims_path}")
    img = ims(str(ims_path))

    # Expected shape: (T, C, Z, Y, X)
    t, c, z, y, x = img.shape

    metadata = {
        "path": str(ims_path),
        "shape": img.shape,
        "timepoints": t,
        "channels": c,
        "z_slices": z,
        "height": y,
        "width": x,
        "voxel_size": None,
        "z_resolution": None,
        "xy_resolution": None,
    }

    print("Shape (T, C, Z, Y, X):", img.shape)
    print("Number of timepoints:", t)
    print("Number of channels:", c)
    print("Image dimensions (Z, Y, X):", (z, y, x))

    # Resolution metadata
    try:
        resolution = img.metaData[(0, 0, 0, "resolution")]

        z_res = resolution[0]
        xy_res = resolution[1:]

        metadata["voxel_size"] = tuple(resolution)
        metadata["z_resolution"] = z_res
        metadata["xy_resolution"] = xy_res

        print("Z Resolution:", z_res)
        print("XY Resolution:", xy_res)

    except Exception:
        print("Resolution metadata not found.")

    return img, metadata

def get_channel(img, channel=0, timepoint=0):
    """
    Extract a single channel as a 3D volume.

    Returns
    -------
    numpy array (Z, Y, X)
    """

    return img[timepoint, channel]


def get_voxel_size(img):
    """
    Return voxel size (Z, Y, X).
    """

    try:
        res = img.metaData[(0, 0, 0, "resolution")]
        return tuple(res)
    except Exception:
        return None
    

def load_ims_metadata(ims_path, verbose=True):
    """
    Load and parse metadata associated with an .ims file.

    Returns
    -------
    dict
        Dictionary containing:
        - metadata_path
        - pixel_size_xy_um
        - z_step_um
        - reference_z_resolution_um
        - voxel_size_z_um
        - voxel_size_y_um
        - voxel_size_x_um
        - channel_names
        - wavelengths
    """

    ims_dir = os.path.dirname(ims_path) or "."
    ims_name = os.path.basename(ims_path)
    ims_stem = ims_name[:-4] if ims_name.lower().endswith(".ims") else ims_name

    candidate_paths = []

    # 1) Exact "<stem>_metadata.txt"
    cand1 = os.path.join(ims_dir, f"{ims_stem}_metadata.txt")
    if os.path.exists(cand1):
        candidate_paths.append(cand1)

    # 2) Any "*metadata*.txt" containing the stem
    if not candidate_paths:
        for fname in os.listdir(ims_dir):
            if fname.lower().endswith(".txt") and ims_stem in fname:
                candidate_paths.append(os.path.join(ims_dir, fname))

    if not candidate_paths:
        if verbose:
            print("No metadata .txt file found next to the .ims file.")
        return None

    metadata_path = candidate_paths[0]

    if verbose:
        print("Using metadata file:\n", metadata_path)

    with open(metadata_path, "r") as f:
        metadata_text = f.read()

    def extract_value(pattern, text, cast=float):
        m = re.search(pattern, text)
        if m:
            try:
                return cast(m.group(1))
            except Exception:
                return None
        return None

    # --- XY pixel size ---
    pixel_size_camera = extract_value(
        r"Pixel Height \(µm\), Value=([\d\.]+)", metadata_text
    )
    magnification = extract_value(
        r"TotalConsolidatedOpticalMagnification, Value=([\d\.]+)", metadata_text
    )
    mag_correction = extract_value(
        r"Magnification Correction, Value=([\d\.]+)", metadata_text
    )

    if mag_correction is None:
        mag_correction = 1.0

    if pixel_size_camera is not None and magnification is not None:
        pixel_size_xy_um = pixel_size_camera / (magnification * mag_correction)
    else:
        pixel_size_xy_um = None

    # --- Z resolution ---
    z_step_um = extract_value(r"StepSize=([\d\.]+)", metadata_text)

    reference_z_resolution_um = extract_value(
        r"Reference Z Resolution, Value=([\d\.]+)", metadata_text
    )

    # Use StepSize first, otherwise Reference Z Resolution
    if z_step_um is not None:
        voxel_size_z_um = z_step_um
    elif reference_z_resolution_um is not None:
        voxel_size_z_um = reference_z_resolution_um
    else:
        voxel_size_z_um = 1.0

    # --- XY voxel fallback ---
    voxel_size_y_um = pixel_size_xy_um if pixel_size_xy_um is not None else 1.0
    voxel_size_x_um = pixel_size_xy_um if pixel_size_xy_um is not None else 1.0

    # --- Channels ---
    channel_names = re.findall(
        r"\[Channel\]\s+Name=([^\n]+)", metadata_text
    )

    wavelengths = re.findall(
        r"Wavelength=([\d\.]+) nm", metadata_text
    )
    wavelengths = [float(w) for w in wavelengths]

    if verbose:
        print()

        if pixel_size_xy_um is not None:
            print(f"Pixel size XY: {pixel_size_xy_um:.4f} µm")
        else:
            print("Pixel size XY: not found (using 1.0 µm fallback).")

        if z_step_um is not None:
            print(f"Z step (StepSize): {z_step_um:.4f} µm")
        elif reference_z_resolution_um is not None:
            print(f"Z resolution (Reference Z Resolution): {reference_z_resolution_um:.4f} µm")
        else:
            print("Z resolution: not found (using 1.0 µm fallback).")

        if channel_names:
            print("Channels:", channel_names)

        if wavelengths:
            print("Wavelengths (nm):", wavelengths)

        print(
            f"\nStored voxel sizes: "
            f"Z={voxel_size_z_um:.4f} µm, "
            f"Y={voxel_size_y_um:.4f} µm, "
            f"X={voxel_size_x_um:.4f} µm"
        )

    return {
        "metadata_path": metadata_path,
        "pixel_size_xy_um": pixel_size_xy_um,
        "z_step_um": z_step_um,
        "reference_z_resolution_um": reference_z_resolution_um,
        "voxel_size_z_um": voxel_size_z_um,
        "voxel_size_y_um": voxel_size_y_um,
        "voxel_size_x_um": voxel_size_x_um,
        "channel_names": channel_names,
        "wavelengths": wavelengths,
    }