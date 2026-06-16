import json
import os


def find_latest_run_dir(base: str) -> str | None:
    """Return the most recent lc-analysis-YYYY-MM-DD_HH-MM-SS subfolder."""
    if not base or not os.path.isdir(base):
        return None
    runs = sorted(
        d for d in os.listdir(base)
        if d.startswith("lc-analysis-") and os.path.isdir(os.path.join(base, d))
    )
    return os.path.join(base, runs[-1]) if runs else None


def load_run_json(run_dir: str) -> dict | None:
    """Load run_parameters.json from a run directory, or None if missing."""
    json_path = os.path.join(run_dir, "csv", "run_parameters.json")
    if not os.path.exists(json_path):
        return None
    try:
        with open(json_path) as f:
            return json.load(f)
    except Exception:
        return None
