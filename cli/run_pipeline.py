"""CLI entrypoint for running DT neuron analysis pipelines.

This module provides a simple argparse-based entrypoint that accepts
--input, --output, --config and --group-by. It scans a folder for
.ims files or accepts a single .ims file and runs a stubbed pipeline:
load -> preprocess -> segment -> associate -> export.

Follow repository conventions: images are TCZYX / (T, C, Z, Y, X)
and metadata should be discovered using `utils.reader.load_ims_metadata`.
"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import datetime

try:
	# project utility for reading .ims and metadata
	from utils.reader import load_image, load_ims_metadata, get_channel, get_voxel_size
except Exception:
	# allow import-time failures to be reported at runtime for environments
	load_image = None
	load_ims_metadata = None
	get_channel = None
	get_voxel_size = None

try:
	from preprocessing.preprocessing import percentile_minmax_normalize
except Exception:
	percentile_minmax_normalize = None


LOG = logging.getLogger("run_pipeline")


def find_ims_files(path: Path) -> List[Path]:
	"""Return list of .ims files for a path (file or folder)."""
	if path.is_file() and path.suffix.lower() == ".ims":
		return [path]

	if path.is_dir():
		return sorted([p for p in path.rglob("*.ims")])

	return []


def infer_group_from_filename(fname: str) -> str:
	"""Simple inference: take prefix before first underscore as the group."""
	stem = Path(fname).stem
	return stem.split("_")[0] if "_" in stem else stem


def stub_load(path: Path):
	"""Load image and metadata using utils.reader. Returns img, metadata."""
	if load_image is None:
		raise RuntimeError("utils.reader.load_image not available in this environment")

	img, metadata = load_image(str(path))
	return img, metadata


def stub_preprocess(img, metadata: Dict[str, Any]):
	"""Apply simple normalization to the first timepoint/channel and return array.

	Returns a dict with 'array' (ZYX) and 'voxel_size'.
	"""
	if get_channel is None:
		raise RuntimeError("utils.reader.get_channel not available")

	t = metadata.get("timepoints", 1)
	c = metadata.get("channels", 1)

	# For this stub, take timepoint 0 and channel 0
	vol = get_channel(img, channel=0, timepoint=0)

	if percentile_minmax_normalize is not None:
		# normalization expects numpy arrays
		try:
			norm = percentile_minmax_normalize(vol, p_low=1, p_high=99)
			vol = norm
		except Exception:
			LOG.debug("Normalization failed, returning raw volume", exc_info=True)

	voxel = get_voxel_size(img) if get_voxel_size is not None else None

	return {"array": vol, "voxel_size": voxel}


def stub_segment(preprocessed: Dict[str, Any]) -> Any:
	"""Stub segmentation: returns placeholder segmentation object (None).
	Replace with real segmentation logic.
	"""
	LOG.info("[stub] segmenting (placeholder)")
	return None


def stub_associate(segmentation: Any, preprocessed: Dict[str, Any]) -> List[Dict[str, Any]]:
	"""Stub association: returns empty list of associations."""
	LOG.info("[stub] associating objects (placeholder)")
	return []


def stub_export(results: Dict[str, Any], out_dir: Path):
	"""Export placeholder CSVs and a run log.

	Creates directories: out_dir/csv, out_dir/figures, out_dir/logs
	and writes three CSVs: features.csv, associations.csv, summary.csv
	with minimal headers so downstream tooling has expected files.
	"""
	csv_dir = out_dir / "csv"
	figs_dir = out_dir / "figures"
	logs_dir = out_dir / "logs"
	csv_dir.mkdir(parents=True, exist_ok=True)
	figs_dir.mkdir(parents=True, exist_ok=True)
	logs_dir.mkdir(parents=True, exist_ok=True)

	# features.csv header
	features_path = csv_dir / "features.csv"
	with open(features_path, "w", newline="") as fh:
		writer = csv.writer(fh)
		writer.writerow(["object_id", "feature_name", "value"])  # empty header

	associations_path = csv_dir / "associations.csv"
	with open(associations_path, "w", newline="") as fh:
		writer = csv.writer(fh)
		writer.writerow(["object_id", "neuron_id"])  # empty

	summary_path = csv_dir / "summary.csv"
	with open(summary_path, "w", newline="") as fh:
		writer = csv.writer(fh)
		writer.writerow(["file", "group", "status", "notes"])  # minimal

	# run log
	runlog = logs_dir / "run.log"
	with open(runlog, "a", encoding="utf-8") as fh:
		ts = datetime.datetime.utcnow().isoformat()
		fh.write(f"{ts} - exported placeholders to {csv_dir}\n")

	LOG.info("Exported placeholders to %s", csv_dir)


def process_file(path: Path, out_root: Path, group_by: str, config_path: Optional[Path] = None):
	LOG.info("Processing %s", path)

	img, metadata = stub_load(path)

	# group inference
	if group_by == "filename":
		group = infer_group_from_filename(path.name)
	else:
		group = infer_group_from_filename(path.name)

	# per-file output dir
	out_dir = out_root / path.stem
	out_dir.mkdir(parents=True, exist_ok=True)

	pre = stub_preprocess(img, metadata)
	seg = stub_segment(pre)
	assoc = stub_associate(seg, pre)

	results = {"image_path": str(path), "group": group, "metadata": metadata, "associations": assoc}

	stub_export(results, out_dir)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	p = argparse.ArgumentParser(prog="run_pipeline")
	p.add_argument("--input", "-i", required=True, help="Path to .ims file or folder containing .ims files")
	p.add_argument("--output", "-o", required=True, help="Output folder for CSVs/figures/logs")
	p.add_argument("--config", "-c", help="YAML config file (optional, currently not parsed)")
	p.add_argument("--group-by", default="filename", help="Grouping mode for experiments (default: filename)")
	return p.parse_args(argv)


def main(argv: Optional[List[str]] = None):
	logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(name)s: %(message)s")
	args = parse_args(argv)

	input_path = Path(args.input).expanduser().resolve()
	output_path = Path(args.output).expanduser().resolve()
	config_path = Path(args.config).expanduser().resolve() if args.config else None

	files = find_ims_files(input_path)
	if not files:
		LOG.error("No .ims files found at %s", input_path)
		return 2

	LOG.info("Found %d .ims files to process", len(files))

	for f in files:
		try:
			process_file(f, output_path, args.group_by, config_path)
		except Exception:
			LOG.exception("Failed processing %s", f)

	LOG.info("Run complete. Outputs written to %s", output_path)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

