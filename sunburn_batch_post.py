# Copyright (C) 2024 - 2026 ANSYS, Inc. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""RAWXM3 batch post-processing tool.

中文：
- 批量读取目录中的 RAWXM3 文件。
- 按阈值过滤后，按角度网格导出每个文件的 layer 最大值到 CSV。
- 支持命令行与桌面 UI 两种模式。

English:
- Batch-load RAWXM3 files from a directory.
- Apply threshold filtering and export per-file angle-grid layer maxima to CSV.
- Supports both CLI and desktop UI modes.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py  
import numpy as np

from PySide6.QtWidgets import (
	QApplication,
	QCheckBox,
	QComboBox,
	QDoubleSpinBox,
	QFileDialog,
	QFormLayout,
	QHBoxLayout,
	QLabel,
	QLineEdit,
	QMainWindow,
	QMessageBox,
	QPlainTextEdit,
	QPushButton,
	QVBoxLayout,
	QWidget,
)
from tqdm import tqdm  # type: ignore[import-untyped]


# 过滤字段常量 / Filter field constants
SURFACE_FIELD = "Triangle Surface (m^2)"
RAY_HITS_FIELD = "Ray hits"
RAWXM3_SUFFIXES = {".rawxm3", ".h5", ".hdf5"}
CATEGORY_PREFIXES = {
	"Transmission": "Transmission Layer",
	"Absorption": "Absorption Layer",
	"Incident": "Incident Layer",
	"Reflection": "Reflection Layer",
}
SOURCE_PATTERN = re.compile(
	r"^\s*Z(?P<zenith>[+-]?\d+(?:\.\d+)?)_A(?P<azimuth>[+-]?\d+(?:\.\d+)?)\s*$",
	re.IGNORECASE,
)


def normalize_text(value: Any) -> str:
	"""统一 bytes 文本到 str / Normalize bytes-like values to str."""
	if isinstance(value, bytes):
		return value.decode("utf-8", errors="replace")
	if isinstance(value, np.bytes_):
		return bytes(value).decode("utf-8", errors="replace")
	return str(value)


def read_triplets(dataset: h5py.Dataset, dataset_name: str) -> np.ndarray:
	"""读取并重排三元组数组 / Read flat triplets and reshape to (N, 3)."""
	values = np.asarray(dataset)
	if values.size % 3 != 0:
		raise ValueError(
			f"Dataset '{dataset_name}' size {values.size} is not divisible by 3."
		)
	return values.reshape(-1, 3)


def get_faces_group(hdf5_file: h5py.File) -> tuple[h5py.Group, list[str]]:
	"""定位 faces 分组 / Resolve faces group and keys from RAWXM3 root."""
	root_keys = list(hdf5_file.keys())
	if not root_keys:
		raise ValueError("RAWXM3 file is empty.")

	faces_group = hdf5_file[root_keys[0]]
	face_keys = list(faces_group.keys())
	if not face_keys:
		raise ValueError("No face entries found in RAWXM3 file.")

	return faces_group, face_keys


@dataclass
class AggregatedMeshData:
	"""聚合网格数据 / Aggregated mesh geometry and per-cell data."""
	vertices: list[list[float]] = field(default_factory=list)
	facets: list[list[int]] = field(default_factory=list)
	facets_data: dict[str, list[float]] = field(default_factory=dict)

	def add_face(
		self,
		vertices_raw: np.ndarray,
		facets_raw: np.ndarray,
		face_data_by_name: dict[str, list[float]],
		vertex_offset: int,
	) -> None:
		"""追加单个 face 数据 / Append one face into the aggregated buffers."""
		self.vertices.extend(
			[[float(x), float(y), float(z)] for x, y, z in vertices_raw]
		)

		shifted_facets = facets_raw.astype(np.int64) + vertex_offset
		for row_index in range(int(shifted_facets.shape[0])):
			self.facets.append(
				[
					int(shifted_facets[row_index, 0]),
					int(shifted_facets[row_index, 1]),
					int(shifted_facets[row_index, 2]),
				]
			)

		if not self.facets_data:
			self.facets_data = {
				name: list(values) for name, values in face_data_by_name.items()
			}
			return

		if set(self.facets_data) != set(face_data_by_name):
			raise ValueError(
				"Facet data schema mismatch between faces: "
				f"expected {sorted(self.facets_data)}, got {sorted(face_data_by_name)}."
			)

		for name, values in face_data_by_name.items():
			self.facets_data[name].extend(values)


def extract_face_data_by_name(
	facets_data_group: h5py.Group,
	facet_count: int,
) -> dict[str, list[float]]:
	"""按名称提取 face 物理量 / Extract per-face scalar arrays by field name."""
	result: dict[str, list[float]] = {}
	for key in facets_data_group.keys():
		data_group = facets_data_group[key]
		name = normalize_text(data_group.attrs["name"])
		data_size = int(data_group.attrs["data_size"])

		if data_size == 1:
			value = float(data_group.attrs["data"])
			result[name] = [value] * facet_count
			continue

		values = np.asarray(data_group["data"], dtype=np.float64).ravel().tolist()
		if len(values) != facet_count:
			raise ValueError(
				f"Facet data size mismatch for '{name}': expected {facet_count}, got {len(values)}."
			)
		result[name] = values

	return result


def convert_rawxm3_to_aggregated_mesh(rawxm3_path: Path) -> AggregatedMeshData:
	"""读取 RAWXM3 并聚合网格 / Parse RAWXM3 file and aggregate mesh data."""
	aggregated = AggregatedMeshData()
	vertex_offset = 0

	with h5py.File(rawxm3_path, "r") as hdf5_file:
		faces_group, face_keys = get_faces_group(hdf5_file)

		for key in tqdm(face_keys, desc=f"Faces {rawxm3_path.name}", unit="face", ncols=90):
			face_group = faces_group[key]

			vertices_raw = read_triplets(face_group["vertices"], "vertices")
			facets_raw = read_triplets(face_group["facets"], "facets")
			facet_count = int(facets_raw.shape[0])

			face_data_by_name = extract_face_data_by_name(
				face_group["facets_data"],
				facet_count,
			)

			aggregated.add_face(
				vertices_raw=vertices_raw,
				facets_raw=facets_raw,
				face_data_by_name=face_data_by_name,
				vertex_offset=vertex_offset,
			)
			vertex_offset += int(vertices_raw.shape[0])

	return aggregated


def build_filter_mask(
	facets_data: dict[str, np.ndarray],
	surface_percentile: float,
	ray_hits_percentile: float,
	cell_count: int,
) -> np.ndarray:
	"""构建过滤掩码 / Build keep-mask from Surface and Ray hits percentiles.

	当前规则：仅当两个条件都不满足时过滤。
	Current rule: filter out a cell only when both conditions fail.
	"""
	# Keep behavior aligned with the latest UI script:
	# filter out a cell only when both active filter conditions fail.
	def _compute_percentile_threshold(name: str, percentile: float) -> float | None:
		values = facets_data.get(name)
		if values is None:
			return None

		finite_values = values[np.isfinite(values)]
		if finite_values.size == 0:
			return None

		if name == RAY_HITS_FIELD:
			# Exclude 0 hits to avoid dragging the threshold down to 0.
			finite_values = finite_values[finite_values != 0.0]
			if finite_values.size == 0:
				return 0.0

		return float(np.percentile(finite_values, percentile))

	surface_fail: np.ndarray | None = None
	ray_hits_fail: np.ndarray | None = None

	surface_values = facets_data.get(SURFACE_FIELD)
	surface_threshold_m2 = _compute_percentile_threshold(SURFACE_FIELD, surface_percentile)
	if surface_values is not None and surface_threshold_m2 is not None:
		surface_fail = surface_values < surface_threshold_m2

	ray_hits_values = facets_data.get(RAY_HITS_FIELD)
	ray_hits_threshold = _compute_percentile_threshold(RAY_HITS_FIELD, ray_hits_percentile)
	if ray_hits_values is not None and ray_hits_threshold is not None:
		ray_hits_fail = ray_hits_values < ray_hits_threshold

	if surface_fail is not None and ray_hits_fail is not None:
		return ~(surface_fail & ray_hits_fail)

	if surface_fail is not None:
		return ~surface_fail

	if ray_hits_fail is not None:
		return ~ray_hits_fail

	return np.ones(cell_count, dtype=bool)


def angle_key(value: float) -> float:
	"""标准化角度键值 / Normalize float angle for stable dictionary keys."""
	return round(float(value), 8)


def parse_source_line(line: str, line_number: int) -> tuple[float, float]:
	"""解析 sourcename 行文本 / Parse one sourcename entry line."""
	match = SOURCE_PATTERN.match(line)
	if match is None:
		raise ValueError(
			f"Invalid sourcename format at line {line_number}: '{line}'. Expected like Z5_A0."
		)
	zenith = float(match.group("zenith"))
	azimuth = float(match.group("azimuth"))
	return zenith, azimuth


def load_source_angles(source_name_file: Path) -> list[tuple[float, float]]:
	"""读取 sourcename 文件 / Load ordered (zenith, azimuth) list from sourcename.txt."""
	if not source_name_file.exists() or not source_name_file.is_file():
		raise ValueError(f"sourcename file does not exist: {source_name_file}")

	entries: list[tuple[float, float]] = []
	with open(source_name_file, "r", encoding="utf-8") as source_file:
		for line_number, raw_line in enumerate(source_file, start=1):
			line = raw_line.strip()
			if not line:
				continue
			entries.append(parse_source_line(line, line_number))

	if not entries:
		raise ValueError("sourcename file is empty.")

	return entries


def build_angle_range(start: float, end: float, step: float, label: str) -> list[float]:
	"""构建角度序列 / Build inclusive angle sequence from start/end/step."""
	if step == 0:
		raise ValueError(f"{label} step cannot be 0.")
	if end > start and step < 0:
		raise ValueError(f"{label} step must be positive when end > start.")
	if end < start and step > 0:
		raise ValueError(f"{label} step must be negative when end < start.")

	values: list[float] = []
	current = float(start)
	if step > 0:
		while current <= end + 1e-10:
			values.append(angle_key(current))
			current += step
	else:
		while current >= end - 1e-10:
			values.append(angle_key(current))
			current += step

	if not values:
		raise ValueError(f"{label} range is empty.")

	return values


def get_layer_index(name: str) -> int | None:
	"""从 layer 名称提取序号 / Extract layer index from scalar field name."""
	match = re.search(r"layer\s*(\d+)", name, flags=re.IGNORECASE)
	if match is None:
		return None
	return int(match.group(1))


def resolve_category(category: str) -> tuple[str, str]:
	"""解析分类参数 / Resolve category to (display_name, match_prefix)."""
	for key, prefix in CATEGORY_PREFIXES.items():
		if key.lower() == category.lower():
			return key, prefix
	# Fallback: treat input as custom prefix.
	return category, category


def compute_layer_filtered_maxima(
	facets_data_np: dict[str, np.ndarray],
	mask: np.ndarray,
	category_prefix: str,
) -> tuple[dict[int, float], int]:
	"""计算每个 layer 序号的过滤后最大值 / Compute filtered max by layer index."""
	result: dict[int, float] = {}
	matched_layer_count = 0
	for layer_name, layer_values in facets_data_np.items():
		if layer_name in {SURFACE_FIELD, RAY_HITS_FIELD}:
			continue
		if category_prefix.lower() not in layer_name.lower():
			continue
		matched_layer_count += 1
		layer_index = get_layer_index(layer_name)
		if layer_index is None:
			continue

		filtered_values = np.where(mask, layer_values, 0.0)
		finite_values = filtered_values[np.isfinite(filtered_values)]
		if finite_values.size == 0:
			result[layer_index] = 0.0
		else:
			result[layer_index] = float(np.max(finite_values))

	return result, matched_layer_count


def export_file_angle_grid(
	rawxm3_path: Path,
	output_dir: Path,
	category_input: str,
	surface_percentile: float,
	ray_hits_percentile: float,
	source_angles: list[tuple[float, float]],
	zenith_values: list[float],
	azimuth_values: list[float],
) -> Path:
	"""导出单文件角度网格结果 / Export one RAWXM3 file angle-grid CSV."""
	aggregated = convert_rawxm3_to_aggregated_mesh(rawxm3_path)

	if not aggregated.vertices or not aggregated.facets:
		raise ValueError("No geometry found in file.")

	facets = np.asarray(aggregated.facets, dtype=np.int64)

	facets_data_np = {
		name: np.asarray(values, dtype=np.float64)
		for name, values in aggregated.facets_data.items()
	}

	cell_count = int(facets.shape[0])
	for name, values in facets_data_np.items():
		if values.size != cell_count:
			raise ValueError(
				f"Cell data size mismatch for '{name}': expected {cell_count}, got {values.size}."
			)

	mask = build_filter_mask(
		facets_data=facets_data_np,
		surface_percentile=surface_percentile,
		ray_hits_percentile=ray_hits_percentile,
		cell_count=cell_count,
	)

	category_name, category_prefix = resolve_category(category_input)
	layer_maxima, matched_layer_count = compute_layer_filtered_maxima(
		facets_data_np,
		mask,
		category_prefix=category_prefix,
	)
	if matched_layer_count == 0:
		raise ValueError(
			f"No layers found for category '{category_name}' in file '{rawxm3_path.name}'."
		)

	angle_to_layer: dict[tuple[float, float], int] = {}
	for layer_index, (zenith, azimuth) in enumerate(source_angles):
		key = (angle_key(zenith), angle_key(azimuth))
		if key in angle_to_layer:
			raise ValueError(
				f"Duplicate angle mapping in sourcename: Z{zenith}_A{azimuth}."
			)
		angle_to_layer[key] = layer_index

	rows: list[list[str]] = []
	header = ["zenith\\azimuth"] + [f"{azimuth:g}" for azimuth in azimuth_values]
	rows.append(header)

	for zenith in zenith_values:
		row = [f"{zenith:g}"]
		for azimuth in azimuth_values:
			lookup_azimuth = azimuth
			if abs(zenith - 90.0) <= 1e-8:
				# Special case: use (Z=90, A=0) value for all azimuth columns.
				lookup_azimuth = 0.0

			key = (zenith, angle_key(lookup_azimuth))
			if key not in angle_to_layer:
				raise ValueError(
					f"No corresponding angle mapping for Z{zenith:g}_A{lookup_azimuth:g} in sourcename file."
				)

			layer_index = angle_to_layer[key]
			if layer_index not in layer_maxima:
				raise ValueError(
					f"No corresponding {category_name} layer{layer_index} data for "
					f"Z{zenith:g}_A{lookup_azimuth:g}."
				)

			row.append(f"{layer_maxima[layer_index]:.9g}")
		rows.append(row)

	output_dir.mkdir(parents=True, exist_ok=True)
	output_csv = output_dir / f"{rawxm3_path.stem}.csv"

	with open(output_csv, "w", encoding="utf-8-sig", newline="") as csv_file:
		writer = csv.writer(csv_file)
		writer.writerows(rows)

	return output_csv


def find_rawxm3_files(input_dir: Path, recursive: bool) -> list[Path]:
	"""查找输入文件 / Find RAWXM3-like files in a directory."""
	if recursive:
		files = [
			path
			for path in input_dir.rglob("*")
			if path.is_file() and path.suffix.lower() in RAWXM3_SUFFIXES
		]
	else:
		files = [
			path
			for path in input_dir.iterdir()
			if path.is_file() and path.suffix.lower() in RAWXM3_SUFFIXES
		]
	return sorted(files)


def parse_args() -> argparse.Namespace:
	"""解析 CLI 参数 / Parse command-line arguments."""
	parser = argparse.ArgumentParser(
		description=(
			"Batch process RAWXM3 files and export one angle-grid CSV per file. "
			"Layer-index-to-angle mapping is loaded from sourcename.txt."
		)
	)
	parser.add_argument("input_dir", nargs="?", type=Path, help="Directory containing RAWXM3 files")
	parser.add_argument("--sourcename-file", type=Path, required=False, help="Path to sourcename.txt")
	parser.add_argument(
		"--category",
		required=False,
		default="Transmission",
		help=(
			"Layer category name (Transmission/Absorption/Incident/Reflection) "
			"or a custom layer-name prefix"
		),
	)
	parser.add_argument(
		"--surface-percentile",
		type=float,
		default=0.0,
		help="Surface percentile (0-100)",
	)
	parser.add_argument(
		"--rayhits-percentile",
		type=float,
		default=0,
		help="Ray hits percentile (0-100), with 0-values excluded in threshold computation",
	)
	parser.add_argument("--zenith-start", type=float, default=0.0, help="Zenith start angle")
	parser.add_argument("--zenith-end", type=float, default=0.0, help="Zenith end angle")
	parser.add_argument("--zenith-step", type=float, default=1.0, help="Zenith angle step")
	parser.add_argument("--azimuth-start", type=float, default=0.0, help="Azimuth start angle")
	parser.add_argument("--azimuth-end", type=float, default=0.0, help="Azimuth end angle")
	parser.add_argument("--azimuth-step", type=float, default=1.0, help="Azimuth angle step")
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=None,
		help="Output directory for CSV files (default: input_dir)",
	)
	parser.add_argument(
		"--recursive",
		action="store_true",
		help="Recursively search RAWXM3 files under input_dir",
	)
	parser.add_argument(
		"--gui",
		action="store_true",
		help="Launch desktop UI",
	)
	return parser.parse_args()


def run_batch_processing(
	input_dir: Path,
	output_dir: Path,
	source_name_file: Path,
	category: str,
	surface_percentile: float,
	rayhits_percentile: float,
	zenith_start: float,
	zenith_end: float,
	zenith_step: float,
	azimuth_start: float,
	azimuth_end: float,
	azimuth_step: float,
	recursive: bool,
	logger: Callable[[str], None],
) -> int:
	"""执行批处理任务 / Run batch processing workflow for all matched files."""
	if not input_dir.exists() or not input_dir.is_dir():
		logger(f"Input directory does not exist or is not a directory: {input_dir}")
		return 1

	files = find_rawxm3_files(input_dir, recursive=recursive)
	if not files:
		logger(f"No RAWXM3 files found in: {input_dir}")
		return 1

	try:
		source_angles = load_source_angles(source_name_file)
		zenith_values = build_angle_range(zenith_start, zenith_end, zenith_step, "Zenith")
		azimuth_values = build_angle_range(azimuth_start, azimuth_end, azimuth_step, "Azimuth")
	except Exception as exc:
		logger(f"Configuration error: {exc}")
		return 1

	logger(f"Found {len(files)} RAWXM3 files.")
	logger(f"Source mapping file: {source_name_file}")
	logger(f"Layer category: {category}")
	logger(f"Surface percentile: {surface_percentile}%")
	logger(f"Ray hits percentile: {rayhits_percentile}% (0-values excluded)")
	logger(
		f"Zenith range: start={zenith_start}, end={zenith_end}, step={zenith_step}"
	)
	logger(
		f"Azimuth range: start={azimuth_start}, end={azimuth_end}, step={azimuth_step}"
	)
	logger(f"Output directory: {output_dir}")

	success = 0
	failed = 0

	for file_path in files:
		logger(f"\nProcessing: {file_path.name}")
		try:
			output_csv = export_file_angle_grid(
				rawxm3_path=file_path,
				output_dir=output_dir,
				category_input=category,
				surface_percentile=surface_percentile,
				ray_hits_percentile=rayhits_percentile,
				source_angles=source_angles,
				zenith_values=zenith_values,
				azimuth_values=azimuth_values,
			)
			logger(f"Exported: {output_csv.name}")
			success += 1
		except Exception as exc:
			logger(f"Failed: {file_path.name} -> {exc}")
			failed += 1

	logger("\nBatch completed.")
	logger(f"Success: {success}")
	logger(f"Failed: {failed}")
	return 0 if failed == 0 else 2


class BatchPostWindow(QMainWindow):
	"""桌面批处理窗口 / Desktop UI window for batch configuration and run."""
	def __init__(self) -> None:
		super().__init__()
		self.setWindowTitle("RAWXM3 Batch Post Tool")
		self.resize(920, 680)
		self._build_ui()

	def _build_ui(self) -> None:
		"""构建界面控件 / Build UI controls and layout."""
		central = QWidget(self)
		self.setCentralWidget(central)

		root_layout = QVBoxLayout(central)
		form_layout = QFormLayout()

		input_row = QHBoxLayout()
		self.input_dir_edit = QLineEdit()
		self.input_dir_edit.setPlaceholderText("Select RAWXM3 directory")
		input_browse = QPushButton("Browse")
		input_browse.clicked.connect(self._choose_input_dir)
		input_row.addWidget(self.input_dir_edit, 1)
		input_row.addWidget(input_browse)
		form_layout.addRow("Input directory", input_row)

		output_row = QHBoxLayout()
		self.output_dir_edit = QLineEdit()
		self.output_dir_edit.setPlaceholderText("Optional, defaults to input directory")
		output_browse = QPushButton("Browse")
		output_browse.clicked.connect(self._choose_output_dir)
		output_row.addWidget(self.output_dir_edit, 1)
		output_row.addWidget(output_browse)
		form_layout.addRow("Output directory", output_row)

		source_row = QHBoxLayout()
		self.source_file_edit = QLineEdit()
		self.source_file_edit.setPlaceholderText("Select sourcename.txt")
		source_browse = QPushButton("Browse")
		source_browse.clicked.connect(self._choose_source_file)
		source_row.addWidget(self.source_file_edit, 1)
		source_row.addWidget(source_browse)
		form_layout.addRow("Sourcename file", source_row)

		self.category_combo = QComboBox()
		self.category_combo.setEditable(True)
		self.category_combo.addItems(list(CATEGORY_PREFIXES.keys()))
		form_layout.addRow("Layer category", self.category_combo)

		self.surface_spin = QDoubleSpinBox()
		self.surface_spin.setRange(0.0, 100.0)
		self.surface_spin.setDecimals(2)
		self.surface_spin.setSingleStep(1.0)
		self.surface_spin.setSuffix(" %")
		self.surface_spin.setValue(0.0)
		form_layout.addRow("Surface percentile", self.surface_spin)

		self.rayhits_spin = QDoubleSpinBox()
		self.rayhits_spin.setRange(0.0, 100.0)
		self.rayhits_spin.setDecimals(2)
		self.rayhits_spin.setSingleStep(1.0)
		self.rayhits_spin.setSuffix(" %")
		self.rayhits_spin.setValue(0)
		form_layout.addRow("Ray hits percentile", self.rayhits_spin)

		self.zenith_start_spin = QDoubleSpinBox()
		self.zenith_start_spin.setRange(-360.0, 360.0)
		self.zenith_start_spin.setDecimals(3)
		self.zenith_start_spin.setValue(0.0)
		form_layout.addRow("Zenith start", self.zenith_start_spin)

		self.zenith_end_spin = QDoubleSpinBox()
		self.zenith_end_spin.setRange(-360.0, 360.0)
		self.zenith_end_spin.setDecimals(3)
		self.zenith_end_spin.setValue(10.0)
		form_layout.addRow("Zenith end", self.zenith_end_spin)

		self.zenith_step_spin = QDoubleSpinBox()
		self.zenith_step_spin.setRange(-360.0, 360.0)
		self.zenith_step_spin.setDecimals(3)
		self.zenith_step_spin.setValue(5.0)
		form_layout.addRow("Zenith step", self.zenith_step_spin)

		self.azimuth_start_spin = QDoubleSpinBox()
		self.azimuth_start_spin.setRange(-360.0, 360.0)
		self.azimuth_start_spin.setDecimals(3)
		self.azimuth_start_spin.setValue(-90.0)
		form_layout.addRow("Azimuth start", self.azimuth_start_spin)

		self.azimuth_end_spin = QDoubleSpinBox()
		self.azimuth_end_spin.setRange(-360.0, 360.0)
		self.azimuth_end_spin.setDecimals(3)
		self.azimuth_end_spin.setValue(90.0)
		form_layout.addRow("Azimuth end", self.azimuth_end_spin)

		self.azimuth_step_spin = QDoubleSpinBox()
		self.azimuth_step_spin.setRange(-360.0, 360.0)
		self.azimuth_step_spin.setDecimals(3)
		self.azimuth_step_spin.setValue(90.0)
		form_layout.addRow("Azimuth step", self.azimuth_step_spin)

		self.recursive_check = QCheckBox("Recursive scan")
		self.recursive_check.setChecked(False)
		form_layout.addRow("Scan mode", self.recursive_check)

		root_layout.addLayout(form_layout)

		button_row = QHBoxLayout()
		self.run_button = QPushButton("Run Batch")
		self.run_button.clicked.connect(self._run_batch)
		button_row.addWidget(self.run_button)
		button_row.addStretch(1)
		root_layout.addLayout(button_row)

		root_layout.addWidget(QLabel("Logs"))
		self.log_output = QPlainTextEdit(self)
		self.log_output.setReadOnly(True)
		root_layout.addWidget(self.log_output, 1)

	def _choose_input_dir(self) -> None:
		"""选择输入目录 / Select input folder."""
		directory = QFileDialog.getExistingDirectory(self, "Select input directory")
		if directory:
			self.input_dir_edit.setText(directory)
			if not self.output_dir_edit.text().strip():
				self.output_dir_edit.setText(directory)

	def _choose_output_dir(self) -> None:
		"""选择输出目录 / Select output folder."""
		directory = QFileDialog.getExistingDirectory(self, "Select output directory")
		if directory:
			self.output_dir_edit.setText(directory)

	def _choose_source_file(self) -> None:
		"""选择 sourcename 文件 / Select sourcename.txt file."""
		file_path, _ = QFileDialog.getOpenFileName(
			self,
			"Select sourcename file",
			"",
			"Text files (*.txt);;All files (*.*)",
		)
		if file_path:
			self.source_file_edit.setText(file_path)

	def _append_log(self, message: str) -> None:
		"""追加日志文本 / Append one line to UI log area."""
		self.log_output.appendPlainText(message)
		QApplication.processEvents()

	def _run_batch(self) -> None:
		"""读取参数并运行 / Read form values and execute batch run."""
		input_dir_raw = self.input_dir_edit.text().strip()
		if not input_dir_raw:
			QMessageBox.warning(self, "Run Batch", "Please select an input directory.")
			return

		source_file_raw = self.source_file_edit.text().strip()
		if not source_file_raw:
			QMessageBox.warning(self, "Run Batch", "Please select sourcename.txt.")
			return

		category = self.category_combo.currentText().strip()
		if not category:
			QMessageBox.warning(self, "Run Batch", "Please select or input a layer category.")
			return

		input_dir = Path(input_dir_raw).resolve()
		output_dir_raw = self.output_dir_edit.text().strip()
		output_dir = Path(output_dir_raw).resolve() if output_dir_raw else input_dir

		self.log_output.clear()
		self.run_button.setEnabled(False)
		try:
			code = run_batch_processing(
				input_dir=input_dir,
				output_dir=output_dir,
				source_name_file=Path(source_file_raw).resolve(),
				category=category,
				surface_percentile=float(self.surface_spin.value()),
				rayhits_percentile=float(self.rayhits_spin.value()),
				zenith_start=float(self.zenith_start_spin.value()),
				zenith_end=float(self.zenith_end_spin.value()),
				zenith_step=float(self.zenith_step_spin.value()),
				azimuth_start=float(self.azimuth_start_spin.value()),
				azimuth_end=float(self.azimuth_end_spin.value()),
				azimuth_step=float(self.azimuth_step_spin.value()),
				recursive=bool(self.recursive_check.isChecked()),
				logger=self._append_log,
			)
		finally:
			self.run_button.setEnabled(True)

		if code == 0:
			QMessageBox.information(self, "Run Batch", "Batch completed successfully.")
		else:
			QMessageBox.warning(self, "Run Batch", "Batch completed with errors. Check logs.")


def launch_gui() -> int:
	"""启动桌面 UI / Launch desktop GUI mode."""
	app = QApplication(sys.argv)
	window = BatchPostWindow()
	window.show()
	return app.exec()


def main() -> int:
	"""程序入口 / Program entrypoint for CLI or GUI mode."""
	args = parse_args()

	if bool(args.gui) or args.input_dir is None:
		return launch_gui()

	if args.sourcename_file is None:
		print("--sourcename-file is required in CLI mode.")
		return 1

	input_dir = args.input_dir.resolve()
	output_dir = (args.output_dir or input_dir).resolve()
	return run_batch_processing(
		input_dir=input_dir,
		output_dir=output_dir,
		source_name_file=args.sourcename_file.resolve(),
		category=str(args.category),
		surface_percentile=float(args.surface_percentile),
		rayhits_percentile=float(args.rayhits_percentile),
		zenith_start=float(args.zenith_start),
		zenith_end=float(args.zenith_end),
		zenith_step=float(args.zenith_step),
		azimuth_start=float(args.azimuth_start),
		azimuth_end=float(args.azimuth_end),
		azimuth_step=float(args.azimuth_step),
		recursive=bool(args.recursive),
		logger=print,
	)


if __name__ == "__main__":
	raise SystemExit(main())
