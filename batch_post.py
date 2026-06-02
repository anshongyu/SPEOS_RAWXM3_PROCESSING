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
- 按阈值过滤后，导出指定分类下各层最大值与坐标到 CSV。
- 支持命令行与桌面 UI 两种模式。

English:
- Batch-load RAWXM3 files from a directory.
- Apply threshold filtering and export per-layer maxima + coordinates to CSV.
- Supports both CLI and desktop UI modes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import h5py  # type: ignore[import-untyped]
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
	QSpinBox,
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
	surface_threshold_mm2: float,
	ray_hits_threshold: int,
	cell_count: int,
) -> np.ndarray:
	"""构建过滤掩码 / Build keep-mask from Surface and Ray hits thresholds.

	当前规则：仅当两个条件都不满足时过滤。
	Current rule: filter out a cell only when both conditions fail.
	"""
	# Keep behavior aligned with the latest UI script:
	# filter out a cell only when both active filter conditions fail.
	surface_fail: np.ndarray | None = None
	ray_hits_fail: np.ndarray | None = None

	surface_values = facets_data.get(SURFACE_FIELD)
	surface_threshold_m2 = float(surface_threshold_mm2) / 1_000_000.0
	if surface_values is not None and surface_threshold_m2 > 0:
		surface_fail = surface_values < surface_threshold_m2

	ray_hits_values = facets_data.get(RAY_HITS_FIELD)
	if ray_hits_values is not None and ray_hits_threshold > 0:
		ray_hits_fail = ray_hits_values < float(ray_hits_threshold)

	if surface_fail is not None and ray_hits_fail is not None:
		return ~(surface_fail & ray_hits_fail)

	if surface_fail is not None:
		return ~surface_fail

	if ray_hits_fail is not None:
		return ~ray_hits_fail

	return np.ones(cell_count, dtype=bool)


def compute_cell_centers(vertices: np.ndarray, facets: np.ndarray) -> np.ndarray:
	"""计算三角面中心点 / Compute triangle cell centers."""
	p1 = vertices[facets[:, 0]]
	p2 = vertices[facets[:, 1]]
	p3 = vertices[facets[:, 2]]
	return (p1 + p2 + p3) / 3.0


def resolve_category(category: str) -> tuple[str, str]:
	"""解析分类参数 / Resolve category to (display_name, match_prefix)."""
	for key, prefix in CATEGORY_PREFIXES.items():
		if key.lower() == category.lower():
			return key, prefix
	# Fallback: treat input as custom prefix.
	return category, category


def sanitize_filename_part(name: str) -> str:
	"""清洗文件名片段 / Sanitize filename segment for Windows-safe output."""
	invalid_chars = '<>:"/\\|?*'
	sanitized = "".join("_" if ch in invalid_chars else ch for ch in name.strip())
	sanitized = sanitized.replace(" ", "_")
	while "__" in sanitized:
		sanitized = sanitized.replace("__", "_")
	return sanitized.strip("._") or "category"


def export_file_category_maxima(
	rawxm3_path: Path,
	output_dir: Path,
	category_input: str,
	surface_threshold_mm2: float,
	ray_hits_threshold: int,
) -> Path:
	"""导出单文件分类层结果 / Export one RAWXM3 file results to CSV.

	CSV 列：category, layer_name, filtered_max, x, y, z
	CSV fields: category, layer_name, filtered_max, x, y, z
	"""
	aggregated = convert_rawxm3_to_aggregated_mesh(rawxm3_path)

	if not aggregated.vertices or not aggregated.facets:
		raise ValueError("No geometry found in file.")

	vertices = np.asarray(aggregated.vertices, dtype=np.float64)
	facets = np.asarray(aggregated.facets, dtype=np.int64)
	centers = compute_cell_centers(vertices, facets)

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

	category_name, category_prefix = resolve_category(category_input)
	layer_names = [
		name for name in facets_data_np.keys() if category_prefix.lower() in name.lower()
	]

	mask = build_filter_mask(
		facets_data=facets_data_np,
		surface_threshold_mm2=surface_threshold_mm2,
		ray_hits_threshold=ray_hits_threshold,
		cell_count=cell_count,
	)

	rows: list[dict[str, str]] = []
	for layer_name in sorted(layer_names):
		layer_values = facets_data_np[layer_name]
		filtered_values = np.where(mask, layer_values, 0.0)
		finite_mask = np.isfinite(filtered_values)

		if filtered_values.size == 0 or not np.any(finite_mask):
			rows.append(
				{
					"category": category_name,
					"layer_name": layer_name,
					"filtered_max": "",
					"x": "",
					"y": "",
					"z": "",
				}
			)
			continue

		valid_indices = np.where(finite_mask)[0]
		valid_values = filtered_values[finite_mask]
		max_local_index = int(np.argmax(valid_values))
		max_cell_id = int(valid_indices[max_local_index])
		max_value = float(valid_values[max_local_index])

		if 0 <= max_cell_id < len(centers):
			max_position = centers[max_cell_id]
			x_value = f"{float(max_position[0]):.9g}"
			y_value = f"{float(max_position[1]):.9g}"
			z_value = f"{float(max_position[2]):.9g}"
		else:
			x_value = ""
			y_value = ""
			z_value = ""

		rows.append(
			{
				"category": category_name,
				"layer_name": layer_name,
				"filtered_max": f"{max_value:.9g}",
				"x": x_value,
				"y": y_value,
				"z": z_value,
			}
		)

	output_dir.mkdir(parents=True, exist_ok=True)
	category_part = sanitize_filename_part(category_name)
	output_csv = output_dir / f"{rawxm3_path.stem}_{category_part}.csv"

	with open(output_csv, "w", encoding="utf-8-sig", newline="") as csv_file:
		writer = csv.DictWriter(
			csv_file,
			fieldnames=["category", "layer_name", "filtered_max", "x", "y", "z"],
		)
		writer.writeheader()
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
			"Batch process RAWXM3 files in a directory and export per-file CSV containing "
			"filtered maxima (value and coordinate) for all layers in a selected category."
		)
	)
	parser.add_argument("input_dir", nargs="?", type=Path, help="Directory containing RAWXM3 files")
	parser.add_argument(
		"--category",
		required=False,
		help=(
			"Category name (Transmission/Absorption/Incident/Reflection) or a custom "
			"layer-name prefix"
		),
	)
	parser.add_argument(
		"--surface-threshold-mm2",
		type=float,
		default=0.0,
		help="Surface threshold in mm^2 (converted to m^2 during filtering)",
	)
	parser.add_argument(
		"--rayhits-threshold",
		type=int,
		default=0,
		help="Ray hits threshold",
	)
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
	category: str,
	surface_threshold_mm2: float,
	rayhits_threshold: int,
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

	logger(f"Found {len(files)} RAWXM3 files.")
	logger(f"Category: {category}")
	logger(f"Surface threshold: {surface_threshold_mm2} mm^2")
	logger(f"Ray hits threshold: {rayhits_threshold}")
	logger(f"Output directory: {output_dir}")

	success = 0
	failed = 0

	for file_path in files:
		logger(f"\nProcessing: {file_path.name}")
		try:
			output_csv = export_file_category_maxima(
				rawxm3_path=file_path,
				output_dir=output_dir,
				category_input=category,
				surface_threshold_mm2=surface_threshold_mm2,
				ray_hits_threshold=rayhits_threshold,
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

		self.category_combo = QComboBox()
		self.category_combo.setEditable(True)
		self.category_combo.addItems(list(CATEGORY_PREFIXES.keys()))
		form_layout.addRow("Category", self.category_combo)

		self.surface_spin = QDoubleSpinBox()
		self.surface_spin.setRange(0.0, 1e12)
		self.surface_spin.setDecimals(6)
		self.surface_spin.setSingleStep(0.1)
		self.surface_spin.setValue(0.0)
		form_layout.addRow("Surface threshold (mm^2)", self.surface_spin)

		self.rayhits_spin = QSpinBox()
		self.rayhits_spin.setRange(0, 2147483647)
		self.rayhits_spin.setValue(0)
		form_layout.addRow("Ray hits threshold", self.rayhits_spin)

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

		category = self.category_combo.currentText().strip()
		if not category:
			QMessageBox.warning(self, "Run Batch", "Please input a category.")
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
				category=category,
				surface_threshold_mm2=float(self.surface_spin.value()),
				rayhits_threshold=int(self.rayhits_spin.value()),
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

	if not args.category:
		print("--category is required in CLI mode.")
		return 1

	input_dir = args.input_dir.resolve()
	output_dir = (args.output_dir or input_dir).resolve()
	return run_batch_processing(
		input_dir=input_dir,
		output_dir=output_dir,
		category=str(args.category),
		surface_threshold_mm2=float(args.surface_threshold_mm2),
		rayhits_threshold=int(args.rayhits_threshold),
		recursive=bool(args.recursive),
		logger=print,
	)


if __name__ == "__main__":
	raise SystemExit(main())
