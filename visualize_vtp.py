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
"""Simple UI for converting RAWXM3 to VTP and visualizing cell data with filters.

中英说明 / Bilingual note:
- 支持直接读取 RAWXM3(HDF5) 并转换为 VTP，再在同一界面可视化。
- Supports direct RAWXM3(HDF5) loading, conversion to VTP, and visualization in one UI.
"""

from __future__ import annotations

import sys
import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import h5py  # type: ignore[import-untyped]
import numpy as np
import pyvista as pv
import vtk  # type: ignore[import-untyped]
from pyvistaqt import QtInteractor  # type: ignore[import-untyped]
from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from tqdm import tqdm  # type: ignore[import-untyped]


# 过滤字段常量 / Filter field constants
SURFACE_FIELD = "Triangle Surface (m^2)"
SURFACE_THRESHOLD_LABEL = "Triangle Surface threshold (mm^2)"
RAY_HITS_FIELD = "Ray hits"
FILTER_FIELDS = {SURFACE_FIELD, RAY_HITS_FIELD}
DISPLAY_FIELD = "__display_values__"
RAWXM3_SUFFIXES = {".rawxm3", ".h5", ".hdf5"}
TEXT_SELECTABLE = Qt.TextInteractionFlag.TextSelectableByMouse
CATEGORY_PREFIXES = {
    "Transmission": "Transmission Layer",
    "Absorption": "Absorption Layer",
    "Incident": "Incident Layer",
    "Reflection": "Reflection Layer",
}


def is_numeric_array(values: object) -> bool:
    """用途 Purpose:
    - 判断输入是否可视为数值数组。
    - Check whether input can be treated as a numeric array.

    参数 Args:
    - values: 任意待检测对象。
    - values: Any object to inspect.

    返回 Returns:
    - bool: 为数值类型返回 True，否则 False。
    - bool: True if numeric dtype, else False.
    """
    array = np.asarray(values)
    return np.issubdtype(array.dtype, np.number)


def normalize_text(value: Any) -> str:
    """用途 Purpose:
    - 将 HDF5 属性中的 bytes / numpy.bytes_ 统一转为 str。
    - Normalize text-like HDF5 attribute values to Python str.

    参数 Args:
    - value: 可能为 bytes、numpy.bytes_ 或其他类型。
    - value: bytes, numpy.bytes_, or any object.

    返回 Returns:
    - str: 统一后的字符串表示。
    - str: Normalized string representation.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def read_triplets(dataset: h5py.Dataset, dataset_name: str) -> np.ndarray:
    """用途 Purpose:
    - 读取一维 triplet 数据并重排为 (N, 3)。
    - Read flat triplet data and reshape to (N, 3).

    参数 Args:
    - dataset: HDF5 数据集，预期长度可被 3 整除。
    - dataset_name: 数据集名称，用于异常信息。

    返回 Returns:
    - np.ndarray: 形状为 (N, 3) 的数组。

    异常 Raises:
    - ValueError: 数据长度不是 3 的倍数。
    """
    values = np.asarray(dataset)
    if values.size % 3 != 0:
        raise ValueError(
            f"Dataset '{dataset_name}' size {values.size} is not divisible by 3."
        )
    return values.reshape(-1, 3)


def get_faces_group(hdf5_file: h5py.File) -> tuple[h5py.Group, list[str]]:
    """用途 Purpose:
    - 从 RAWXM3 根节点中定位 faces 分组及其子键。
    - Locate the faces group and face keys from RAWXM3 root.

    参数 Args:
    - hdf5_file: 打开的 HDF5 文件对象。

    返回 Returns:
    - tuple[h5py.Group, list[str]]: faces group 与 face key 列表。

    异常 Raises:
    - ValueError: 文件为空或不存在 face 节点。
    """
    root_keys = list(hdf5_file.keys())
    if not root_keys:
        raise ValueError("RAWXM3 file is empty.")

    faces_group = hdf5_file[root_keys[0]]
    face_keys = list(faces_group.keys())
    if not face_keys:
        raise ValueError("No face entries found in RAWXM3 file.")

    return faces_group, face_keys


@dataclass(slots=True)
class Normal:
    """顶点法向量 / Vertex normal."""
    x: float
    y: float
    z: float


@dataclass(slots=True)
class Vertex:
    """三维顶点坐标 / 3D vertex position."""
    x: float
    y: float
    z: float


@dataclass(slots=True)
class Facet:
    """三角面索引 / Triangle facet indices."""
    index_1: int
    index_2: int
    index_3: int


@dataclass
class FaceFacetsData:
    """单个 face 上某个物理量的 cell 数据 / Per-face scalar values."""
    name: str
    data: list[float] = field(default_factory=list)

    @classmethod
    def from_hdf5(cls, hdf5_data: h5py.Group, facet_count: int) -> "FaceFacetsData":
        """用途 Purpose:
        - 读取单个物理量节点并展开为逐面片数据。
        - Build per-facet scalar values from one HDF5 scalar node.

        参数 Args:
        - hdf5_data: `facets_data` 下某个物理量 group。
        - facet_count: 当前 face 的面片数量。

        返回 Returns:
        - FaceFacetsData: 标量名称与逐面片值。

        异常 Raises:
        - ValueError: 标量长度与面片数量不匹配。
        """
        name = normalize_text(hdf5_data.attrs["name"])
        data_size = int(hdf5_data.attrs["data_size"])

        if data_size == 1:
            value = float(hdf5_data.attrs["data"])
            data = [value] * facet_count
        else:
            data = np.asarray(hdf5_data["data"], dtype=np.float64).ravel().tolist()
            if len(data) != facet_count:
                raise ValueError(
                    f"Facet data size mismatch for '{name}': expected {facet_count}, got {len(data)}."
                )

        return cls(name=name, data=data)


@dataclass
class FaceDescription:
    """RAWXM3 中单个 face 的几何与属性 / Full geometry+data for one RAWXM3 face."""
    facets: list[Facet] = field(default_factory=list)
    normals: list[Normal] = field(default_factory=list)
    vertices: list[Vertex] = field(default_factory=list)
    facets_data: list[FaceFacetsData] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_hdf5(
        cls,
        hdf5_face: h5py.Group,
        vertex_offset: int = 0,
    ) -> "FaceDescription":
        """用途 Purpose:
        - 从单个 face 节点读取几何、metadata 与物理量。
        - Parse geometry, metadata, and scalar data from one face group.

        参数 Args:
        - hdf5_face: RAWXM3 中的 face group。
        - vertex_offset: 合并多 face 时用于修正索引的顶点偏移。

        返回 Returns:
        - FaceDescription: 完整 face 描述对象。

        异常 Raises:
        - ValueError: metadata 键值长度不一致等数据结构错误。
        """
        normals_raw = read_triplets(hdf5_face["normals"], "normals")
        vertices_raw = read_triplets(hdf5_face["vertices"], "vertices")
        facets_raw = read_triplets(hdf5_face["facets"], "facets")

        normals = [Normal(float(x), float(y), float(z)) for x, y, z in normals_raw]
        vertices = [Vertex(float(x), float(y), float(z)) for x, y, z in vertices_raw]
        facets = [
            Facet(
                int(index_1) + vertex_offset,
                int(index_2) + vertex_offset,
                int(index_3) + vertex_offset,
            )
            for index_1, index_2, index_3 in facets_raw
        ]

        metadata_group = hdf5_face["metadata"]
        metadata_keys = [normalize_text(key) for key in metadata_group.attrs.get("keys", [])]
        metadata_values = [
            normalize_text(value) for value in metadata_group.attrs.get("values", [])
        ]
        if len(metadata_keys) != len(metadata_values):
            raise ValueError("Metadata keys and values count mismatch.")
        metadata = dict(zip(metadata_keys, metadata_values))

        facets_data_group = hdf5_face["facets_data"]
        data_keys = list(facets_data_group.keys())
        facets_data = [
            FaceFacetsData.from_hdf5(facets_data_group[key], len(facets))
            for key in data_keys
        ]

        return cls(
            facets=facets,
            normals=normals,
            vertices=vertices,
            facets_data=facets_data,
            metadata=metadata,
        )


@dataclass
class CompleteMapDescription:
    """合并后的完整网格描述 / Aggregated mesh built from all faces."""
    vertices: list[Vertex] = field(default_factory=list)
    normals: list[Normal] = field(default_factory=list)
    facets: list[Facet] = field(default_factory=list)
    facets_data: dict[str, list[float]] = field(default_factory=dict)
    metadata_summary: dict[str, set[str]] = field(default_factory=dict)
    metadata_cell_data: dict[str, list[str]] = field(default_factory=dict)

    def add_face(self, face: FaceDescription) -> None:
        """用途 Purpose:
        - 将一个 face 追加到总网格，并同步合并 metadata 与标量。
        - Append one face into aggregate mesh and merge metadata/scalars.

        参数 Args:
        - face: 待合并的 face 描述对象。

        异常 Raises:
        - ValueError: 不同 face 的标量 schema 不一致。
        """
        previous_cell_count = len(self.facets)
        current_cell_count = len(face.facets)

        self.vertices.extend(face.vertices)
        self.normals.extend(face.normals)
        self.facets.extend(face.facets)

        existing_metadata_keys = set(self.metadata_cell_data)
        current_metadata_keys = set(face.metadata)

        for missing_key in existing_metadata_keys - current_metadata_keys:
            self.metadata_cell_data[missing_key].extend([""] * current_cell_count)

        for key, value in face.metadata.items():
            if key not in self.metadata_cell_data:
                self.metadata_cell_data[key] = [""] * previous_cell_count
            self.metadata_cell_data[key].extend([value] * current_cell_count)
            self.metadata_summary.setdefault(key, set()).add(value)

        face_data_by_name = {item.name: item.data for item in face.facets_data}
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

        for name, numeric_values in face_data_by_name.items():
            self.facets_data[name].extend(numeric_values)

    def to_mesh(self) -> pv.PolyData:
        """用途 Purpose:
        - 把聚合后的顶点/面片/属性转换为 `pyvista.PolyData`。
        - Convert aggregated geometry and arrays into `pyvista.PolyData`.

        返回 Returns:
        - pv.PolyData: 可直接渲染与导出的三角网格。

        异常 Raises:
        - ValueError: 几何为空或数组长度与点/单元数量不一致。
        """
        if not self.vertices or not self.facets:
            raise ValueError("No geometry available to export.")

        points = np.asarray(
            [[vertex.x, vertex.y, vertex.z] for vertex in self.vertices],
            dtype=float,
        )

        faces = np.empty((len(self.facets), 4), dtype=int)
        faces[:, 0] = 3
        faces[:, 1] = [facet.index_1 for facet in self.facets]
        faces[:, 2] = [facet.index_2 for facet in self.facets]
        faces[:, 3] = [facet.index_3 for facet in self.facets]

        mesh = pv.PolyData(cast(Any, points), cast(Any, faces.ravel()))

        if self.normals:
            if len(self.normals) != mesh.n_points:
                raise ValueError(
                    f"Normal count mismatch: expected {mesh.n_points}, got {len(self.normals)}."
                )
            normals = np.asarray(
                [[normal.x, normal.y, normal.z] for normal in self.normals],
                dtype=float,
            )
            mesh.point_data["Normals"] = normals

        for name, numeric_values in self.facets_data.items():
            data = np.asarray(numeric_values, dtype=float)
            if data.size != mesh.n_cells:
                raise ValueError(
                    f"Cell data size mismatch for '{name}': expected {mesh.n_cells}, got {data.size}."
                )
            mesh.cell_data[name] = data

        for metadata_key, metadata_values in self.metadata_cell_data.items():
            if len(metadata_values) != mesh.n_cells:
                raise ValueError(
                    f"Metadata cell data size mismatch for '{metadata_key}': expected {mesh.n_cells}, got {len(metadata_values)}."
                )
            mesh.cell_data[f"meta_{metadata_key}"] = np.asarray(metadata_values)

        for metadata_key, summary_values in self.metadata_summary.items():
            mesh.field_data[f"meta_{metadata_key}_summary"] = np.asarray(
                sorted(summary_values)
            )

        return mesh

    def save_vtp(self, filename: str | Path) -> pv.PolyData:
        """用途 Purpose:
        - 导出当前聚合网格为 VTP 文件。
        - Save aggregated mesh to a VTP file.

        参数 Args:
        - filename: 输出路径。

        返回 Returns:
        - pv.PolyData: 导出使用的网格对象。
        """
        mesh = self.to_mesh()
        mesh.save(str(filename), binary=True)
        return mesh


def convert_rawxm3_to_map(rawxm3_path: str | Path) -> CompleteMapDescription:
    """用途 Purpose:
    - 遍历 RAWXM3 中所有 face，构建完整聚合网格描述。
    - Iterate all faces in RAWXM3 and build aggregated map description.

    参数 Args:
    - rawxm3_path: RAWXM3 文件路径。

    返回 Returns:
    - CompleteMapDescription: 后续可转换为 PolyData 或导出 VTP 的对象。
    """
    complete_map = CompleteMapDescription()
    vertex_offset = 0

    with h5py.File(rawxm3_path, "r") as hdf5_file:
        faces_group, face_keys = get_faces_group(hdf5_file)

        for key in tqdm(face_keys, desc="Processing Faces", unit="face", ncols=90):
            face = FaceDescription.from_hdf5(
                faces_group[key],
                vertex_offset=vertex_offset,
            )
            vertex_offset += len(face.vertices)
            complete_map.add_face(face)

    return complete_map


class ScientificDoubleSpinBox(QDoubleSpinBox):
    """支持科学计数法输入 / Double spin box with scientific-notation validation."""
    def validate(self, text: str, pos: int) -> tuple[QValidator.State, str, int]:
        """用途 Purpose:
        - 扩展输入校验，允许科学计数法输入过程中的中间态。
        - Accept intermediate scientific-notation text during typing.

        参数 Args:
        - text: 当前输入文本。
        - pos: 当前光标位置。

        返回 Returns:
        - tuple[QValidator.State, str, int]: Qt 校验结果三元组。
        """
        stripped = text.strip()
        if stripped in {"", "+", "-", ".", "+.", "-.", "e", "E", "+e", "-e"}:
            return QValidator.State.Intermediate, text, pos

        try:
            float(stripped)
        except ValueError:
            if stripped.lower().endswith(("e", "e+", "e-")):
                return QValidator.State.Intermediate, text, pos
            return QValidator.State.Invalid, text, pos

        return QValidator.State.Acceptable, text, pos

    def valueFromText(self, text: str) -> float:
        """将文本解析为浮点数 / Parse editor text to float value."""
        try:
            return float(text.strip())
        except ValueError:
            return self.minimum()

    def textFromValue(self, value: float) -> str:
        """格式化显示文本 / Format float value for editor display."""
        return f"{value:.12g}"


class VtpViewerWindow(QMainWindow):
    """主界面窗口 / Main UI window for RAWXM3/VTP loading and visualization."""
    def __init__(self, initial_path: str | None = None) -> None:
        """用途 Purpose:
        - 初始化窗口状态、渲染器与交互对象。
        - Initialize UI state, renderer, and interaction helpers.

        参数 Args:
        - initial_path: 启动时可选输入路径。
        """
        super().__init__()
        self.setWindowTitle("RAWXM3 / VTP Physical Quantity Viewer")
        self.resize(1400, 900)

        self.mesh: pv.PolyData | None = None
        self.actor = None
        self.scalar_names: list[str] = []
        self.current_scalar_name = ""
        self.current_display_values: np.ndarray | None = None
        self.default_color_limits: tuple[float, float] = (0.0, 1.0)
        self.custom_color_limits: tuple[float, float] | None = None
        self._updating_colorbar_controls = False
        self.measurement_enabled = False
        self.max_measurement_enabled = False
        self.max_marker_actor = None
        self.max_label_actor = None
        self.category_compare_enabled = False
        self.layer_categories: dict[str, list[str]] = {}
        self.category_marker_actor = None
        self.category_label_actor = None
        self.generated_vtp_path: Path | None = None
        self.cell_picker = vtk.vtkCellPicker()
        self.cell_picker.SetTolerance(0.0005)

        self._build_ui()

        if initial_path:
            self.load_file(initial_path)

    def _build_ui(self) -> None:
        """用途 Purpose:
        - 创建左侧控制区与右侧三维视图，并绑定信号槽。
        - Build control panel + 3D view and wire UI signals.

        返回 Returns:
        - None
        """
        central = QWidget(self)
        self.setCentralWidget(central)

        root_layout = QHBoxLayout(central)

        control_panel = QWidget(self)
        control_panel.setMinimumWidth(320)
        control_panel.setMaximumWidth(420)
        control_layout = QVBoxLayout(control_panel)

        self.open_button = QPushButton("Open RAWXM3 / VTP")
        self.open_button.clicked.connect(self.choose_file)
        control_layout.addWidget(self.open_button)

        self.file_label = QLabel("No file loaded")
        self.file_label.setWordWrap(True)
        self.file_label.setTextInteractionFlags(TEXT_SELECTABLE)
        control_layout.addWidget(self.file_label)

        form_layout = QFormLayout()

        self.scalar_combo = QComboBox()
        self.scalar_combo.currentIndexChanged.connect(self.on_scalar_changed)
        self.scalar_combo.setEnabled(False)
        form_layout.addRow("Display layer", self.scalar_combo)

        self.colorbar_min_spin = ScientificDoubleSpinBox()
        self.colorbar_min_spin.setDecimals(6)
        self.colorbar_min_spin.setRange(-1e12, 1e12)
        self.colorbar_min_spin.setSingleStep(0.001)
        self.colorbar_min_spin.valueChanged.connect(self.on_colorbar_limits_changed)
        self.colorbar_min_spin.setEnabled(False)
        form_layout.addRow("Colorbar min", self.colorbar_min_spin)

        self.colorbar_max_spin = ScientificDoubleSpinBox()
        self.colorbar_max_spin.setDecimals(6)
        self.colorbar_max_spin.setRange(-1e12, 1e12)
        self.colorbar_max_spin.setSingleStep(0.001)
        self.colorbar_max_spin.valueChanged.connect(self.on_colorbar_limits_changed)
        self.colorbar_max_spin.setEnabled(False)
        form_layout.addRow("Colorbar max", self.colorbar_max_spin)

        self.surface_threshold = ScientificDoubleSpinBox()
        self.surface_threshold.setDecimals(6)
        self.surface_threshold.setRange(0.0, 1e12)
        self.surface_threshold.setSingleStep(0.001)
        self.surface_threshold.valueChanged.connect(self.update_visualization)
        self.surface_threshold.setEnabled(False)
        form_layout.addRow(SURFACE_THRESHOLD_LABEL, self.surface_threshold)

        self.ray_hits_threshold = QSpinBox()
        self.ray_hits_threshold.setRange(0, 2147483647)
        self.ray_hits_threshold.setSingleStep(1)
        self.ray_hits_threshold.valueChanged.connect(self.update_visualization)
        self.ray_hits_threshold.setEnabled(False)
        form_layout.addRow(RAY_HITS_FIELD, self.ray_hits_threshold)

        control_layout.addLayout(form_layout)

        self.reset_button = QPushButton("Reset thresholds")
        self.reset_button.clicked.connect(self.reset_thresholds)
        self.reset_button.setEnabled(False)
        control_layout.addWidget(self.reset_button)

        self.reset_colorbar_button = QPushButton("Reset colorbar")
        self.reset_colorbar_button.clicked.connect(self.reset_colorbar_limits)
        self.reset_colorbar_button.setEnabled(False)
        control_layout.addWidget(self.reset_colorbar_button)

        self.measure_checkbox = QCheckBox("Enable hover measurement")
        self.measure_checkbox.toggled.connect(self.toggle_measurement)
        self.measure_checkbox.setEnabled(False)
        control_layout.addWidget(self.measure_checkbox)

        self.max_measure_checkbox = QCheckBox("Enable max measurement")
        self.max_measure_checkbox.toggled.connect(self.toggle_max_measurement)
        self.max_measure_checkbox.setEnabled(False)
        control_layout.addWidget(self.max_measure_checkbox)

        self.category_compare_checkbox = QCheckBox("Enable category max compare")
        self.category_compare_checkbox.toggled.connect(self.toggle_category_compare)
        self.category_compare_checkbox.setEnabled(False)
        control_layout.addWidget(self.category_compare_checkbox)

        self.export_category_csv_button = QPushButton("Export category layers CSV")
        self.export_category_csv_button.clicked.connect(self.export_category_layers_csv)
        self.export_category_csv_button.setEnabled(False)
        control_layout.addWidget(self.export_category_csv_button)

        self.category_combo = QComboBox()
        self.category_combo.addItems(list(CATEGORY_PREFIXES.keys()))
        self.category_combo.currentIndexChanged.connect(self.update_category_compare)
        self.category_combo.setEnabled(False)
        form_layout.addRow("Layer category", self.category_combo)

        self.info_label = QLabel(
            "Load a RAWXM3 or VTP file. RAWXM3 files are converted to VTP automatically "
            "and then visualized."
        )
        self.info_label.setWordWrap(True)
        self.info_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)
        control_layout.addWidget(self.info_label)

        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        self.stats_label.setTextInteractionFlags(TEXT_SELECTABLE)
        control_layout.addWidget(self.stats_label)

        self.measurement_label = QLabel("Measurement: Off")
        self.measurement_label.setWordWrap(True)
        self.measurement_label.setTextInteractionFlags(TEXT_SELECTABLE)
        control_layout.addWidget(self.measurement_label)

        self.max_measure_label = QLabel("Max measurement: Off")
        self.max_measure_label.setWordWrap(True)
        self.max_measure_label.setTextInteractionFlags(TEXT_SELECTABLE)
        control_layout.addWidget(self.max_measure_label)

        self.category_compare_label = QLabel("Category compare: Off")
        self.category_compare_label.setWordWrap(True)
        self.category_compare_label.setTextInteractionFlags(TEXT_SELECTABLE)
        control_layout.addWidget(self.category_compare_label)

        control_layout.addStretch(1)

        self.plotter = QtInteractor(cast(Any, self))
        self.plotter.set_background("#202124")
        self.plotter.add_axes()
        self.plotter.interactor.AddObserver("MouseMoveEvent", self.on_mouse_move)

        root_layout.addWidget(control_panel)
        root_layout.addWidget(self.plotter.interactor, 1)

    def choose_file(self) -> None:
        """用途 Purpose:
        - 打开文件选择框并触发统一加载入口。
        - Open file dialog and dispatch to the unified loader.
        """
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open RAWXM3 or VTP file",
            "",
            (
                "Supported files (*.rawxm3 *.h5 *.hdf5 *.vtp *.vtk);;"
                "RAWXM3 files (*.rawxm3 *.h5 *.hdf5);;"
                "VTP files (*.vtp *.vtk);;All files (*.*)"
            ),
        )
        if file_path:
            self.load_file(file_path)

    def load_file(self, file_path: str) -> None:
        """用途 Purpose:
        - 根据后缀在 RAWXM3 转换路径与 VTP 读取路径之间分流。
        - Route by suffix to RAWXM3 conversion or VTP loading.
        """
        suffix = Path(file_path).suffix.lower()
        if suffix in RAWXM3_SUFFIXES:
            self.load_rawxm3(file_path)
            return
        self.load_vtp(file_path)

    def load_rawxm3(self, file_path: str) -> None:
        """用途 Purpose:
        - 将 RAWXM3 转为网格、落盘 VTP，并加载到当前可视化窗口。
        - Convert RAWXM3 to mesh, save VTP, and load it into current viewer.

        参数 Args:
        - file_path: RAWXM3 输入路径。

        异常处理 Errors:
        - 转换失败时弹窗提示，不抛出到 UI 事件循环外。
        - Conversion errors are shown in a dialog and handled in-place.
        """
        rawxm3_path = Path(file_path).resolve()
        generated_vtp_path = rawxm3_path.with_name(f"{rawxm3_path.stem}_with_metadata.vtp")

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        QApplication.processEvents()
        try:
            complete_map = convert_rawxm3_to_map(rawxm3_path)
            mesh = complete_map.to_mesh()
            mesh.save(str(generated_vtp_path), binary=True)
        except Exception as exc:
            QMessageBox.critical(
                self,
                "RAWXM3 load error",
                f"Failed to convert RAWXM3 file:\n{exc}",
            )
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.generated_vtp_path = generated_vtp_path
        self._apply_loaded_mesh(
            mesh,
            source_label=(
                f"Source: {rawxm3_path}\n"
                f"Generated VTP: {generated_vtp_path}"
            ),
        )
        self.info_label.setText(
            "RAWXM3 converted successfully. The generated VTP has been saved and loaded for visualization."
        )

    def load_vtp(self, file_path: str) -> None:
        """用途 Purpose:
        - 读取现有 VTP/VTK 文件并标准化为 PolyData。
        - Load an existing VTP/VTK dataset and normalize to PolyData.
        """
        try:
            loaded = pv.read(file_path)
        except Exception as exc:
            QMessageBox.critical(self, "Load error", f"Failed to load file:\n{exc}")
            return

        mesh = self._ensure_polydata(loaded)
        if mesh is None:
            QMessageBox.critical(
                self,
                "Unsupported mesh",
                "The file could not be converted to PolyData.",
            )
            return

        self.generated_vtp_path = None
        self._apply_loaded_mesh(mesh, source_label=str(Path(file_path).resolve()))

    def _ensure_polydata(self, mesh: Any) -> pv.PolyData | None:
        """确保输入对象可用于面片渲染 / Ensure input dataset can be rendered as PolyData.

        中文：
        - 若已是 `pv.PolyData`，直接返回。
        - 否则尝试抽取表面并三角化，失败返回 `None`。

        English:
        - Return directly if already `pv.PolyData`.
        - Otherwise try `extract_surface().triangulate()`, return `None` on failure.
        """
        if isinstance(mesh, pv.PolyData):
            return mesh

        try:
            surface = mesh.extract_surface().triangulate()
        except Exception:
            return None

        if isinstance(surface, pv.PolyData):
            return surface
        return cast(pv.PolyData, surface)

    def _apply_loaded_mesh(self, mesh: pv.PolyData, source_label: str) -> None:
        """用途 Purpose:
        - 将新网格应用到窗口状态并初始化控件。
        - Apply a new mesh to window state and initialize controls.

        参数 Args:
        - mesh: 已准备好的 PolyData。
        - source_label: 显示在 UI 的来源文本。
        """
        numeric_names = [
            name for name in mesh.cell_data.keys() if is_numeric_array(mesh.cell_data[name])
        ]
        if not numeric_names:
            QMessageBox.warning(self, "No scalar data", "No numeric cell_data arrays were found.")
            return

        self.mesh = mesh
        self.scalar_names = numeric_names
        self.file_label.setText(source_label)

        self.scalar_combo.blockSignals(True)
        self.scalar_combo.clear()
        for layer_name in sorted(self.scalar_names):
            self.scalar_combo.addItem(layer_name)
            item_index = self.scalar_combo.count() - 1
            self.scalar_combo.setItemData(
                item_index,
                layer_name,
                Qt.ItemDataRole.ToolTipRole,
            )
        self.scalar_combo.blockSignals(False)
        self.scalar_combo.setEnabled(True)
        self.layer_categories = self._build_layer_categories(self.scalar_names)
        self._initialize_category_controls()

        self._initialize_threshold_controls()
        self.reset_button.setEnabled(True)
        self.reset_colorbar_button.setEnabled(True)
        self.measure_checkbox.setEnabled(True)
        self.max_measure_checkbox.setEnabled(True)
        self.colorbar_min_spin.setEnabled(True)
        self.colorbar_max_spin.setEnabled(True)
        self._clear_measurement_text()
        self._clear_max_measure_text()
        self.custom_color_limits = None

        default_index = 0
        for preferred_name in self.scalar_names:
            if preferred_name not in FILTER_FIELDS:
                default_index = self.scalar_names.index(preferred_name)
                break
        self.scalar_combo.setCurrentIndex(default_index)

        self.update_visualization()

    def _compute_default_color_limits(self, values: np.ndarray) -> tuple[float, float]:
        """用途 Purpose:
        - 依据当前显示值计算默认 colorbar 范围。
        - Compute default colorbar limits from current display values.

        参数 Args:
        - values: 当前用于渲染的标量数组。

        返回 Returns:
        - tuple[float, float]: 合法的 (min, max) 范围。
        """
        finite_values = np.asarray(values, dtype=np.float64)
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size == 0:
            return (0.0, 1.0)

        min_value = float(np.min(finite_values))
        max_value = float(np.max(finite_values))
        if min_value < max_value:
            return (min_value, max_value)

        padding = max(abs(min_value) * 0.05, 1.0)
        return (min_value - padding, max_value + padding)

    def _normalize_color_limits(
        self,
        min_value: float,
        max_value: float,
        fallback: tuple[float, float],
    ) -> tuple[float, float]:
        """用途 Purpose:
        - 规范化 colorbar 范围，避免最小值大于等于最大值。
        - Normalize colorbar limits and avoid invalid min/max order.
        """
        low = float(min_value)
        high = float(max_value)
        if low < high:
            return (low, high)

        fallback_low, fallback_high = fallback
        if fallback_low < fallback_high:
            return (fallback_low, fallback_high)

        padding = max(abs(low) * 0.05, 1.0)
        return (low - padding, high + padding)

    def _set_colorbar_controls(self, min_value: float, max_value: float) -> None:
        """用途 Purpose:
        - 在不触发二次刷新时同步 colorbar 输入框。
        - Synchronize colorbar inputs without triggering recursive updates.
        """
        self._updating_colorbar_controls = True
        self.colorbar_min_spin.blockSignals(True)
        self.colorbar_max_spin.blockSignals(True)
        self.colorbar_min_spin.setValue(min_value)
        self.colorbar_max_spin.setValue(max_value)
        self.colorbar_min_spin.blockSignals(False)
        self.colorbar_max_spin.blockSignals(False)
        self._updating_colorbar_controls = False

    def on_scalar_changed(self) -> None:
        """用途 Purpose:
        - 切换显示层时恢复该层默认 colorbar 范围并刷新。
        - Reset to default colorbar limits for the new layer and refresh.
        """
        self.custom_color_limits = None
        self.update_visualization()

    def on_colorbar_limits_changed(self) -> None:
        """用途 Purpose:
        - 用户修改 colorbar 最小/最大值后应用自定义范围。
        - Apply custom colorbar limits after user edits min/max controls.
        """
        if self.mesh is None or self._updating_colorbar_controls:
            return

        normalized_limits = self._normalize_color_limits(
            self.colorbar_min_spin.value(),
            self.colorbar_max_spin.value(),
            self.default_color_limits,
        )
        self.custom_color_limits = normalized_limits
        self._set_colorbar_controls(*normalized_limits)
        self.update_visualization()

    def reset_colorbar_limits(self) -> None:
        """用途 Purpose:
        - 恢复当前层默认 colorbar 范围。
        - Restore default colorbar limits for the current layer.
        """
        self.custom_color_limits = None
        self._set_colorbar_controls(*self.default_color_limits)
        self.update_visualization()

    def _initialize_threshold_controls(self) -> None:
        """初始化阈值控件状态 / Initialize threshold widgets and availability.

        中文：重置阈值为 0，并根据网格中是否存在 `Triangle Surface` 与 `Ray hits`
        字段启用/禁用对应控件，同时刷新说明文本。

        English: Reset thresholds to 0, enable/disable controls based on whether
        `Triangle Surface` and `Ray hits` arrays exist, and update help text.
        """
        assert self.mesh is not None

        surface_values = self._get_numeric_cell_data(SURFACE_FIELD)
        ray_hits_values = self._get_numeric_cell_data(RAY_HITS_FIELD)

        self.surface_threshold.blockSignals(True)
        self.ray_hits_threshold.blockSignals(True)

        self.surface_threshold.setValue(0.0)
        self.ray_hits_threshold.setValue(0)

        self.surface_threshold.setEnabled(surface_values is not None)
        self.ray_hits_threshold.setEnabled(ray_hits_values is not None)

        self.surface_threshold.blockSignals(False)
        self.ray_hits_threshold.blockSignals(False)

        missing_filters = [
            name
            for name, values in (
                (SURFACE_FIELD, surface_values),
                (RAY_HITS_FIELD, ray_hits_values),
            )
            if values is None
        ]

        if missing_filters:
            self.info_label.setText(
                "Loaded file, but these filter arrays are missing: "
                + ", ".join(missing_filters)
                + ". Missing filters are ignored."
            )
        elif self.generated_vtp_path is None:
            self.info_label.setText(
                "For non-filter layers, cells are set to 0 only when both Surface and Ray hits are below thresholds."
            )

    def _get_numeric_cell_data(self, name: str) -> np.ndarray | None:
        """获取指定 cell_data 的数值数组 / Get numeric cell_data array by name.

        返回 `float64` 视图（尽量不拷贝）；不存在或非数值类型时返回 `None`。
        Returns a `float64` view when possible; returns `None` if missing/non-numeric.
        """
        if self.mesh is None or name not in self.mesh.cell_data:
            return None

        values = np.asarray(self.mesh.cell_data[name])
        if not np.issubdtype(values.dtype, np.number):
            return None
        return values.astype(np.float64, copy=False)

    def _get_surface_threshold_m2(self) -> float:
        """用途 Purpose:
        - 将界面输入的 mm² 阈值换算为 m²。
        - Convert the UI surface threshold from mm² to m².

        返回 Returns:
        - float: 以 m² 为单位的过滤阈值。
        """
        return self.surface_threshold.value() / 1_000_000.0

    def build_filter_mask(self) -> np.ndarray:
        """用途 Purpose:
        - 根据 Surface 与 Ray hits 阈值生成单元可见掩码。
        - Build a cell visibility mask from threshold controls.

        返回 Returns:
        - np.ndarray: `bool` 掩码，长度等于 `mesh.n_cells`。
        """
        assert self.mesh is not None

        surface_fail: np.ndarray | None = None
        ray_hits_fail: np.ndarray | None = None

        surface_values = self._get_numeric_cell_data(SURFACE_FIELD)
        surface_threshold_m2 = self._get_surface_threshold_m2()
        if surface_values is not None and surface_threshold_m2 > 0:
            surface_fail = surface_values < surface_threshold_m2

        ray_hits_values = self._get_numeric_cell_data(RAY_HITS_FIELD)
        if ray_hits_values is not None and self.ray_hits_threshold.value() > 0:
            ray_hits_fail = ray_hits_values < self.ray_hits_threshold.value()

        if surface_fail is not None and ray_hits_fail is not None:
            filtered_mask = surface_fail & ray_hits_fail
            return ~filtered_mask

        if surface_fail is not None:
            return ~surface_fail

        if ray_hits_fail is not None:
            return ~ray_hits_fail

        return np.ones(self.mesh.n_cells, dtype=bool)

    def build_display_values(self, scalar_name: str) -> np.ndarray:
        """用途 Purpose:
        - 读取指定标量并应用阈值掩码，生成显示数组。
        - Build display array by applying filter mask to selected scalar.

        参数 Args:
        - scalar_name: 要显示的 cell_data 名称。

        返回 Returns:
        - np.ndarray: 过滤后用于渲染的标量数组。

        异常 Raises:
        - ValueError: 所选标量不存在或非数值类型。
        """
        assert self.mesh is not None

        values = self._get_numeric_cell_data(scalar_name)
        if values is None:
            raise ValueError(f"Selected layer '{scalar_name}' is not numeric.")

        mask = self.build_filter_mask()
        return np.where(mask, values, 0.0)

    def update_visualization(self) -> None:
        """用途 Purpose:
        - 依据当前层与阈值重新渲染，并更新统计与辅助标注。
        - Re-render by current scalar/thresholds and refresh labels/markers.

        返回 Returns:
        - None
        """
        if self.mesh is None or not self.scalar_names:
            return

        scalar_name = self.scalar_combo.currentText()
        if not scalar_name:
            return

        try:
            display_values = self.build_display_values(scalar_name)
        except Exception as exc:
            QMessageBox.critical(self, "Visualization error", str(exc))
            return

        self.default_color_limits = self._compute_default_color_limits(display_values)
        active_color_limits = self.default_color_limits
        if self.custom_color_limits is not None:
            active_color_limits = self._normalize_color_limits(
                self.custom_color_limits[0],
                self.custom_color_limits[1],
                self.default_color_limits,
            )
            self.custom_color_limits = active_color_limits

        self._set_colorbar_controls(*active_color_limits)

        self.mesh.cell_data[DISPLAY_FIELD] = display_values
        self.current_scalar_name = scalar_name
        self.current_display_values = display_values

        camera_position = self.plotter.camera_position if self.actor is not None else None

        self.plotter.clear()
        self.plotter.add_axes()
        self.actor = self.plotter.add_mesh(
            self.mesh,
            scalars=DISPLAY_FIELD,
            clim=active_color_limits,
            show_edges=False,
            render_points_as_spheres=False,
            cmap="viridis",
            scalar_bar_args={"title": scalar_name},
            show_scalar_bar=True,
            copy_mesh=False,
        )

        if camera_position is not None:
            self.plotter.camera_position = camera_position
        else:
            self.plotter.reset_camera()

        self.plotter.render()
        self._update_stats_label(scalar_name, display_values)
        if self.category_compare_enabled:
            self.update_category_compare()
        else:
            self._clear_category_marker()
            self._clear_category_compare_text()
        if self.max_measurement_enabled:
            self.update_max_measurement_marker()
        else:
            self._clear_max_marker()
            self._clear_max_measure_text()
        if self.measurement_enabled:
            self.update_measurement_at_cursor()
        else:
            self._clear_measurement_text()

    def _update_stats_label(self, scalar_name: str, values: np.ndarray) -> None:
        """用途 Purpose:
        - 生成并刷新统计文本（层名、通过阈值数量、最值、阈值）。
        - Build and refresh stats text (layer, passed count, min/max, thresholds).

        参数 Args:
        - scalar_name: 当前显示层名称。
        - values: 当前用于显示的标量数组。
        """
        mask = self.build_filter_mask()
        visible_count = int(np.count_nonzero(mask))
        total_count = int(values.size)

        if values.size == 0:
            stats_text = "No cells available."
        else:
            stats_text = (
                f"Layer: {scalar_name}\n"
                f"Cells passing threshold: {visible_count}/{total_count}\n"
                f"Displayed min/max: {float(np.min(values)):.6g} / {float(np.max(values)):.6g}\n"
                f"Colorbar min/max: {self.colorbar_min_spin.value():.6g} / {self.colorbar_max_spin.value():.6g}\n"
                f"Surface threshold (mm^2): {self.surface_threshold.value():.6g}\n"
                f"Ray hits threshold: {self.ray_hits_threshold.value():.6g}"
            )

        self.stats_label.setText(stats_text)

    def reset_thresholds(self) -> None:
        """用途 Purpose:
        - 将 Surface 与 Ray hits 阈值恢复为 0。
        - Reset Surface and Ray hits thresholds to 0.
        """
        self.surface_threshold.setValue(0.0)
        self.ray_hits_threshold.setValue(0)

    def toggle_measurement(self, enabled: bool) -> None:
        """用途 Purpose:
        - 启用/关闭鼠标悬停测量信息。
        - Enable/disable hover measurement display.

        参数 Args:
        - enabled: 是否启用该功能。
        """
        self.measurement_enabled = enabled
        if enabled:
            self.measurement_label.setText(
                "Measurement: Move the cursor over the geometry view to inspect the nearest cell."
            )
            self.update_measurement_at_cursor()
        else:
            self._clear_measurement_text()

    def toggle_max_measurement(self, enabled: bool) -> None:
        """用途 Purpose:
        - 启用/关闭当前层最大值自动标注。
        - Enable/disable max-value marker for current layer.

        参数 Args:
        - enabled: 是否启用该功能。
        """
        self.max_measurement_enabled = enabled
        if enabled:
            self.update_max_measurement_marker()
        else:
            self._clear_max_marker()
            self._clear_max_measure_text()

    def toggle_category_compare(self, enabled: bool) -> None:
        """用途 Purpose:
        - 启用/关闭分类层间最大值比较。
        - Enable/disable category-wise max comparison.

        参数 Args:
        - enabled: 是否启用该功能。
        """
        self.category_compare_enabled = enabled
        if enabled:
            self.update_category_compare()
        else:
            self._clear_category_marker()
            self._clear_category_compare_text()

    def on_mouse_move(self, _obj: object, _event: object) -> None:
        """用途 Purpose:
        - 处理三维视图鼠标移动事件。
        - Handle mouse-move events from the 3D interactor.

        行为 Behavior:
        - 仅在测量模式启用时更新光标测量信息。
        - Updates hover measurement only when measurement mode is enabled.
        """
        if self.measurement_enabled:
            self.update_measurement_at_cursor()

    def update_measurement_at_cursor(self) -> None:
        """用途 Purpose:
        - 计算并展示光标所在单元的测量信息。
        - Compute and show measurement for the hovered cell.

        输出 Output:
        - 更新 `measurement_label`，包含 cell id、原始值、显示值、位置与过滤提示。
        - Updates `measurement_label` with cell id, raw/display values, position,
          and threshold-filter note when applicable.
        """
        if self.mesh is None or self.current_display_values is None or not self.current_scalar_name:
            self._clear_measurement_text()
            return

        x_pos, y_pos = self.plotter.interactor.GetEventPosition()
        if x_pos == 0 and y_pos == 0:
            return

        self.cell_picker.Pick(x_pos, y_pos, 0, self.plotter.renderer)
        cell_id = self.cell_picker.GetCellId()
        if cell_id < 0 or cell_id >= self.mesh.n_cells:
            self.measurement_label.setText("Measurement: No cell under cursor.")
            return

        raw_values = self._get_numeric_cell_data(self.current_scalar_name)
        if raw_values is None:
            self.measurement_label.setText("Measurement: Current layer is not numeric.")
            return

        display_value = float(self.current_display_values[cell_id])
        raw_value = float(raw_values[cell_id])
        pick_position = self.cell_picker.GetPickPosition()
        # 若原始值非 0 但显示值为 0，则说明被阈值过滤。
        # If raw value is non-zero but displayed value is zero, it is filtered out.
        is_filtered = display_value == 0.0 and raw_value != 0.0

        extra_lines = []
        surface_values = self._get_numeric_cell_data(SURFACE_FIELD)
        if surface_values is not None:
            extra_lines.append(f"{SURFACE_FIELD}: {float(surface_values[cell_id]):.6g}")

        ray_hits_values = self._get_numeric_cell_data(RAY_HITS_FIELD)
        if ray_hits_values is not None:
            extra_lines.append(f"{RAY_HITS_FIELD}: {float(ray_hits_values[cell_id]):.6g}")

        filter_note = "\nFiltered to 0 by current thresholds." if is_filtered else ""
        extra_text = "\n" + "\n".join(extra_lines) if extra_lines else ""

        self.measurement_label.setText(
            f"Measurement: On\n"
            f"Cell: {cell_id}\n"
            f"Layer: {self.current_scalar_name}\n"
            f"Displayed: {display_value:.6g}\n"
            f"Raw: {raw_value:.6g}\n"
            f"Position: ({pick_position[0]:.4f}, {pick_position[1]:.4f}, {pick_position[2]:.4f})"
            f"{extra_text}{filter_note}"
        )

    def _clear_measurement_text(self) -> None:
        """用途 Purpose:
        - 根据功能开关状态重置测量标签文本。
        - Reset measurement label text according to feature state.
        """
        if self.measurement_enabled:
            self.measurement_label.setText(
                "Measurement: Move the cursor over the geometry view to inspect the nearest cell."
            )
        else:
            self.measurement_label.setText("Measurement: Off")

    def update_max_measurement_marker(self) -> None:
        """用途 Purpose:
        - 在当前显示数组中定位最大有限值并绘制红色标记。
        - Locate max finite value in current display array and draw a red marker.

        输出 Output:
        - 更新 `max_measure_label`，并在 3D 视图添加/刷新 `MAX` 标注。
        - Updates `max_measure_label` and refreshes `MAX` marker in 3D view.
        """
        if self.mesh is None or self.current_display_values is None or not self.current_scalar_name:
            self._clear_max_marker()
            self._clear_max_measure_text()
            return

        values = np.asarray(self.current_display_values, dtype=np.float64)
        if values.size == 0:
            self._clear_max_marker()
            self.max_measure_label.setText("Max measurement: No cells available.")
            return

        finite_mask = np.isfinite(values)
        if not np.any(finite_mask):
            self._clear_max_marker()
            self.max_measure_label.setText("Max measurement: No finite values in current layer.")
            return

        valid_indices = np.where(finite_mask)[0]
        valid_values = values[finite_mask]
        max_local_index = int(np.argmax(valid_values))
        max_cell_id = int(valid_indices[max_local_index])
        max_value = float(valid_values[max_local_index])

        centers = self.mesh.cell_centers().points
        if max_cell_id < 0 or max_cell_id >= len(centers):
            self._clear_max_marker()
            self.max_measure_label.setText("Max measurement: Failed to resolve max cell center.")
            return

        max_position = centers[max_cell_id]
        self._clear_max_marker()

        marker_points = np.array([max_position], dtype=np.float32)
        self.max_marker_actor = self.plotter.add_points(
            marker_points,
            color="red",
            point_size=14,
            render_points_as_spheres=True,
        )

        self.max_label_actor = self.plotter.add_point_labels(
            marker_points,
            ["MAX"],
            font_size=12,
            show_points=False,
            always_visible=True,
        )

        self.max_measure_label.setText(
            f"Max measurement: On\n"
            f"Layer: {self.current_scalar_name}\n"
            f"Max value: {max_value:.6g}\n"
            f"Cell: {max_cell_id}\n"
            f"Position: ({float(max_position[0]):.4f}, {float(max_position[1]):.4f}, {float(max_position[2]):.4f})"
        )

        self.plotter.render()

    def _clear_max_marker(self) -> None:
        """用途 Purpose:
        - 删除最大值相关的点标记与文字标记 actor。
        - Remove point and label actors used by max marker.
        """
        if self.max_marker_actor is not None:
            try:
                self.plotter.remove_actor(self.max_marker_actor)
            except Exception:
                pass
            self.max_marker_actor = None

        if self.max_label_actor is not None:
            try:
                self.plotter.remove_actor(self.max_label_actor)
            except Exception:
                pass
            self.max_label_actor = None

    def _clear_max_measure_text(self) -> None:
        """用途 Purpose:
        - 根据最大值模式开关状态重置文本提示。
        - Reset max-measurement text by current feature state.
        """
        if self.max_measurement_enabled:
            self.max_measure_label.setText("Max measurement: Waiting for current layer values.")
        else:
            self.max_measure_label.setText("Max measurement: Off")

    def _build_layer_categories(self, scalar_names: list[str]) -> dict[str, list[str]]:
        """用途 Purpose:
        - 将标量层名按预设类别关键字进行分组。
        - Group scalar layer names using configured category keywords.

        参数 Args:
        - scalar_names: 所有可用标量层名。

        返回 Returns:
        - dict[str, list[str]]: 分类名到层名列表的映射。
        """
        categories: dict[str, list[str]] = {name: [] for name in CATEGORY_PREFIXES}
        for scalar_name in scalar_names:
            lower_name = scalar_name.lower()
            for category_name, prefix in CATEGORY_PREFIXES.items():
                if prefix.lower() in lower_name:
                    categories[category_name].append(scalar_name)

        return categories

    def _initialize_category_controls(self) -> None:
        """用途 Purpose:
        - 根据分类结果初始化下拉框与开关控件状态。
        - Initialize category controls based on available grouped layers.

        行为 Behavior:
        - 无可用分类时自动关闭比较功能并显示说明。
        - Disables compare mode and shows hint when no categories are available.
        """
        available_categories = [
            category_name
            for category_name in CATEGORY_PREFIXES
            if self.layer_categories.get(category_name)
        ]

        self.category_combo.blockSignals(True)
        self.category_combo.clear()
        self.category_combo.addItems(available_categories)
        self.category_combo.blockSignals(False)

        has_category = bool(available_categories)
        self.category_compare_checkbox.setEnabled(has_category)
        self.category_combo.setEnabled(has_category)
        self.export_category_csv_button.setEnabled(has_category)

        if not has_category:
            self.category_compare_checkbox.blockSignals(True)
            self.category_compare_checkbox.setChecked(False)
            self.category_compare_checkbox.blockSignals(False)
            self.category_compare_enabled = False
            self._clear_category_marker()
            self.category_compare_label.setText(
                "Category compare: No Transmission/Absorption/Incident/Reflection layers found."
            )
        else:
            self.category_compare_label.setText("Category compare: Off")

    def export_category_layers_csv(self) -> None:
        """用途 Purpose:
        - 导出当前 Layer category 下所有 layer 的过滤后最大值与坐标到 CSV。
        - Export filtered max value and max position for all layers in current category.
        """
        if self.mesh is None:
            QMessageBox.warning(self, "Export CSV", "Please load a mesh first.")
            return

        category_name = self.category_combo.currentText().strip()
        if not category_name:
            QMessageBox.warning(self, "Export CSV", "Please select a layer category.")
            return

        layer_names = self.layer_categories.get(category_name, [])
        if not layer_names:
            QMessageBox.warning(
                self,
                "Export CSV",
                f"No layers found for category '{category_name}'.",
            )
            return

        centers = self.mesh.cell_centers().points
        rows: list[dict[str, str]] = []

        for layer_name in sorted(layer_names):
            layer_values = self.build_display_values(layer_name)
            finite_mask = np.isfinite(layer_values)

            if layer_values.size == 0 or not np.any(finite_mask):
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
            valid_values = layer_values[finite_mask]
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

        default_name = f"{category_name.replace(' ', '_')}_layer_maxima.csv"
        output_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save category layer maxima CSV",
            str(Path(default_name)),
            "CSV files (*.csv)",
        )
        if not output_path:
            return

        try:
            with open(output_path, "w", encoding="utf-8-sig", newline="") as csv_file:
                writer = csv.DictWriter(
                    csv_file,
                    fieldnames=["category", "layer_name", "filtered_max", "x", "y", "z"],
                )
                writer.writeheader()
                writer.writerows(rows)
        except Exception as exc:
            QMessageBox.critical(self, "Export CSV", f"Failed to write CSV file:\n{exc}")
            return

        QMessageBox.information(
            self,
            "Export CSV",
            f"Exported {len(rows)} layers to:\n{output_path}",
        )

    def update_category_compare(self) -> None:
        """用途 Purpose:
        - 计算当前分类下各层过滤后最大值并选出全局最优。
        - Compare filtered maxima across layers in selected category.

        输出 Output:
        - 更新 `category_compare_label`，并在最佳位置显示黄色 `CAT MAX` 标记。
        - Updates `category_compare_label` and draws yellow `CAT MAX` marker.
        """
        if not self.category_compare_enabled:
            return

        if self.mesh is None:
            self._clear_category_marker()
            self._clear_category_compare_text()
            return

        category_name = self.category_combo.currentText().strip()
        if not category_name:
            self._clear_category_marker()
            self.category_compare_label.setText("Category compare: No category selected.")
            return

        layer_names = self.layer_categories.get(category_name, [])
        if not layer_names:
            self._clear_category_marker()
            self.category_compare_label.setText(
                f"Category compare: No layers found for {category_name}."
            )
            return

        centers = self.mesh.cell_centers().points
        best_layer = ""
        best_value = -np.inf
        best_cell_id = -1

        for layer_name in layer_names:
            layer_values = self.build_display_values(layer_name)
            if layer_values.size == 0:
                continue

            finite_mask = np.isfinite(layer_values)
            if not np.any(finite_mask):
                continue

            valid_indices = np.where(finite_mask)[0]
            valid_values = layer_values[finite_mask]
            local_max_index = int(np.argmax(valid_values))
            local_value = float(valid_values[local_max_index])
            local_cell_id = int(valid_indices[local_max_index])

            if local_value > best_value:
                # 记录当前分类中的最优层与位置。
                # Keep the best layer and location within current category.
                best_value = local_value
                best_layer = layer_name
                best_cell_id = local_cell_id

        if best_cell_id < 0 or not np.isfinite(best_value):
            self._clear_category_marker()
            self.category_compare_label.setText(
                f"Category compare: No finite values for {category_name} after filtering."
            )
            return

        if best_cell_id >= len(centers):
            self._clear_category_marker()
            self.category_compare_label.setText("Category compare: Failed to resolve max location.")
            return

        best_position = centers[best_cell_id]
        self._clear_category_marker()

        marker_points = np.array([best_position], dtype=np.float32)
        self.category_marker_actor = self.plotter.add_points(
            marker_points,
            color="yellow",
            point_size=13,
            render_points_as_spheres=True,
        )

        self.category_label_actor = self.plotter.add_point_labels(
            marker_points,
            ["CAT MAX"],
            font_size=11,
            show_points=False,
            always_visible=True,
        )

        self.category_compare_label.setText(
            f"Category compare: On\n"
            f"Category: {category_name}\n"
            f"Layer: {best_layer}\n"
            f"Filtered max: {best_value:.6g}\n"
            f"Cell: {best_cell_id}\n"
            f"Position: ({float(best_position[0]):.4f}, {float(best_position[1]):.4f}, {float(best_position[2]):.4f})"
        )

        self.plotter.render()

    def _clear_category_marker(self) -> None:
        """用途 Purpose:
        - 删除分类比较功能产生的标记 actor。
        - Remove actors created by category comparison marker.
        """
        if self.category_marker_actor is not None:
            try:
                self.plotter.remove_actor(self.category_marker_actor)
            except Exception:
                pass
            self.category_marker_actor = None

        if self.category_label_actor is not None:
            try:
                self.plotter.remove_actor(self.category_label_actor)
            except Exception:
                pass
            self.category_label_actor = None

    def _clear_category_compare_text(self) -> None:
        """用途 Purpose:
        - 根据分类比较开关状态恢复默认提示文本。
        - Reset category-compare label text based on feature state.
        """
        if self.category_compare_enabled:
            self.category_compare_label.setText(
                "Category compare: Select a category to compare filtered layer maxima."
            )
        else:
            self.category_compare_label.setText("Category compare: Off")


def main() -> int:
    """程序入口 / Application entry point."""
    app = QApplication(sys.argv)
    initial_path = sys.argv[1] if len(sys.argv) > 1 else None

    window = VtpViewerWindow(initial_path=initial_path)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
