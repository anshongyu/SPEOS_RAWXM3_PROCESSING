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
"""将 Speos RAWXM3（HDF5）结果转换为 VTP（VTK PolyData）。

主要功能：
1. 读取每个 face 的几何信息（顶点、法向、三角面）
2. 读取每个 face 的面片物理量（facets_data）
3. 合并所有 face 到一个完整网格
4. 将元数据写入 cell_data / field_data
5. 导出为可在 ParaView / PyVista 中使用的 .vtp 文件
"""

from __future__ import annotations

import os
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog
from typing import Any

import h5py
import numpy as np
import pyvista as pv
from tqdm import tqdm


# =============================================================================
# GUI: 选择 Speos RAWXM3 文件
# =============================================================================

def select_rawxm3_file() -> str | None:
    """通过文件对话框选择 RAWXM3 文件路径。"""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select Speos RAWXM3 file",
        filetypes=[
            ("Speos RAWXM3 files", "*.rawxm3"),
            ("HDF5 files", "*.h5 *.hdf5"),
            ("All files", "*.*"),
        ],
    )

    root.destroy()

    if not file_path:
        print("No file selected. Exit.")
        return None

    return file_path


def normalize_text(value: Any) -> str:
    """将 HDF5 属性中的 bytes / numpy.bytes_ 统一转换为 str。"""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="replace")
    return str(value)


def read_triplets(dataset: h5py.Dataset, dataset_name: str) -> np.ndarray:
    """读取并校验形如 [x,y,z,x,y,z,...] 的一维数组，重排为 (N, 3)。"""
    values = np.asarray(dataset)
    if values.size % 3 != 0:
        raise ValueError(
            f"Dataset '{dataset_name}' size {values.size} is not divisible by 3."
        )
    return values.reshape(-1, 3)


def get_faces_group(hdf5_file: h5py.File) -> tuple[h5py.Group, list[str]]:
    """从 RAWXM3 根节点中提取 faces 组及其子键列表。"""
    root_keys = list(hdf5_file.keys())
    if not root_keys:
        raise ValueError("RAWXM3 file is empty.")

    faces_group = hdf5_file[root_keys[0]]
    face_keys = list(faces_group.keys())
    if not face_keys:
        raise ValueError("No face entries found in RAWXM3 file.")

    return faces_group, face_keys


def prompt_data_index(available_names: list[str]) -> int:
    """让用户选择要导出的物理量索引（0 表示全部）。"""
    while True:
        raw_value = input("\nWhich data to process? (0 = all): ").strip()
        try:
            selected_index = int(raw_value) - 1
        except ValueError:
            print("Please enter an integer.")
            continue

        if -1 <= selected_index < len(available_names):
            return selected_index

        print(f"Please enter a value between 0 and {len(available_names)}.")


# =============================================================================
# 基础数据结构
# =============================================================================

@dataclass(slots=True)
class Normal:
    """顶点法向量。"""
    x: float
    y: float
    z: float


@dataclass(slots=True)
class Vertex:
    """三维顶点坐标。"""
    x: float
    y: float
    z: float


@dataclass(slots=True)
class Facet:
    """三角面片（存储 3 个顶点索引）。"""
    index_1: int
    index_2: int
    index_3: int


# =============================================================================
# 面片物理量
# =============================================================================

@dataclass
class FaceFacetsData:
    """单个物理量在一个 face 上的所有 cell 值。"""
    name: str
    data: list[float] = field(default_factory=list)
    data_size: int = 0

    @classmethod
    def from_hdf5(cls, hdf5_data: h5py.Group, facet_count: int) -> "FaceFacetsData":
        """从 HDF5 facets_data 节点构建 FaceFacetsData。

        - data_size == 1: 使用单值扩展到该 face 的所有面片
        - data_size > 1: 直接读取逐面片数组
        """
        name = normalize_text(hdf5_data.attrs["name"])
        data_size = int(hdf5_data.attrs["data_size"])

        if data_size == 1:
            value = float(hdf5_data.attrs["data"])
            data = [value] * facet_count
        else:
            data = np.asarray(hdf5_data["data"], dtype=np.float32).ravel().tolist()
            if len(data) != facet_count:
                raise ValueError(
                    f"Facet data size mismatch for '{name}': "
                    f"expected {facet_count}, got {len(data)}."
                )

        return cls(name=name, data=data, data_size=data_size)


# =============================================================================
# 单个 Face
# =============================================================================

@dataclass
class FaceDescription:
    """RAWXM3 中一个 face 的完整描述（几何 + 元数据 + 物理量）。"""
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
        data_index: int = -1,
    ) -> "FaceDescription":
        """从一个 HDF5 face 节点读取数据并构造对象。

        vertex_offset 用于多 face 合并时修正三角索引。
        data_index = -1 表示加载全部物理量。
        """
        # 读取几何基础数组并重排为 N×3
        normals_raw = read_triplets(hdf5_face["normals"], "normals")
        vertices_raw = read_triplets(hdf5_face["vertices"], "vertices")
        facets_raw = read_triplets(hdf5_face["facets"], "facets")

        # 构建法向、顶点、三角索引对象
        normals = [Normal(float(x), float(y), float(z)) for x, y, z in normals_raw]
        vertices = [Vertex(float(x), float(y), float(z)) for x, y, z in vertices_raw]
        facets = [
            Facet(
                int(i1) + vertex_offset,
                int(i2) + vertex_offset,
                int(i3) + vertex_offset,
            )
            for i1, i2, i3 in facets_raw
        ]

        # metadata 读取：保证 keys 与 values 长度一致
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
        if not data_keys:
            raise ValueError("No facet data found in face.")

        if data_index != -1 and not 0 <= data_index < len(data_keys):
            raise IndexError(
                f"Facet data index {data_index} is out of range for {len(data_keys)} items."
            )

        # 读取全部或指定物理量
        selected_keys = data_keys if data_index == -1 else [data_keys[data_index]]
        facets_data = [
            FaceFacetsData.from_hdf5(facets_data_group[key], len(facets))
            for key in selected_keys
        ]

        return cls(
            facets=facets,
            normals=normals,
            vertices=vertices,
            facets_data=facets_data,
            metadata=metadata,
        )


# =============================================================================
# 合并所有 Face → VTK
# =============================================================================

@dataclass
class CompleteMapDescription:
    """所有 face 合并后的完整网格描述。"""
    vertices: list[Vertex] = field(default_factory=list)
    normals: list[Normal] = field(default_factory=list)
    facets: list[Facet] = field(default_factory=list)
    facets_data: dict[str, list[float]] = field(default_factory=dict)
    metadata_summary: dict[str, set[str]] = field(default_factory=dict)
    metadata_cell_data: dict[str, list[str]] = field(default_factory=dict)

    def add_face(self, face: FaceDescription) -> None:
        """将一个 face 合并到总网格中。"""
        previous_cell_count = len(self.facets)
        current_cell_count = len(face.facets)

        # 合并几何
        self.vertices.extend(face.vertices)
        self.normals.extend(face.normals)
        self.facets.extend(face.facets)

        existing_metadata_keys = set(self.metadata_cell_data.keys())
        current_metadata_keys = set(face.metadata.keys())

        # 对缺失 metadata 键补空字符串，确保所有 cell_data 长度一致
        for missing_key in existing_metadata_keys - current_metadata_keys:
            self.metadata_cell_data[missing_key].extend([""] * current_cell_count)

        # 写入本 face 的 metadata 到 cell_data，并维护 summary
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

        if set(self.facets_data.keys()) != set(face_data_by_name.keys()):
            raise ValueError(
                "Facet data schema mismatch between faces: "
                f"expected {sorted(self.facets_data.keys())}, "
                f"got {sorted(face_data_by_name.keys())}."
            )

        # 逐物理量追加 cell 值
        for name, values in face_data_by_name.items():
            self.facets_data[name].extend(values)

    def to_vtp(self, filename: str) -> None:
        """导出为 VTP 文件，并写入几何、法向、cell_data、field_data。"""
        if not self.vertices or not self.facets:
            raise ValueError("No geometry available to export.")

        points = np.array(
            [[vertex.x, vertex.y, vertex.z] for vertex in self.vertices],
            dtype=np.float32,
        )

        faces = np.empty((len(self.facets), 4), dtype=np.int64)
        faces[:, 0] = 3
        faces[:, 1] = [facet.index_1 for facet in self.facets]
        faces[:, 2] = [facet.index_2 for facet in self.facets]
        faces[:, 3] = [facet.index_3 for facet in self.facets]

        mesh = pv.PolyData(points, faces.ravel())

        # 法向写入 point_data
        if self.normals:
            if len(self.normals) != mesh.n_points:
                raise ValueError(
                    f"Normal count mismatch: expected {mesh.n_points}, got {len(self.normals)}."
                )

            normals = np.array(
                [[normal.x, normal.y, normal.z] for normal in self.normals],
                dtype=np.float32,
            )
            mesh.point_data["Normals"] = normals

        # 物理量写入 cell_data
        for name, values in self.facets_data.items():
            data = np.asarray(values, dtype=np.float32)
            if data.size != mesh.n_cells:
                raise ValueError(
                    f"Cell data size mismatch for '{name}': expected {mesh.n_cells}, got {data.size}."
                )
            mesh.cell_data[name] = data

        # metadata 逐 cell 写入，便于后续筛选
        for key, values in self.metadata_cell_data.items():
            if len(values) != mesh.n_cells:
                raise ValueError(
                    f"Metadata cell data size mismatch for '{key}': "
                    f"expected {mesh.n_cells}, got {len(values)}."
                )
            mesh.cell_data[f"meta_{key}"] = np.asarray(values)

        # metadata 摘要写入 field_data
        for key, values in self.metadata_summary.items():
            mesh.field_data[f"meta_{key}_summary"] = np.asarray(sorted(values))

        mesh.save(filename, binary=True)

        print("\nVTK export finished")
        print("File:", filename)
        print("Vertices:", mesh.n_points)
        print("Faces:", mesh.n_cells)
        print("CellData:", list(mesh.cell_data.keys()))
        print("FieldData:", list(mesh.field_data.keys()))


def main() -> int:
    """脚本入口：读取 RAWXM3、合并数据并导出 VTP。"""
    # 1) 选择输入文件
    raw_xm3_file = select_rawxm3_file()
    if raw_xm3_file is None:
        return 0

    print("Analyzing:", os.path.basename(raw_xm3_file))

    # 2) 读取 HDF5 并解析 faces
    with h5py.File(raw_xm3_file, "r") as hdf5_file:
        faces_group, face_keys = get_faces_group(hdf5_file)

        # 使用第一个 face 枚举可选物理量
        first_face = FaceDescription.from_hdf5(faces_group[face_keys[0]])
        available_names = [item.name for item in first_face.facets_data]

        print("\nAvailable facet data:")
        for index, name in enumerate(available_names, start=1):
            print(f"{index}: {name}")

        # 3) 用户选择物理量并开始全量合并
        data_index = prompt_data_index(available_names)

        complete_map = CompleteMapDescription()
        vertex_offset = 0

        for key in tqdm(
            face_keys,
            desc="Processing Faces",
            unit="face",
            ncols=90,
        ):
            face = FaceDescription.from_hdf5(
                faces_group[key],
                vertex_offset=vertex_offset,
                data_index=data_index,
            )

            vertex_offset += len(face.vertices)
            complete_map.add_face(face)

            # 4) 导出 VTP
    output_vtp = os.path.splitext(raw_xm3_file)[0] + "_with_metadata.vtp"
    print("\nWriting VTK file...")
    complete_map.to_vtp(output_vtp)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Error: {exc}")
        raise SystemExit(1)