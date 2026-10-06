"""
The v1 web app's mesher: meshes a whole volume chunk by chunk on one machine.
"""

import json
import pathlib

import numpy as np
import skimage.measure
import stl
import tqdm
from stl import mesh as stl_mesh
from zmesh import Mesher

from ..blocks import block_ranges
from ..volume_providers import VolumeProvider
from . import MESH_INFO_FILENAME


class ChunkedMesher:
    def __init__(
        self,
        volume_provider: VolumeProvider,
        mesh_path: pathlib.Path,
        chunk_size: tuple[int, int, int],
        downsample_factor: int = 1,
        voxel_size_xyz_mm: tuple[float, float, float] | None = None,
    ):
        """
        Arguments:
            voxel_size_xyz_mm: The physical voxel size. If given (or recorded
                by the volume provider), mesh vertices are written in mm;
                otherwise they are in voxel units.

        """
        self.volume_provider = volume_provider
        self.mesh_path = mesh_path
        self.mesh_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self._ids = None
        if downsample_factor < 1:
            raise ValueError("downsample_factor must be at least 1")
        self.downsample_factor = int(downsample_factor)
        if voxel_size_xyz_mm is None:
            voxel_size_xyz_mm = getattr(volume_provider, "voxel_size_xyz_mm", None)
        self.voxel_size_xyz_mm = voxel_size_xyz_mm
        self._vertex_scale = np.array(
            voxel_size_xyz_mm if voxel_size_xyz_mm is not None else (1, 1, 1),
            dtype=np.float64,
        )

    def _add_id(self, obj_id: int):
        if self._ids is None:
            self._ids = set()
        self._ids.add(obj_id)

    def mesh_all(self, progress: bool = True):
        chunks_to_mesh = block_ranges(self.volume_provider.shape, self.chunk_size)

        # Remove per-chunk meshes from any earlier run so they are not combined
        # with this one.
        for old_chunk_mesh in self.mesh_path.glob("_*.stl"):
            old_chunk_mesh.unlink()

        # Now mesh each chunk.
        _prog = tqdm.tqdm if progress else lambda x: x
        for xs, ys, zs in _prog(chunks_to_mesh):
            self.mesh_chunk(xs, ys, zs)

        # Combine meshes
        for obj_id in self._ids or []:
            self.combine_meshes(obj_id)
        self.write_mesh_info()

    def write_mesh_info(self):
        """
        Record the axis order and units of the mesh files next to them.
        """
        mesh_info = {
            "axis_order": "xyz",
            "units": "mm" if self.voxel_size_xyz_mm is not None else "voxels",
            "voxel_size_xyz": [float(v) for v in self._vertex_scale],
        }
        with open(self.mesh_path / MESH_INFO_FILENAME, "w") as f:
            json.dump(mesh_info, f, indent=2)

    def mesh_chunk(self, xs, ys, zs):
        labels = self.volume_provider[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]]
        m = self.downsample_factor
        if m > 1:
            mesh_labels = skimage.measure.block_reduce(labels, (m, m, m), np.max)
        else:
            mesh_labels = labels

        # Don't mesh empty chunks:
        if np.max(mesh_labels) == 0:
            return

        mesher = Mesher((m, m, m))
        # labels[
        #     ::m,
        #     ::m,
        #     ::m,
        # ],
        mesher.mesh(
            mesh_labels,
            close=False,
        )
        meshes = {}
        for obj_id in mesher.ids():
            self._add_id(obj_id)
            meshes[obj_id] = mesher.get_mesh(
                obj_id,
                normals=False,
                simplification_factor=50,
                max_simplification_error=10,
            )
            mesher.erase(obj_id)
        mesher.clear()
        for obj_id, mesh in meshes.items():
            # zmesh reads the (X, Y, Z) array as (Z, Y, X), so its vertices come
            # out in (z, y, x) order. Reverse them to (x, y, z). Swapping axes
            # mirrors the mesh, so also reverse the triangle winding to keep the
            # normals pointing outward. Then offset the chunk into place and
            # scale to physical units.
            vertices = mesh.vertices[:, ::-1] + np.array([xs[0], ys[0], zs[0]])
            vertices = vertices * self._vertex_scale
            faces = mesh.faces[:, ::-1]

            chunk_mesh = stl_mesh.Mesh(
                np.zeros(faces.shape[0], dtype=stl_mesh.Mesh.dtype)
            )
            chunk_mesh.vectors[:] = vertices[faces]
            chunk_mesh.save(
                str(self.mesh_path / f"_{obj_id}_{xs[0]}_{ys[0]}_{zs[0]}.stl"),
                mode=stl.Mode.ASCII,
            )

    def combine_meshes(self, object_id: int):
        obj_meshes = list(self.mesh_path.glob(f"_{object_id}_*.stl"))
        if len(obj_meshes) == 0:
            return
        combined_mesh = stl_mesh.Mesh(np.zeros(0, dtype=stl_mesh.Mesh.dtype))
        combined_mesh = stl_mesh.Mesh(
            np.concatenate(
                [
                    stl_mesh.Mesh.from_file(str(obj_mesh), mode=stl.Mode.ASCII).data
                    for obj_mesh in obj_meshes
                ]
            )
        )
        combined_mesh.save(
            str(self.mesh_path / f"{object_id}.combined.stl"), mode=stl.Mode.BINARY
        )
        write_obj(combined_mesh, str(self.mesh_path / f"{object_id}.combined.obj"))


def write_obj(mesh, filename):
    with open(filename, "w") as f:
        for v in mesh.vectors:
            for p in v:
                f.write(f"v {p[0]} {p[1]} {p[2]}\n")
        for i in range(len(mesh.vectors)):
            f.write(f"f {3 * i + 1} {3 * i + 2} {3 * i + 3}\n")
