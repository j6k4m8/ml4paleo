import logging
from typing import Tuple
import tqdm
from intern.utils.parallel import block_compute
from ..volume_providers import VolumeProvider
import pathlib
from zmesh import Mesher
from stl import mesh as stl_mesh
import stl
import numpy as np

logging.basicConfig(level=logging.DEBUG)


class ChunkedMesher:
    def __init__(
        self,
        volume_provider: VolumeProvider,
        mesh_path: pathlib.Path,
        chunk_size: Tuple[int, int, int],
        mip: int = 1,
    ):
        self.volume_provider = volume_provider
        self.mesh_path = mesh_path
        self.mesh_path.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self._ids = None
        self.mip = mip

    def _add_id(self, obj_id: int):
        if self._ids is None:
            self._ids = set()
        self._ids.add(obj_id)

    def mesh_all(self, progress: bool = True):
        chunks_to_mesh = block_compute(
            0,
            self.volume_provider.shape[0],
            0,
            self.volume_provider.shape[1],
            0,
            self.volume_provider.shape[2],
            block_size=self.chunk_size,
        )

        # Now mesh each chunk.
        _prog = tqdm.tqdm if progress else lambda x: x
        for xs, ys, zs in _prog(chunks_to_mesh):
            self.mesh_chunk(xs, ys, zs)

        # Combine meshes
        if self._ids is None:
            return
        for obj_id in self._ids:
            self.combine_meshes(obj_id)

    def mesh_chunk(self, xs, ys, zs):
        labels = self.volume_provider[xs[0] : xs[1], ys[0] : ys[1], zs[0] : zs[1]]
        m = 2**self.mip
        mesher = Mesher((m, m, m))
        mesher.mesh(
            labels[
                ::m,
                ::m,
                ::m,
            ],
            close=False,
        )
        meshes = {}
        for obj_id in mesher.ids():
            self._add_id(obj_id)
            meshes[obj_id] = mesher.get_mesh(
                obj_id,
                normals=False,
                simplification_factor=100,
                max_simplification_error=20,
            )
            mesher.erase(obj_id)
        mesher.clear()
        for obj_id, mesh in meshes.items():
            # with open(
            #     str(self.mesh_path / f"_{obj_id}_{xs[0]}_{ys[0]}_{zs[0]}.obj"), "wb"
            # ) as f:
            #     f.write(mesh.to_obj())

            m = stl_mesh.Mesh(np.zeros(mesh.faces.shape[0], dtype=stl_mesh.Mesh.dtype))
            for i, f in enumerate(mesh.faces):
                for j in range(3):
                    m.vectors[i][j] = mesh.vertices[f[j], :]
            # Offset the mesh to the correct position.
            m.x += zs[0]
            m.y += ys[0]
            m.z += xs[0]
            m.save(
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
            f.write(f"f {3*i+1} {3*i+2} {3*i+3}\n")
