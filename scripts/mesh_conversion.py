from pathlib import Path

import trimesh

from argus import ROOT

if __name__ == "__main__":
    mesh_path = Path(ROOT) / "LeapProject/Assets/urdf/meshes"

    for mesh_file in mesh_path.glob("*.stl"):
        mesh = trimesh.load_mesh(mesh_file)
        mesh.export(mesh_path / f"{mesh_file.stem}.obj", file_type="obj")

        # Add scene with mesh
        scene = trimesh.scene.scene.Scene()
        scene.add_geometry(mesh)

        # Plot mesh origin.
        origin = trimesh.creation.axis(origin_size=0.001)
        scene.add_geometry(origin)

        scene.show()
