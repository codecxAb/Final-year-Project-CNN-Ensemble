import trimesh
import numpy as np

try:
    # trimesh.load returns a Scene if there are multiple meshes, or a Trimesh if singular
    # force='mesh' concatenates them into a single Trimesh
    mesh = trimesh.load('assets/lungs.glb', force='mesh')
    print("Mesh loaded successfully!")
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        print("Vertex colors shape:", mesh.visual.vertex_colors.shape)
    else:
        print("No vertex colors found")
        
except Exception as e:
    print("Error:", repr(e))
