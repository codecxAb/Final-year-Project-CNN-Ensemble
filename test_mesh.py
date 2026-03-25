import sys
import os
import plotly.graph_objects as go
import numpy as np

# emulate what load_glb_mesh does:
import trimesh
_path = os.path.join("frontend-streamlit", "assets", "lungs.glb")
print(f"Loading '{_path}'...")
mesh = trimesh.load(_path, force='mesh')
mesh.apply_scale(20.0)
print(f"Loaded: {mesh.vertices.shape[0]} vertices")

# emulate plot
fig = go.Figure()
fig.add_trace(go.Mesh3d(
    x=mesh.vertices[:, 0],
    y=mesh.vertices[:, 1],
    z=mesh.vertices[:, 2],
    i=mesh.faces[:, 0],
    j=mesh.faces[:, 1],
    k=mesh.faces[:, 2],
    opacity=0.15,
    color='rgba(250, 180, 180, 1.0)'
))
print("Added trace successfully.")
