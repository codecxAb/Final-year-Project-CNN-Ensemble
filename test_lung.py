import numpy as np
import plotly.graph_objects as go

def build_lung(side, cx, cy, cz, rx, ry, rz, res=80):
    u = np.linspace(0, 2*np.pi, res)
    v = np.linspace(0, np.pi, res)
    U, V = np.meshgrid(u, v)

    # Basic blob
    xs = np.sin(V) * np.cos(U)
    ys = np.sin(V) * np.sin(U)
    zs = np.cos(V)

    # Lungs are wide at bottom, narrow at top.
    # sin(V) naturally goes 0 -> 1 -> 0.
    # We want to widen the bottom.
    # zs goes +1 to -1.
    widen = 1.2 - 0.4 * zs 
    xs = xs * widen
    ys = ys * widen
    
    # Diaphragm concavity: push Z up when near the bottom core
    # The bottom is around zs = -1. Core is xs=0, ys=0
    # distance from center
    r_xy = np.sqrt(xs**2 + ys**2)
    # push up Z at the bottom where r_xy is small
    diaphragm = np.exp(-(r_xy**2)/0.5) * np.clip(-zs, 0, 1) * 0.4
    zs = zs + diaphragm
    
    # Medial flattening (inside face)
    if side == "right":
        medial = np.clip(-xs, 0, 1)  # right lung inside is -X
    else:
        medial = np.clip(xs, 0, 1)   # left lung inside is +X
    xs = xs + medial * 0.5 * (1 if side=="right" else -1)
    
    # Cardiac notch
    if side == "left":
        # Notch is roughly front-medial
        front_medial = np.clip(ys, 0, 1) * np.clip(xs, 0, 1) * np.clip(-zs, 0, 1)
        xs = xs - front_medial * 0.8
        ys = ys - front_medial * 0.8
        
    X = cx + rx * xs
    Y = cy + ry * ys
    Z = cz + rz * zs
    return X, Y, Z

Xr, Yr, Zr = build_lung('right', 3, 0, 0, 2.5, 3, 4)
Xl, Yl, Zl = build_lung('left', -3, 0, 0, 2.3, 3, 4)

fig = go.Figure()
fig.add_trace(go.Surface(x=Xr, y=Yr, z=Zr, colorscale='Reds', showscale=False))
fig.add_trace(go.Surface(x=Xl, y=Yl, z=Zl, colorscale='Blues', showscale=False))
fig.update_layout(scene=dict(aspectmode='data'))
fig.write_html('test_lung.html')
print("done")
