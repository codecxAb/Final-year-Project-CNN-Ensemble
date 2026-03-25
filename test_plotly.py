import plotly.graph_objects as go
import numpy as np
from PIL import Image

img = Image.open("/Users/anurag/College/lungCanerProject/lungcare_triage/frontend-streamlit/assets/lung_texture.png").convert("RGB")
img = img.resize((100, 100))
img_q = img.quantize(colors=256)
palette = img_q.getpalette()

colorscale = []
for i in range(256):
    r, g, b = palette[i*3 : i*3+3]
    colorscale.append([i / 255.0, f"rgb({r},{g},{b})"])

res = 100
u = np.linspace(0, 2*np.pi, res)
v = np.linspace(0, np.pi, res)
U, V = np.meshgrid(u, v)
X = np.sin(V) * np.cos(U)
Y = np.sin(V) * np.sin(U)
Z = np.cos(V)

surfacecolor = np.array(img_q) / 255.0

fig = go.Figure(go.Surface(
    x=X, y=Y, z=Z,
    surfacecolor=surfacecolor,
    colorscale=colorscale,
    showscale=False
))
fig.write_image("test_surface.png")
print("Saved test_surface.png")
