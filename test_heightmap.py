import numpy as np
from PIL import Image
import plotly.graph_objects as go

# Load image
img = Image.open('frontend-streamlit/assets/lung_texture.png').convert('RGB')
img = img.resize((150, 150))
arr = np.array(img, dtype=float) / 255.0

# Calculate luminance
is_bg = (arr[:,:,0] > 0.95) & (arr[:,:,1] > 0.95) & (arr[:,:,2] > 0.95)
gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]

# Heightmap
Z = 1.0 - gray
Z[is_bg] = np.nan
# no smoothing needed for simple test

fig = go.Figure(data=[go.Surface(
    z=Z,
    colorscale='Reds',
    showscale=False
)])
fig.write_html('test_heightmap.html')
print("Done")
