import codecs

with codecs.open('frontend-streamlit/app.py', 'r', 'utf-8') as f:
    text = f.read()

start_sig = "# 3D LUNG VISUALIZATION\n# ═══════════════════════════════════════════════════════════════════════════════"
end_sig = "def render_lung_hud"

start_idx = text.find(start_sig)
end_idx = text.find(end_sig)

if start_idx == -1 or end_idx == -1:
    print("Could not find the markers in the file.")
    exit(1)

new_code = """# 3D LUNG VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

import pathlib as _pathlib
from PIL import Image, ImageFilter
import numpy as np
import plotly.graph_objects as go

@st.cache_data(show_spinner=False)
def create_3d_lung_mesh_from_image(res=120):
    img_path = _pathlib.Path(__file__).parent / "assets" / "lung_texture.png"
    if not img_path.exists():
        return None, None, None, None, None, None
        
    try:
        # Load the image and resize for 3D grid resolution
        orig_img = Image.open(img_path).convert("RGBA")
        
        # Determine background threshold
        arr_orig = np.array(orig_img)
        if arr_orig.shape[2] == 4:
            alpha_mask = arr_orig[:,:,3] > 10
        else:
            # Grayscale luminance mask
            lum = arr_orig[:,:,0]*0.299 + arr_orig[:,:,1]*0.587 + arr_orig[:,:,2]*0.114
            alpha_mask = lum < 240
            
        # Crop to content to maximize resolution usage
        y_indices, x_indices = np.where(alpha_mask)
        if len(y_indices) == 0:
             return None, None, None, None, None, None
        
        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        
        pad = 5
        y_min, y_max = max(0, y_min-pad), min(arr_orig.shape[0], y_max+pad)
        x_min, x_max = max(0, x_min-pad), min(arr_orig.shape[1], x_max+pad)
        
        cropped_img = orig_img.crop((x_min, y_min, x_max, y_max))
        img = cropped_img.resize((res, res), Image.LANCZOS)
        
        # 1) Make the Heightmap (Thickness)
        arr = np.array(img)
        if arr.shape[2] == 4:
            mask = arr[:,:,3] > 10
        else:
            lum = arr[:,:,0]*0.299 + arr[:,:,1]*0.587 + arr[:,:,2]*0.114
            mask = lum < 240
            
        mask_img = Image.fromarray((mask * 255).astype(np.uint8), mode='L')
        # Blur the mask to puff it up into a bell-curve 3D thickness
        blurred = mask_img.filter(ImageFilter.GaussianBlur(15))
        thickness = np.array(blurred, dtype=float) / 255.0
        
        # Exaggerate thickness in the middle, curving out to the edges
        thickness = np.power(thickness, 0.7) * 4.0  # Max Z thickness = 4 units
        
        # 2) Texturing (Quantize to 256 exact colors)
        img_rgb = img.convert("RGB")
        img_q = img_rgb.quantize(colors=256)
        palette = img_q.getpalette()
        
        colorscale = []
        for i in range(256):
            r, g, b = palette[i*3:i*3+3]
            v = i / 255.0
            colorscale.append([v, f"rgb({r},{g},{b})"])
            
        surfacecolor = np.array(img_q) / 255.0
        
        # 3) Build 3D coordinates (X, Y mapped to image bounds)
        # We will use anatomical ratios. The lungs are ~ 25cm high, 20cm wide
        x_lin = np.linspace(-10, 10, res)  # width
        y_lin = np.linspace(15, -10, res)  # height (flipped so Y goes UP)
        X, Y = np.meshgrid(x_lin, y_lin)
        
        # Return arrays to build front and back
        return X, Y, thickness, surfacecolor, colorscale, mask
    except Exception as e:
        print(f"Error building 3D image mesh: {e}")
        return None, None, None, None, None, None

def render_3d_lung(x_coord=256, tumor_mm=5.0, show_previous=False, prev_size=None):
    fig = go.Figure()

    res = 120
    X, Y, Z_thick, sc_val, cscale, mask = create_3d_lung_mesh_from_image(res=res)
    has_texture = X is not None
    
    if has_texture:
        # Prevent plotting zero-thickness background as flat sheets
        # Using np.nan tears the surface mesh leaving the background entirely transparent
        z_front = Z_thick.copy()
        z_back = -Z_thick.copy()
        z_front[~mask] = np.nan
        z_back[~mask] = np.nan
        
        # Very low opacity to clearly see the nodal attachment
        lung_opacity = 0.25
        
        kwargs_common = dict(
            colorscale=cscale,
            surfacecolor=sc_val,
            showscale=False,
            opacity=lung_opacity,
            hoverinfo='skip',
            showlegend=False,
            contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            lighting=dict(ambient=0.6, diffuse=0.4, specular=0.1)
        )
        
        # Front face
        fig.add_trace(go.Surface(
            x=X, y=Y, z=z_front,
            name='Lungs (Front)',
            **kwargs_common
        ))
        
        # Back face
        fig.add_trace(go.Surface(
            x=X, y=Y, z=z_back,
            name='Lungs (Back)',
            **kwargs_common
        ))

    # ── NODULE (TUMOR MASS) attached ──
    # Map raw 0-512 input into our new X, Y anatomic coordinate scaling
    norm_x = (x_coord / 512.0)
    nodule_x = -10.0 + (norm_x * 20.0)  # matching X linspace (-10 to +10)
    
    # Estimate Y dynamically (15 to -10 -> roughly 0 to 12 anatomy)
    nodule_y = 5.0 - (tumor_mm / 30.0)*8.0
    
    # Tumor sits on the anterior surface (approx attaching to the 'thickness' z value)
    nodule_z = 3.0 

    nodule_r = max(0.4, min(tumor_mm / 9.0, 2.0))

    nU, nV = np.meshgrid(np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 20))
    rng = np.random.default_rng(42)  
    noise = 0.05 * np.sin(10*nU) * np.sin(8*nV) + 0.02 * rng.standard_normal(nU.shape)
    nr = nodule_r * (1 + noise)

    nX = nr * np.sin(nV) * np.cos(nU) + nodule_x
    nY = nr * np.sin(nV) * np.sin(nU) + nodule_y
    nZ = nr * np.cos(nV) + nodule_z

    if tumor_mm > 8:
        n_cscale = [[0.0, 'rgba(120,18,18,1.0)'], [1.0, 'rgba(230,40,40,1.0)']]
    elif tumor_mm > 4:
        n_cscale = [[0.0, 'rgba(140,80,15,1.0)'], [1.0, 'rgba(240,150,40,1.0)']]
    else:
        n_cscale = [[0.0, 'rgba(20,100,40,1.0)'],  [1.0,'rgba(60,200,90,1.0)']]

    lung_label = "Left Lung" if nodule_x < 0 else "Right Lung"
    
    fig.add_trace(go.Surface(
        x=nX, y=nY, z=nZ,
        surfacecolor=np.ones_like(nr),
        colorscale=n_cscale,
        showscale=False,
        opacity=1.0,
        name=f'Nodule ({tumor_mm:.1f}mm)',
        hovertemplate=(
            f"<b>🔴 Malignant Mass</b><br>"
            f"Diameter: <b>{tumor_mm:.1f} mm</b><br>"
            f"Location: <b>Attached Structure</b><br>"
            f"X: {nodule_x:.1f} cm | Y: {nodule_y:.1f} cm"
            f"<extra></extra>"
        ),
    ))

    # Blood vessel attaching the nodule (showing the attack vector!)
    # Represents vasculature feeding the node and structural connection to lung tissue
    v_z_base = 0  # center-line of the lung depth
    v_idx = np.linspace(0, 1, 10)
    vx = nodule_x + 0.1*np.sin(v_idx*10)
    vy = nodule_y + 0.1*np.cos(v_idx*10)
    vz = np.linspace(nodule_z - nodule_r, v_z_base, 10)
    
    fig.add_trace(go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode='lines',
        line=dict(color='rgba(239, 68, 68, 0.85)', width=8),
        hoverinfo='skip', showlegend=False,
    ))

    # ── LAYOUT ──
    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-12, 12]),
            yaxis=dict(visible=False, range=[-12, 18]),
            zaxis=dict(visible=False, range=[-8, 8]),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=0.0, y=-0.5, z=2.0),
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
            ),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        height=550,
        showlegend=False,
        uirevision='lung-mesh',
    )
    return fig

"""

with codecs.open('frontend-streamlit/app.py', 'w', 'utf-8') as f:
    f.write(text[:start_idx] + new_code + text[end_idx:])
print("Successfully redesigned the 3D model with connected topology and nodule attachment")
