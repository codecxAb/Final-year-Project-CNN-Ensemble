"""
app.py — LungCare Dashboard (Redesigned)
=================================================
Folder-based triage workflow for radiologists.
Light mode · Minimal · 2-color palette · Animated · Age-friendly.
"""

import requests
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from datetime import datetime

# ─── Config ───────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="LungCare",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Session State Defaults ───────────────────────────────────────────────────
for key, default in {
    "view": "HOME", "selected_folder": None, "selected_patient_id": None,
    "scan_result": None, "previous_risk": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─── Navigation ──────────────────────────────────────────────────────────────

def nav(view, **kwargs):
    st.session_state.view = view
    for k, v in kwargs.items():
        st.session_state[k] = v


# ─── Design System ───────────────────────────────────────────────────────────
# Palette: Primary Blue #2563EB, Slate Gray #64748B, White #FFFFFF
# Risk semantic only: Red #EF4444, Amber #F59E0B, Green #22C55E

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ─── GLOBAL RESET (Data Brutalism Dark Theme) ─── */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: #0A0A0A;
    color: #FAFAFA;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem; max-width: 1200px; }

/* ─── ANIMATIONS ─── */
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity: 0; }
    to   { opacity: 1; }
}
@keyframes slideInRight {
    from { opacity: 0; transform: translateX(20px); }
    to   { opacity: 1; transform: translateX(0); }
}
@keyframes pulse {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.02); }
}

.animate-in { animation: fadeInUp 0.5s ease-out both; }
.animate-in-delay-1 { animation-delay: 0.1s; }
.animate-in-delay-2 { animation-delay: 0.2s; }
.animate-in-delay-3 { animation-delay: 0.3s; }

/* ─── BRAND ─── */
.brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 2rem;
    animation: fadeIn 0.6s ease-out;
}
.brand-icon {
    width: 40px;
    height: 40px;
    background: #0A0A0A;
    border: 2px solid #22D3EE;
    border-radius: 4px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.2rem;
    color: #22D3EE;
    box-shadow: 0 0 10px rgba(34,211,238,0.2);
}
.brand-text {
    font-size: 1.4rem;
    font-weight: 800;
    color: #FAFAFA;
    letter-spacing: -0.5px;
    text-shadow: 0 0 10px rgba(34,211,238,0.1);
}
.brand-sub {
    font-size: 0.75rem;
    color: #A1A1AA;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ─── CARDS ─── */
.card {
    background: #171717;
    border: 1px solid #27272A;
    border-radius: 4px;
    padding: 1.5rem;
    transition: border-color 0.2s;
    position: relative;
    overflow: hidden;
}
.card:hover {
    border-color: #3F3F46;
}

/* ─── FOLDER CARDS ─── */
.folder-card {
    background: #171717;
    border: 1px solid #27272A;
    border-radius: 4px;
    padding: 2rem 1.5rem;
    text-align: center;
    transition: all 0.2s;
    cursor: pointer;
    position: relative;
}
.folder-card:hover {
    transform: translateY(-2px);
    border-color: #22D3EE;
    box-shadow: 0 4px 15px rgba(34,211,238,0.1);
}
.folder-card::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    transition: all 0.3s;
    opacity: 0.5;
}
.folder-critical::after { background: #EF4444; }
.folder-observation::after { background: #F59E0B; }
.folder-clear::after { background: #22C55E; }

.folder-emoji {
    font-size: 2.5rem;
    margin-bottom: 0.6rem;
    text-shadow: 0 0 15px rgba(255,255,255,0.1);
}
.folder-label {
    font-size: 0.75rem;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #A1A1AA;
    margin-bottom: 0.4rem;
}
.folder-count {
    font-size: 3.5rem;
    font-weight: 900;
    line-height: 1;
    margin: 0.3rem 0;
    color: #FAFAFA;
}
.folder-desc {
    font-size: 0.8rem;
    color: #71717A;
    font-weight: 500;
}

/* ─── PATIENT LIST ITEM ─── */
.patient-row {
    background: #171717;
    border: 1px solid #27272A;
    border-radius: 4px;
    padding: 1rem 1.5rem;
    margin-bottom: 0.6rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    transition: all 0.2s ease;
    animation: slideInRight 0.3s ease-out both;
}
.patient-row:hover {
    border-color: #22D3EE;
    transform: translateX(4px);
    background: #0A0A0A;
}
.patient-avatar {
    width: 44px;
    height: 44px;
    border-radius: 4px;
    background: rgba(34,211,238,0.1);
    color: #22D3EE;
    border: 1px solid rgba(34,211,238,0.3);
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 800;
    font-size: 1rem;
    flex-shrink: 0;
}
.patient-details { flex: 1; min-width: 0; }
.patient-name {
    font-weight: 700;
    font-size: 1rem;
    color: #FAFAFA;
    margin-bottom: 2px;
}
.patient-meta {
    font-size: 0.8rem;
    color: #A1A1AA;
    font-family: monospace;
}

/* ─── RISK PILLS ─── */
.pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.25rem 0.75rem;
    border-radius: 2px;
    font-size: 0.75rem;
    font-weight: 800;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    border: 1px solid;
}
.pill-critical {
    background: rgba(239, 68, 68, 0.1);
    color: #EF4444;
    border-color: rgba(239, 68, 68, 0.4);
}
.pill-observation {
    background: rgba(245, 158, 11, 0.1);
    color: #F59E0B;
    border-color: rgba(245, 158, 11, 0.4);
}
.pill-clear {
    background: rgba(34, 197, 94, 0.1);
    color: #22C55E;
    border-color: rgba(34, 197, 94, 0.4);
}

/* ─── SECTION HEADER ─── */
.sh {
    font-size: 1.1rem;
    font-weight: 800;
    color: #FAFAFA;
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 1px;
}
.sh-icon {
    width: 28px;
    height: 28px;
    border-radius: 4px;
    background: rgba(34,211,238,0.1);
    color: #22D3EE;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.85rem;
    border: 1px solid rgba(34,211,238,0.3);
}

/* ─── STAT CHIPS ─── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
    animation: fadeInUp 0.5s ease-out 0.3s both;
}
.stat-chip {
    background: #171717;
    border: 1px solid #27272A;
    border-radius: 4px;
    padding: 1.5rem;
    text-align: center;
    flex: 1;
    transition: border-color 0.2s;
}
.stat-chip:hover { border-color: #3F3F46; }
.stat-num {
    font-size: 2.2rem;
    font-weight: 900;
    color: #FAFAFA;
    line-height: 1;
    font-family: monospace;
}
.stat-label {
    font-size: 0.75rem;
    color: #71717A;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 700;
    margin-top: 0.5rem;
}

/* ─── DETAIL HEADER ─── */
.detail-header {
    display: flex;
    align-items: center;
    gap: 1.2rem;
    margin-bottom: 2rem;
    animation: fadeIn 0.5s ease-out;
    border-bottom: 1px solid #27272A;
    padding-bottom: 1.5rem;
}
.detail-avatar {
    width: 64px;
    height: 64px;
    border-radius: 4px;
    background: #171717;
    border: 2px solid #22D3EE;
    color: #22D3EE;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 900;
    font-size: 1.5rem;
    box-shadow: 0 0 15px rgba(34,211,238,0.15);
}
.detail-name {
    font-size: 1.8rem;
    font-weight: 900;
    color: #FAFAFA;
    letter-spacing: -0.5px;
}
.detail-sub {
    font-size: 0.9rem;
    color: #A1A1AA;
    font-family: monospace;
}

/* ─── STEP INDICATORS ─── */
.step-badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 28px;
    height: 28px;
    border-radius: 4px;
    background: #171717;
    border: 1px solid #22D3EE;
    color: #22D3EE;
    font-weight: 900;
    font-size: 0.8rem;
    margin-right: 0.8rem;
    flex-shrink: 0;
}
.step-section {
    background: #171717;
    border: 1px solid #27272A;
    border-radius: 4px;
    padding: 2rem;
    margin-bottom: 1rem;
    animation: fadeInUp 0.4s ease-out both;
}

/* ─── MATCH FEEDBACK ─── */
.match-found {
    background: rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.3);
    border-radius: 2px;
    padding: 0.8rem 1.2rem;
    margin-top: 0.8rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #4ADE80;
    animation: fadeIn 0.3s;
}
.match-new {
    background: rgba(34, 211, 238, 0.1);
    border: 1px solid rgba(34, 211, 238, 0.3);
    border-radius: 2px;
    padding: 0.8rem 1.2rem;
    margin-top: 0.8rem;
    font-size: 0.85rem;
    font-weight: 600;
    color: #67E8F9;
    animation: fadeIn 0.3s;
}

/* ─── METRIC TILES ─── */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0;
    animation: fadeInUp 0.4s ease-out;
}
.metric-tile {
    background: #171717;
    border: 1px solid #27272A;
    border-radius: 4px;
    padding: 1.5rem;
    text-align: center;
}
.metric-tile-value {
    font-size: 1.8rem;
    font-weight: 900;
    line-height: 1;
    margin-bottom: 0.5rem;
    color: #FAFAFA;
    font-family: monospace;
}
.metric-tile-label {
    font-size: 0.75rem;
    color: #71717A;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 800;
}

/* ─── CATEGORY ALERT ─── */
.cat-alert {
    background: rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 2px;
    padding: 1rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #FCD34D;
    font-weight: 600;
    animation: fadeIn 0.4s;
}

/* ─── APPROVAL ─── */
.badge-approved {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1rem;
    border-radius: 2px;
    font-weight: 800;
    font-size: 0.8rem;
    background: rgba(34, 197, 94, 0.1);
    color: #4ADE80;
    border: 1px solid rgba(34,197,94,0.4);
}
.badge-pending {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1rem;
    border-radius: 2px;
    font-weight: 800;
    font-size: 0.8rem;
    background: rgba(245, 158, 11, 0.1);
    color: #FCD34D;
    border: 1px solid rgba(245,158,11,0.4);
}

/* ─── STREAMLIT OVERRIDES ─── */
.stButton > button {
    border-radius: 4px;
    font-weight: 700;
    font-family: 'Inter', sans-serif;
    transition: all 0.2s;
    border: 1px solid #3F3F46;
    background: #171717;
    color: #FAFAFA;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-size: 0.85rem;
    padding: 0.6rem 1.2rem;
}
.stButton > button:hover {
    border-color: #FAFAFA;
    color: #0A0A0A;
    background: #FAFAFA;
}
.stButton > button[data-testid="stBaseButton-primary"] {
    background: rgba(34,211,238,0.1);
    color: #22D3EE;
    border: 1px solid #22D3EE;
}
.stButton > button[data-testid="stBaseButton-primary"]:hover {
    background: #0A0A0A;
    color: #FAFAFA;
    border-color: #FAFAFA;
    box-shadow: 0 0 15px rgba(250,250,250,0.3);
}

div[data-testid="stDataFrame"] {
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid #27272A;
}

.stTextInput > div > div > input {
    border-radius: 2px;
    border: 1px solid #3F3F46;
    background: #171717;
    color: #FAFAFA;
    font-family: 'Inter', sans-serif;
    transition: border-color 0.2s;
}
.stTextInput > div > div > input:focus {
    border-color: #22D3EE;
    box-shadow: 0 0 0 1px #22D3EE;
}

.stTextArea > div > div > textarea {
    border-radius: 2px;
    border: 1px solid #3F3F46;
    background: #171717;
    color: #FAFAFA;
    font-family: 'Inter', sans-serif;
}
.stTextArea > div > div > textarea:focus {
    border-color: #22D3EE;
    box-shadow: 0 0 0 1px #22D3EE;
}

/* Divider */
hr { border-color: #27272A; }

/* Markdown text */
.stMarkdown { color: #A1A1AA; }
h1, h2, h3, h4 { color: #FAFAFA !important; }

/* Footer */
.footer {
    text-align: center;
    padding: 3rem 0 1rem;
    color: #52525B;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1px;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 3D LUNG VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

import pathlib as _pathlib
from PIL import Image, ImageFilter
import numpy as np
import plotly.graph_objects as go

@st.cache_data(show_spinner=False)
def load_glb_mesh(scale=20.0, _force_cache_invalidation=1):
    try:
        import trimesh
        _path = _pathlib.Path(__file__).parent / "assets" / "lungs.glb"
        if not _path.exists():
            return None, None
        mesh = trimesh.load(str(_path), force='mesh')
        mesh.apply_scale(scale)
        return mesh.vertices, mesh.faces
    except Exception as e:
        print(f"Error loading glb: {e}")
        return None, None

def render_3d_lung(x_coord=256, tumor_mm=5.0, show_previous=False, prev_size=None):
    fig = go.Figure()

    vertices, faces = load_glb_mesh(scale=20.0, _force_cache_invalidation=1)
    has_mesh = vertices is not None
    
    if has_mesh:
        # Decrease lung opacity 10-20% to see nodule inside clearly
        lung_opacity = 0.15
        
        fig.add_trace(go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            opacity=lung_opacity,
            color='rgba(250, 180, 180, 1.0)', # light red/pink for healthy lung base
            hoverinfo='skip',
            showlegend=False,
            name='Lungs',
            lighting=dict(ambient=0.6, diffuse=0.6, specular=0.1, roughness=0.8),
        ))

    # ── NODULE (TUMOR MASS) attached ──
    # Map raw 0-512 input into our new X, Y anatomic coordinate scaling
    norm_x = (x_coord / 512.0)
    # The new GLB is roughly bounded between X=-9.3 and X=+9.3
    nodule_x = -9.3 + (norm_x * 18.6)
    
    # Estimate Y dynamically
    nodule_y = 5.0 - (tumor_mm / 30.0)*8.0
    
    # Nodule Z: Tumor sits towards the anterior surface
    nodule_z = 3.0 

    # Render nodule EXACT SIZE (1 plot unit = 1 cm, so tumor_mm / 10 = diameter in cm)
    nodule_r = (tumor_mm / 10.0) / 2.0

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

    # Blood vessel (Cord) attaching the nodule
    v_x_base = nodule_x * 0.5  # Heading towards median plane
    v_y_base = nodule_y + 2.0  # Heading slightly up
    v_z_base = 0.0             # towards center line depth
    
    v_idx = np.linspace(0, 1, 15)
    
    vx = nodule_x + (v_x_base - nodule_x) * v_idx + 0.3 * np.sin(v_idx * np.pi)
    vy = nodule_y + (v_y_base - nodule_y) * v_idx - 0.2 * np.cos(v_idx * np.pi)
    vz = (nodule_z - nodule_r) + (v_z_base - (nodule_z - nodule_r)) * v_idx + 0.5 * np.sin(v_idx * np.pi)
    
    fig.add_trace(go.Scatter3d(
        x=vx, y=vy, z=vz,
        mode='lines',
        line=dict(color='rgba(239, 68, 68, 0.85)', width=8),
        name="Vascular Cord",
        hovertemplate="Vascular Cord Attachment<extra></extra>"
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

def render_lung_hud(x_coord, tumor_mm, risk_level, lung_label):
    norm = x_coord / 512.0
    if norm < 0.5:
        nX = round(-6.0 + norm * 2.0 / 0.5 * 2.2, 1)
    else:
        nX = round(1.5 + (norm - 0.5) * 2.0 / 0.5 * 2.5, 1)
    nY = round(0.2 + (tumor_mm / 30.0) * -0.9, 1)
    nZ = round(0.5 - (tumor_mm / 25.0) * 1.2, 1)

    risk_color = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#22C55E"}.get(risk_level, "#94A3B8")

    st.markdown(f'''
    <div style="
        display:grid;
        grid-template-columns: repeat(6, 1fr);
        gap: 0.75rem;
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 4px;
        padding: 1rem 1.5rem;
        font-family: monospace;
        margin-top: 0.5rem;
    ">
        <div>
            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">X (MED–LAT)</div>
            <div style="font-size:1rem;font-weight:800;color:#22D3EE;">{nX} cm</div>
        </div>
        <div>
            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">Y (SUP–INF)</div>
            <div style="font-size:1rem;font-weight:800;color:#4ADE80;">{nY} cm</div>
        </div>
        <div>
            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">Z (ANT–POST)</div>
            <div style="font-size:1rem;font-weight:800;color:#FBBF24;">{nZ} cm</div>
        </div>
        <div>
            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">Lobe</div>
            <div style="font-size:1rem;font-weight:800;color:#FAFAFA;">{lung_label}</div>
        </div>
        <div>
            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">Diameter</div>
            <div style="font-size:1rem;font-weight:800;color:{risk_color};">{tumor_mm:.1f} mm</div>
        </div>
        <div>
            <div style="font-size:0.65rem;color:#6B7280;text-transform:uppercase;letter-spacing:1px;margin-bottom:3px;">Risk</div>
            <div style="font-size:1rem;font-weight:800;color:{risk_color};">{risk_level.upper()}</div>
        </div>
    </div>
    ''', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# API HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def api_get(path):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def api_post_json(path, data):
    try:
        r = requests.post(f"{API_BASE}{path}", json=data, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        try: detail = e.response.json().get("detail", str(e))
        except Exception: detail = str(e)
        st.error(f"Error: {detail}")
        return None
    except Exception as e:
        st.error(f"Connection error: {e}")
        return None

def api_post_form(path, data, files):
    try:
        r = requests.post(f"{API_BASE}{path}", data=data, files=files, timeout=60)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Error: {e}")
        return None


def pill_html(risk):
    if risk == "High":
        return '<span class="pill pill-critical">Critical</span>'
    elif risk == "Medium":
        return '<span class="pill pill-observation">Under Observation</span>'
    return '<span class="pill pill-clear">Clear</span>'

def risk_to_folder(r):
    return {"High": "critical", "Medium": "under_observation"}.get(r, "clear")

def folder_label(f):
    return {"critical": "Critical", "under_observation": "Under Observation", "clear": "Clear"}.get(f, f)

def initials(name):
    parts = name.split()
    return (parts[0][0] + (parts[1][0] if len(parts) > 1 else "")).upper()


# ═══════════════════════════════════════════════════════════════════════════════
# BRAND HEADER (reusable)
# ═══════════════════════════════════════════════════════════════════════════════

def brand(subtitle="Clinical Triage System"):
    st.markdown(f"""
    <div class="brand">
        <div class="brand-icon">🫁</div>
        <div>
            <div class="brand-text">LungCare</div>
            <div class="brand-sub">{subtitle}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: HOME
# ═══════════════════════════════════════════════════════════════════════════════

def view_home():
    brand()

    # New Scan CTA
    _, center, _ = st.columns([3, 2, 3])
    with center:
        if st.button("🔬  New Scan", use_container_width=True, type="primary"):
            nav("NEW_SCAN")
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    folders = api_get("/api/patients/by-folder")
    if not folders:
        return

    cc = len(folders.get("critical", []))
    oc = len(folders.get("under_observation", []))
    gc = len(folders.get("clear", []))

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="folder-card folder-critical animate-in animate-in-delay-1">
            <div class="folder-emoji">🔴</div>
            <div class="folder-label">Critical</div>
            <div class="folder-count">{cc}</div>
            <div class="folder-desc">Immediate review needed</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", use_container_width=True, key="fc"):
            nav("FOLDER", selected_folder="critical"); st.rerun()

    with c2:
        st.markdown(f"""
        <div class="folder-card folder-observation animate-in animate-in-delay-2">
            <div class="folder-emoji">🟡</div>
            <div class="folder-label">Under Observation</div>
            <div class="folder-count">{oc}</div>
            <div class="folder-desc">Monitoring required</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", use_container_width=True, key="fo"):
            nav("FOLDER", selected_folder="under_observation"); st.rerun()

    with c3:
        st.markdown(f"""
        <div class="folder-card folder-clear animate-in animate-in-delay-3">
            <div class="folder-emoji">🟢</div>
            <div class="folder-label">Clear</div>
            <div class="folder-count">{gc}</div>
            <div class="folder-desc">Low risk / healthy</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open →", use_container_width=True, key="fg"):
            nav("FOLDER", selected_folder="clear"); st.rerun()

    # Stats
    all_p = api_get("/api/patients") or []
    total_scans = sum(p.get("scan_count", 0) for p in all_p)

    st.markdown(f"""
    <div class="stat-row">
        <div class="stat-chip">
            <div class="stat-num">{cc + oc + gc}</div>
            <div class="stat-label">Total Patients</div>
        </div>
        <div class="stat-chip">
            <div class="stat-num">{total_scans}</div>
            <div class="stat-label">Total Scans</div>
        </div>
        <div class="stat-chip">
            <div class="stat-num">{cc}</div>
            <div class="stat-label">Need Review</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: FOLDER
# ═══════════════════════════════════════════════════════════════════════════════

def view_folder():
    folder = st.session_state.selected_folder
    brand(subtitle=f"{folder_label(folder)} Patients")

    if st.button("← Back", key="bf"):
        nav("HOME"); st.rerun()

    folders = api_get("/api/patients/by-folder")
    if not folders:
        return

    patients = folders.get(folder, [])

    if not patients:
        st.info("No patients in this folder yet.")
        return

    st.caption(f"{len(patients)} patient(s)")

    for i, p in enumerate(patients):
        ini = initials(p["name"])
        risk = p.get("latest_risk", "")
        sd = p.get("latest_scan_date", "—")
        sc = p.get("scan_count", 0)

        col1, col2 = st.columns([6, 1])
        with col1:
            st.markdown(f"""
            <div class="patient-row" style="animation-delay:{i*0.08}s;">
                <div class="patient-avatar">{ini}</div>
                <div class="patient-details">
                    <div class="patient-name">{p['name']}</div>
                    <div class="patient-meta">#{p.get('patient_number','—')} · {sc} scan(s) · {sd}</div>
                </div>
                {pill_html(risk)}
            </div>
            """, unsafe_allow_html=True)
        with col2:
            if st.button("View", key=f"v{p['id']}"):
                nav("PATIENT", selected_patient_id=p["id"]); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: PATIENT DETAIL
# ═══════════════════════════════════════════════════════════════════════════════

def view_patient():
    pid = st.session_state.selected_patient_id
    brand(subtitle="Patient Detail")

    if st.button("← Back", key="bp"):
        if st.session_state.selected_folder:
            nav("FOLDER")
        else:
            nav("HOME")
        st.rerun()

    detail = api_get(f"/api/patients/{pid}/full")
    if not detail:
        st.error("Patient not found."); return

    risk = detail.get("latest_risk", "None")
    ini = initials(detail["name"])

    st.markdown(f"""
    <div class="detail-header">
        <div class="detail-avatar">{ini}</div>
        <div>
            <div class="detail-name">{detail['name']}</div>
            <div class="detail-sub">#{detail.get('patient_number','—')} · {detail.get('scan_count',0)} scans {pill_html(risk)}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_viz, col_data = st.columns([1, 1])

    scans = detail.get("scans", [])

    with col_viz:
        st.markdown('<div class="sh"><div class="sh-icon">🫁</div> 3D Visualization</div>', unsafe_allow_html=True)
        if scans:
            latest = scans[-1]
            prev = scans[-2]["tumor_diameter_mm"] if len(scans) > 1 else None
            x_coord = latest.get("x_coordinate", 256)
            tmm_viz = latest["tumor_diameter_mm"]
            risk_viz = latest.get("risk_level", "Low")
            lung_label_viz = "Left Lung" if (x_coord / 512.0) < 0.5 else "Right Lung"

            st.markdown('<div style="background:#0A0D12;border:1px solid #1F2937;border-radius:4px;padding:0.25rem;">', unsafe_allow_html=True)
            fig = render_3d_lung(x_coord, tmm_viz, prev is not None, prev)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            render_lung_hud(x_coord, tmm_viz, risk_viz, lung_label_viz)
        else:
            st.info("No scans yet.")

    with col_data:
        st.markdown('<div class="sh"><div class="sh-icon">📋</div> Scan History</div>', unsafe_allow_html=True)
        if scans:
            df = pd.DataFrame(scans)[["scan_id","date","tumor_diameter_mm","risk_level","status"]]
            df.columns = ["ID", "Date", "Size (mm)", "Risk", "Status"]
            st.dataframe(df, use_container_width=True, hide_index=True)

            if len(scans) > 1:
                st.markdown('<div class="sh"><div class="sh-icon">📈</div> Growth Trend</div>', unsafe_allow_html=True)
                chart = pd.DataFrame({"Scan": range(1, len(scans)+1), "Size (mm)": [s["tumor_diameter_mm"] for s in scans]})
                st.line_chart(chart, x="Scan", y="Size (mm)", use_container_width=True)

            growth = detail.get("growth_rate", {})
            if growth:
                g1, g2 = st.columns(2)
                with g1: st.metric("Growth Rate", f"{growth.get('growth_rate_mm_per_scan', growth.get('growth_mm', 0)):.2f} mm/scan")
                with g2: st.metric("Trend", growth.get("velocity", growth.get("trend", "—")))
        else:
            st.info("No scans recorded.")

    # Report
    if scans:
        st.markdown('<div class="sh"><div class="sh-icon">📝</div> AI Report</div>', unsafe_allow_html=True)
        latest = scans[-1]
        sid = latest["scan_id"]

        if latest.get("has_report"):
            report_data = api_get(f"/api/scans/{sid}")
            if report_data and report_data.get("ai_report_draft"):
                with st.expander("View Report", expanded=True):
                    st.markdown(report_data["ai_report_draft"])
        else:
            if st.button("🤖 Generate Report", key="gr"):
                with st.spinner("Generating via LangGraph + Groq..."):
                    r = api_post_json("/api/generate_report", {"scan_id": sid})
                    if r: st.success("Done!"); st.rerun()

        # Approval
        st.markdown('<div class="sh"><div class="sh-icon">✅</div> Doctor Approval</div>', unsafe_allow_html=True)
        status_val = latest.get("status", "Pending")

        if status_val == "Approved":
            st.markdown('<div class="badge-approved">✅ Approved</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-pending">⏳ Pending</div>', unsafe_allow_html=True)
            ac1, ac2 = st.columns([3, 1])
            with ac1:
                notes = st.text_area("Notes", key="dn", placeholder="Clinical remarks...", label_visibility="collapsed")
            with ac2:
                st.write(""); st.write("")
                if st.button("✅ Approve", type="primary", key="ab"):
                    api_post_json(f"/api/scans/{sid}/remarks", {"status": "Approved", "doctor_notes": notes or None})
                    st.success("Approved!"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: NEW SCAN
# ═══════════════════════════════════════════════════════════════════════════════

def view_new_scan():
    brand(subtitle="New Scan Analysis")

    if st.button("← Back", key="bn"):
        nav("HOME"); st.rerun()

    # Step 1
    st.markdown('<div class="step-section animate-in">', unsafe_allow_html=True)
    st.markdown('<span class="step-badge">1</span> **Patient Number**', unsafe_allow_html=True)
    pnum = st.text_input("Patient Number", placeholder="e.g. P-1001", key="spn", label_visibility="collapsed")

    matched = None
    is_new = False

    if pnum and pnum.strip():
        try:
            r = requests.get(f"{API_BASE}/api/patients/lookup", params={"number": pnum.strip()}, timeout=5)
            if r.status_code == 200:
                matched = r.json()
                st.session_state.previous_risk = matched.get("latest_risk")
                st.markdown(f"""
                <div class="match-found">
                    ✅ <strong>{matched['name']}</strong> — {matched.get('scan_count',0)} scan(s)
                    {(' · ' + pill_html(matched.get('latest_risk',''))) if matched.get('latest_risk') else ''}
                </div>
                """, unsafe_allow_html=True)
            elif r.status_code == 404:
                is_new = True
                st.session_state.previous_risk = None
                st.markdown('<div class="match-new">🆕 New patient — enter name below</div>', unsafe_allow_html=True)
        except Exception:
            st.warning("Backend connection issue.")

    st.markdown('</div>', unsafe_allow_html=True)

    # Step 1b
    new_name = None
    if is_new:
        st.markdown('<div class="step-section animate-in animate-in-delay-1">', unsafe_allow_html=True)
        st.markdown('<span class="step-badge">+</span> **Register Patient**', unsafe_allow_html=True)
        new_name = st.text_input("Name", placeholder="e.g. John Smith", key="nn", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)

    # Step 2
    st.markdown('<div class="step-section animate-in animate-in-delay-2">', unsafe_allow_html=True)
    st.markdown('<span class="step-badge">2</span> **Upload CT Scan**', unsafe_allow_html=True)
    uploaded = st.file_uploader("Scan", type=["jpg","jpeg","png","bmp","tif","tiff"], label_visibility="collapsed")
    if uploaded:
        st.image(uploaded, width=200)
    st.markdown('</div>', unsafe_allow_html=True)

    # Step 3
    st.markdown('<div class="step-section animate-in animate-in-delay-3">', unsafe_allow_html=True)
    st.markdown('<span class="step-badge">3</span> **Analyze**', unsafe_allow_html=True)

    ready = uploaded and pnum and (matched or (is_new and new_name))

    if st.button("🧠 Run AI Analysis", type="primary", disabled=not ready, use_container_width=True):
        patient_id = None

        if is_new and new_name:
            res = api_post_json("/api/patients", {"patient_number": pnum.strip(), "name": new_name.strip()})
            if res: patient_id = res["id"]
            else: st.markdown('</div>', unsafe_allow_html=True); return
        elif matched:
            patient_id = matched["id"]

        if patient_id:
            with st.spinner("Running inference..."):
                uploaded.seek(0)
                result = api_post_form("/api/analyze", {"patient_id": str(patient_id)}, {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)})
                if result:
                    st.session_state.scan_result = result
                    with st.spinner("Generating report..."):
                        rpt = api_post_json("/api/generate_report", {"scan_id": result["scan_id"]})
                        if rpt: st.session_state.scan_result["report"] = rpt.get("report_draft", "")
                    nav("SCAN_RESULT"); st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# VIEW: SCAN RESULT
# ═══════════════════════════════════════════════════════════════════════════════

def view_scan_result():
    result = st.session_state.scan_result
    if not result:
        nav("HOME"); st.rerun(); return

    brand(subtitle="Analysis Complete")

    risk = result.get("risk_level", "Low")
    prob = result.get("malignancy_probability", 0)
    conf = result.get("confidence", 0)
    pred = result.get("predicted_class", "—")
    tmm = result.get("tumor_diameter_mm", 0)

    prob_color = "#EF4444" if prob > 0.6 else "#F59E0B" if prob > 0.3 else "#22C55E"

    st.markdown(f"""
    <div class="metric-grid">
        <div class="metric-tile">
            <div class="metric-tile-value" style="color:{prob_color}">{prob*100:.1f}%</div>
            <div class="metric-tile-label">Probability</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-value">{pred}</div>
            <div class="metric-tile-label">Class</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-value">{conf*100:.1f}%</div>
            <div class="metric-tile-label">Confidence</div>
        </div>
        <div class="metric-tile">
            <div class="metric-tile-value">{tmm:.1f}mm</div>
            <div class="metric-tile-label">Est. Size</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Categorization
    folder = risk_to_folder(risk)
    st.markdown(f"""
    <div style="text-align:center; margin:1rem 0;">
        <span style="font-size:0.9rem; font-weight:600; color:#64748B;">Categorized as</span>
        <span style="margin-left:0.5rem;">{pill_html(risk)}</span>
    </div>
    """, unsafe_allow_html=True)

    prev_risk = st.session_state.previous_risk
    if prev_risk and prev_risk != risk:
        prev_lbl = folder_label(risk_to_folder(prev_risk))
        new_lbl = folder_label(folder)
        st.markdown(f'<div class="cat-alert">⚠️ Moved from <strong>{prev_lbl}</strong> → <strong>{new_lbl}</strong></div>', unsafe_allow_html=True)

    c3d, crpt = st.columns([1, 1])
    with c3d:
        st.markdown('<div class="sh"><div class="sh-icon">🫁</div> 3D Visualization</div>', unsafe_allow_html=True)
        xc_r = result.get("x_coordinate", 256)
        ll_r = "Left Lung" if (xc_r / 512.0) < 0.5 else "Right Lung"
        st.markdown('<div style="background:#0A0D12;border:1px solid #1F2937;border-radius:4px;padding:0.25rem;">', unsafe_allow_html=True)
        fig = render_3d_lung(xc_r, tmm)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        render_lung_hud(xc_r, tmm, risk, ll_r)

    with crpt:
        st.markdown('<div class="sh"><div class="sh-icon">📝</div> AI Report</div>', unsafe_allow_html=True)
        rpt = result.get("report", "")
        if rpt:
            with st.expander("View Report", expanded=True):
                st.markdown(rpt)
        else:
            st.info("Report pending.")

    st.markdown("---")
    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button("👤 Patient Detail", use_container_width=True):
            nav("PATIENT", selected_patient_id=result["patient_id"]); st.rerun()
    with b2:
        if st.button("🔬 Another Scan", use_container_width=True):
            st.session_state.scan_result = None; nav("NEW_SCAN"); st.rerun()
    with b3:
        if st.button("🏠 Home", use_container_width=True):
            st.session_state.scan_result = None; nav("HOME"); st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════════════════════════════

{"HOME": view_home, "FOLDER": view_folder, "PATIENT": view_patient,
 "NEW_SCAN": view_new_scan, "SCAN_RESULT": view_scan_result
}.get(st.session_state.view, view_home)()

st.markdown('<div class="footer">🫁 LungCare — For clinical decision support only</div>', unsafe_allow_html=True)
