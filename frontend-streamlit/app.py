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

def render_3d_lung(x_coord=256, tumor_mm=5.0, show_previous=False, prev_size=None):
    """Render highly realistic medical 3D lung with nodule."""
    fig = go.Figure()

    # High resolution mesh for organic realism
    u = np.linspace(0, 2 * np.pi, 120)
    v = np.linspace(0, np.pi, 120)
    U, V = np.meshgrid(u, v)

    # Biological tissue lighting configuration
    flesh_lighting = dict(
        ambient=0.4,
        diffuse=0.7,
        specular=0.1,
        roughness=0.8,
        fresnel=4.0
    )

    # Medical tissue colorscale (CT volume render style - translucent gray/pink)
    lung_colorscale = [
        [0.0, 'rgba(180, 180, 190, 0.05)'], 
        [0.5, 'rgba(200, 170, 170, 0.15)'], 
        [1.0, 'rgba(220, 190, 190, 0.25)']
    ]

    # Right lung (organic deformations to simulate lobes)
    rx, ry, rz = 3.2, 4.6, 5.8
    x_r = rx * np.sin(V) * np.cos(U) + 3.8
    y_r = ry * np.sin(V) * np.sin(U)
    z_r = rz * np.cos(V)
    # Complex noise for biological surface irregularity
    deform1 = 0.2 * np.sin(3*U) * np.sin(2*V) 
    deform2 = 0.15 * np.sin(5*U) * np.cos(3*V)
    deform3 = 0.05 * np.sin(10*U) * np.cos(8*V)
    deform = deform1 + deform2 + deform3
    
    # Flatten the inner medial surface where the heart sits (cardiac notch area)
    medial_flattening = np.clip(np.cos(U) * np.sin(V), -1, 0) * 1.5
    x_r += deform + medial_flattening
    y_r += deform * 0.8
    z_r += deform * 0.5

    fig.add_trace(go.Surface(
        x=x_r, y=y_r, z=z_r,
        colorscale=lung_colorscale,
        showscale=False, opacity=0.4,
        lighting=flesh_lighting,
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        name='Right Lung',
        hovertemplate="<b>Right Lung</b><extra></extra>"
    ))

    # Left lung
    lx, ly, lz = 3.0, 4.3, 5.5
    x_l = lx * np.sin(V) * np.cos(U) - 3.8
    y_l = ly * np.sin(V) * np.sin(U)
    z_l = lz * np.cos(V)
    deform_l = 0.18 * np.sin(3*U) * np.sin(2*V) + 0.1 * np.sin(4*U) * np.cos(2*V) + 0.05 * np.sin(9*U) * np.cos(7*V)
    
    # Cardiac notch (Left lung is slightly smaller due to heart)
    cardiac_notch = np.clip(-np.cos(U) * np.sin(V), -1, 0) * 2.5
    x_l += deform_l + cardiac_notch
    y_l += deform_l * 0.7
    z_l += deform_l * 0.5

    fig.add_trace(go.Surface(
        x=x_l, y=y_l, z=z_l,
        colorscale=lung_colorscale,
        showscale=False, opacity=0.4,
        lighting=flesh_lighting,
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        name='Left Lung (Cardiac Notch)',
        hovertemplate="<b>Left Lung</b><extra></extra>"
    ))

    # Realistic Nodule (Tumor mass)
    # Map x_coord exactly to correct lung volume
    if x_coord < 256:
        # Left lung: roughly x between -5.8 to -1.8
        nodule_x = (x_coord / 255.0) * 3.5 - 5.5
    else:
        # Right lung: roughly x between 1.8 to 5.8
        nodule_x = ((x_coord - 256) / 256.0) * 3.5 + 2.0

    nodule_r = max(0.3, min(tumor_mm / 6.0, 1.8))
    
    # Very high frequency noise for a bumpy, malignant-looking organic mass
    nU, nV = np.meshgrid(np.linspace(0, 2*np.pi, 60), np.linspace(0, np.pi, 60))
    noise = 0.15 * np.sin(8*nU) * np.cos(6*nV) + 0.08 * np.sin(15*nU + 5*nV) + 0.05 * np.random.rand(*nU.shape)
    nr = nodule_r + noise

    # Malignant masses are highly vascularized and dense (dark red/brownish in false-color CT)
    nodule_colorscale = [[0.0, 'rgba(120, 20, 20, 1.0)'], [1.0, 'rgba(180, 50, 50, 1.0)']]
    
    lung_side_name = "Left Lung" if nodule_x < 0 else "Right Lung"
    
    fig.add_trace(go.Surface(
        x=nr*np.sin(nV)*np.cos(nU) + nodule_x,
        y=nr*np.sin(nV)*np.sin(nU) + 0.5 + noise*0.5,
        z=nr*np.cos(nV) + 0.5 + noise*0.5,
        colorscale=nodule_colorscale,
        showscale=False, opacity=1.0,
        lighting=dict(ambient=0.3, diffuse=0.9, specular=0.1, roughness=1.0),
        contours=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
        name=f'Malignant Mass ({tumor_mm:.1f}mm)',
        hovertemplate=f"<b>Malignant Mass</b><br>{tumor_mm:.1f}mm<br>Located in {lung_side_name}<extra></extra>"
    ))

    # Measurement Callout line (Clinical style)
    fig.add_trace(go.Scatter3d(
        x=[nodule_x, nodule_x + nodule_r + 1.5], 
        y=[0.5, 0.5], 
        z=[0.5, 0.5 + nodule_r + 1.5],
        mode='lines',
        line=dict(color='rgba(255,255,255,0.6)', width=2, dash='dot'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatter3d(
        x=[nodule_x + nodule_r + 1.5], y=[0.5], z=[0.5 + nodule_r + 1.5],
        mode='text',
        text=[f'Dx: {tumor_mm:.1f}mm | {lung_side_name}'], textposition='top right',
        textfont=dict(color='#FAFAFA', size=13, family='monospace'),
        showlegend=False,
        hoverinfo='skip'
    ))

    fig.update_layout(
        scene=dict(
            xaxis=dict(visible=False, range=[-8, 8]),
            yaxis=dict(visible=False, range=[-7, 7]),
            zaxis=dict(visible=False, range=[-8, 10]),
            bgcolor='rgba(10,10,10,0)',
            camera=dict(
                eye=dict(x=1.3, y=1.0, z=0.4),
                up=dict(x=0, y=0, z=1)
            ),
            aspectmode='manual', aspectratio=dict(x=1.2, y=1, z=1.3),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=450, showlegend=False,
    )
    return fig


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
            fig = render_3d_lung(latest.get("x_coordinate", 256), latest["tumor_diameter_mm"], prev is not None, prev)
            st.plotly_chart(fig, use_container_width=True)
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
        fig = render_3d_lung(result.get("x_coordinate", 256), tmm)
        st.plotly_chart(fig, use_container_width=True)

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
