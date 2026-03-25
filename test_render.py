import sys
import os
import plotly
# Need to patch st out
sys.path.insert(0, 'frontend-streamlit')
from unittest.mock import MagicMock
import streamlit as st
st.cache_data = lambda *args, **kwargs: lambda f: f
import app

fig = app.render_3d_lung()
if fig is not None:
    print("Success! Traces:", len(fig.data))
    
