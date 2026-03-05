import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Mollecul | AI Real Estate Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STYLING ---
st.markdown("""
    <style>
    /* Global Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #E8F4F8 0%, #FFF4E8 100%);
        background-attachment: fixed;
    }
    
    /* Smooth transition for elements */
    .stMarkdown, .stPlotlyChart {
        animation: fadeIn 1.2s;
    }
    @keyframes fadeIn { 0% { opacity: 0; } 100% { opacity: 1; } }

    /* Modern Metric Styling */
    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.4);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }
    </style>
    """, unsafe_allow_html=True)

# --- CACHED DATA ---
@st.cache_data(ttl=3600)
def get_cached_map_data(lat, lon):
    return pd.DataFrame({
        'lat': lat + np.random.uniform(-0.015, 0.015, 12),
        'lon': lon + np.random.uniform(-0.015, 0.015, 12),
        'score': np.random.randint(70, 99, 12)
    })

@st.cache_data(ttl=3600)
def get_cached_stock_data():
    dates = pd.date_range(start="2024-01-01", periods=30, freq='D')
    return pd.DataFrame({
        'Date': dates,
        'S&P 500 Index': np.cumsum(np.random.randn(30) + 0.1) + 100,
        'Real Estate Index': np.cumsum(np.random.randn(30) + 0.05) + 95
    })

@st.cache_data(ttl=3600)
def get_property_data():
    return pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
        'Value': [45000, 52000, 48000, 61000, 55000, 67000, 72000, 69000],
        'ROI': [12.5, 13.2, 12.8, 14.1, 13.5, 15.2, 16.0, 15.5]
    })

@st.cache_data(ttl=3600)
def get_roi_by_type():
    return pd.DataFrame({
        'Type': ['Residential', 'Commercial', 'Apartments', 'Mixed-Use', 'Luxury'],
        'ROI': [14.5, 12.8, 15.2, 13.9, 11.3]
    })

logo_path = os.path.join(os.path.dirname(__file__), "logo.png")

# --- HERO SECTION ---
st.write("")
_, col_logo, _ = st.columns([2, 1, 2])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.title("🧬 MOLLECUL")

st.markdown("<h3 style='text-align: center; color: #1D3557; font-weight: 400;'>Objective. Data-Driven. Transparent.</h3>", unsafe_allow_html=True)
st.write("")

# Feature Overview
f1, f2, f3 = st.columns(3)
with f1:
    with st.container(border=True):
        st.subheader("🤖 ML Prediction")
        st.write("Predictive analysis trained on school districts, crime rates, and environment.")
with f2:
    with st.container(border=True):
        st.subheader("📈 Macro Analysis")
        st.write("Accounting for stock market trends and economic indicators.")
with f3:
    with st.container(border=True):
        st.subheader("🔍 Transparency")
        st.write("Completely objective approach to intelligent property investment.")

st.divider()

# --- ANALYTICS CHARTS ---
st.subheader("📈 Market Performance Analytics")
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("**Property Value Trends**")
    p_data = get_property_data()
    fig1 = px.line(p_data, x='Month', y='Value', template="plotly_white", color_discrete_sequence=['#457B9D'])
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=350)
    st.plotly_chart(fig1, use_container_width=True)

with c_right:
    st.markdown("**ROI by Property Type**")
    roi_data = get_roi_by_type()
    fig2 = px.bar(roi_data, x='Type', y='ROI', color='ROI', color_continuous_scale='Sunset')
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', height=350)
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

# --- INTELLIGENCE SUITE (LANDING CONTENT) ---
st.title("🏙️ Real Estate Intelligence Suite")

# Quick Controls
col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 2])
with col_ctrl1:
    target_city = st.selectbox("Market Location", ["Dallas", "Austin", "Houston"])

st.write("")

# --- MAP SECTION ---
st.subheader("📍 Interactive Property Map")
city_coords = {"Dallas": [32.98, -96.75], "Austin": [30.26, -97.74], "Houston": [29.76, -95.36]}
location = city_coords[target_city]

map_df = get_cached_map_data(location[0], location[1])
m = folium.Map(location=location, zoom_start=13, tiles="CartoDB positron")

for _, row in map_df.iterrows():
    marker_color = "#457B9D" if row['score'] > 88 else "#E9C46A"
    folium.CircleMarker(
        [row['lat'], row['lon']],
        radius=10,
        popup=f"Score: {int(row['score'])}%",
        color=marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.7
    ).add_to(m)

st_folium(m, width="100%", height=500, key=f"map_{target_city}")

# --- RANKING TABLE ---
st.write("---")
st.subheader(f"📋 Top Performing Neighborhoods in {target_city}")
mock_table = pd.DataFrame({
    "Rank": ["🥇 1", "🥈 2", "🥉 3", "4"],
    "Neighborhood": ["Richardson North", "Campbell Ridge", "Canyon Creek", "University Village"],
    "Mollecul Score": [98, 94, 91, 89],
    "Avg Price": ["$485K", "$425K", "$515K", "$395K"],
    "ROI": ["15.2%", "14.8%", "16.5%", "13.9%"]
})

st.dataframe(mock_table, use_container_width=True, hide_index=True)