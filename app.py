"""
Streamlit frontend for Market Analyzer
Main entry point for the application
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Market Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add title
st.title("📈 Market Analyzer")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.write("Configure your analysis settings here.")

# Main content
col1, col2 = st.columns(2)

with col1:
    st.subheader("Data")
    st.write("Upload and manage your market data.")

with col2:
    st.subheader("Model")
    st.write("Load and run your XGBoost predictions.")

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: gray;'>Market Analyzer v0.1 | Powered by Streamlit</p>",
    unsafe_allow_html=True,
)
