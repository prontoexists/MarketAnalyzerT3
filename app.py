import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from streamlit_folium import st_folium
import folium
import os

st.set_page_config(
    page_title="Mollecul | AI Real Estate Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- STYLING ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #E8F4F8 0%, #FFF4E8 100%);
        background-attachment: fixed;
    }

    .stApp, .stApp p, .stApp div, .stApp span, .stApp label {
        color: #111111 !important;
    }

    /* Headings */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        color: #1D3557 !important;
    }

    /* Selectbox / input labels */
    label, .stSelectbox label, .stMarkdown {
        color: #111111 !important;
    }

    [data-testid="stMetric"] {
        background-color: rgba(255, 255, 255, 0.4);
        padding: 15px;
        border-radius: 15px;
        backdrop-filter: blur(5px);
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {
        color: #111111 !important;
    }

    [data-testid="stDataFrame"] {
        color: #111111 !important;
    }

    .stMarkdown, .stPlotlyChart {
        animation: fadeIn 1.2s;
    }

    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    </style>
""", unsafe_allow_html=True)

# --- PATHS ---
BASE_DIR = os.path.dirname(__file__)
logo_path = os.path.join(BASE_DIR, "logo.png")
csv_path = os.path.join(BASE_DIR, "data", "dfw_real_estate.csv")

@st.cache_data(ttl=3600)
def load_real_estate_data():
    df = pd.read_csv(csv_path)

    df.columns = [col.strip() for col in df.columns]

    numeric_cols = [
        "PRICE", "BEDS", "BATHS", "SQUARE FEET", "LOT SIZE",
        "YEAR BUILT", "DAYS ON MARKET", "$/SQUARE FEET",
        "HOA/MONTH", "LATITUDE", "LONGITUDE"
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\$,]", "", regex=True)
                .str.replace(r"--", "", regex=True)
                .str.strip()
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    string_cols = ["CITY", "PROPERTY TYPE", "STATUS", "ADDRESS"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    required_cols = ["CITY", "PRICE", "LATITUDE", "LONGITUDE"]
    existing_required = [c for c in required_cols if c in df.columns]
    df = df.dropna(subset=existing_required)

    if "PRICE" in df.columns and "SQUARE FEET" in df.columns:
        df["PRICE_PER_SQFT_CALC"] = df["PRICE"] / df["SQUARE FEET"].replace(0, np.nan)

    if "$/SQUARE FEET" in df.columns:
        df["PRICE_PER_SQFT_FINAL"] = df["$/SQUARE FEET"].fillna(df["PRICE_PER_SQFT_CALC"])
    else:
        df["PRICE_PER_SQFT_FINAL"] = df["PRICE_PER_SQFT_CALC"]

    if "DAYS ON MARKET" in df.columns:
        df["DAYS_ON_MARKET_SAFE"] = df["DAYS ON MARKET"].fillna(df["DAYS ON MARKET"].median())
    else:
        df["DAYS_ON_MARKET_SAFE"] = 0

    df["investment_score"] = (
        df["PRICE_PER_SQFT_FINAL"].fillna(0) * 0.6 +
        (1 / (df["DAYS_ON_MARKET_SAFE"] + 1)) * 1000 * 0.3 +
        df["BEDS"].fillna(0) * 2 * 0.1
    )

    return df


@st.cache_data(ttl=3600)
def get_price_trends(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()

    if "YEAR BUILT" in city_df.columns and city_df["YEAR BUILT"].notna().sum() > 0:
        trend_df = (
            city_df.dropna(subset=["YEAR BUILT", "PRICE"])
            .groupby("YEAR BUILT", as_index=False)["PRICE"]
            .mean()
            .sort_values("YEAR BUILT")
        )
        trend_df = trend_df.rename(columns={"YEAR BUILT": "Period", "PRICE": "Value"})
        trend_df["Period"] = trend_df["Period"].astype(int).astype(str)
        return trend_df.tail(20)

    return pd.DataFrame({
        "Period": ["Sample 1", "Sample 2", "Sample 3"],
        "Value": [0, 0, 0]
    })


@st.cache_data(ttl=3600)
def get_roi_by_type(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()

    roi_df = (
        city_df.dropna(subset=["PROPERTY TYPE", "PRICE_PER_SQFT_FINAL"])
        .groupby("PROPERTY TYPE", as_index=False)["PRICE_PER_SQFT_FINAL"]
        .mean()
        .sort_values("PRICE_PER_SQFT_FINAL", ascending=False)
        .head(10)
    )
    roi_df = roi_df.rename(columns={"PRICE_PER_SQFT_FINAL": "ROI_PROXY"})
    return roi_df


@st.cache_data(ttl=3600)
def get_top_neighborhoods(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()

    display_cols = [
        "ADDRESS", "PROPERTY TYPE", "PRICE", "BEDS", "BATHS",
        "SQUARE FEET", "DAYS ON MARKET", "investment_score"
    ]
    display_cols = [c for c in display_cols if c in city_df.columns]

    top_df = city_df.sort_values("investment_score", ascending=False)[display_cols].head(12).copy()

    if "PRICE" in top_df.columns:
        top_df["PRICE"] = top_df["PRICE"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    if "investment_score" in top_df.columns:
        top_df["investment_score"] = top_df["investment_score"].round(2)

    return top_df


if not os.path.exists(csv_path):
    st.error(f"CSV not found at: {csv_path}")
    st.stop()

df = load_real_estate_data()

st.write("")
_, col_logo, _ = st.columns([2, 1, 2])
with col_logo:
    if os.path.exists(logo_path):
        st.image(logo_path, use_container_width=True)
    else:
        st.title("🧬 MOLLECUL")

st.markdown(
    "<h3 style='text-align: center; color: #1D3557; font-weight: 400;'>Objective. Data-Driven. Transparent.</h3>",
    unsafe_allow_html=True
)
st.write("")

f1, f2, f3 = st.columns(3)
with f1:
    with st.container(border=True):
        st.subheader("🤖 ML Prediction")
        st.write("Predictive analysis trained on real listing structure, pricing, and market activity.")
with f2:
    with st.container(border=True):
        st.subheader("📈 Macro Analysis")
        st.write("Analyze market behavior using pricing trends, property types, and time-on-market signals.")
with f3:
    with st.container(border=True):
        st.subheader("🔍 Transparency")
        st.write("Grounded on real Dallas–Fort Worth listing data instead of synthetic demo-only points.")

st.divider()

st.title("🏙️ Real Estate Intelligence Suite")

available_cities = sorted(df["CITY"].dropna().unique().tolist())

default_city = "Dallas" if "Dallas" in available_cities else available_cities[0]

col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 2])
with col_ctrl1:
    target_city = st.selectbox("Market Location", available_cities, index=available_cities.index(default_city))

city_df = df[df["CITY"].str.lower() == target_city.lower()].copy()

st.write("")

m1, m2, m3, m4 = st.columns(4)

avg_price = city_df["PRICE"].mean() if "PRICE" in city_df.columns else np.nan
avg_ppsqft = city_df["PRICE_PER_SQFT_FINAL"].mean() if "PRICE_PER_SQFT_FINAL" in city_df.columns else np.nan
avg_dom = city_df["DAYS ON MARKET"].mean() if "DAYS ON MARKET" in city_df.columns else np.nan
listing_count = len(city_df)

with m1:
    st.metric("Average Price", f"${avg_price:,.0f}" if pd.notna(avg_price) else "N/A")
with m2:
    st.metric("Avg $ / Sq Ft", f"${avg_ppsqft:,.0f}" if pd.notna(avg_ppsqft) else "N/A")
with m3:
    st.metric("Avg Days on Market", f"{avg_dom:.1f}" if pd.notna(avg_dom) else "N/A")
with m4:
    st.metric("Listings", f"{listing_count:,}")

st.divider()

st.subheader("📈 Market Performance Analytics")
c_left, c_right = st.columns(2)

with c_left:
    st.markdown("**Average Price Trend**")
    trend_data = get_price_trends(df, target_city)

    fig1 = px.line(
        trend_data,
        x="Period",
        y="Value",
        template="plotly_white"
    )
    fig1.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        xaxis_title="Period",
        yaxis_title="Average Price"
    )
    st.plotly_chart(fig1, use_container_width=True)

with c_right:
    st.markdown("**Price Strength by Property Type**")
    roi_data = get_roi_by_type(df, target_city)

    fig2 = px.bar(
        roi_data,
        x="PROPERTY TYPE",
        y="ROI_PROXY",
        color="ROI_PROXY",
        color_continuous_scale="Sunset"
    )
    fig2.update_layout(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=350,
        xaxis_title="Property Type",
        yaxis_title="Avg $ / Sq Ft"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("📍 Interactive Property Map")

city_coords = {
    "Dallas": [32.7767, -96.7970],
    "Fort Worth": [32.7555, -97.3308],
    "Arlington": [32.7357, -97.1081],
    "Plano": [33.0198, -96.6989],
    "Frisco": [33.1507, -96.8236],
    "Irving": [32.8140, -96.9489],
    "Richardson": [32.9483, -96.7299],
    "Garland": [32.9126, -96.6389],
    "Grand Prairie": [32.7459, -96.9978],
    "Mansfield": [32.5632, -97.1417],
    "Austin": [30.2672, -97.7431],
    "Houston": [29.7604, -95.3698],
}

if target_city in city_coords:
    location = city_coords[target_city]
else:
    location = [
        city_df["LATITUDE"].mean(),
        city_df["LONGITUDE"].mean()
    ]

map_df = city_df.dropna(subset=["LATITUDE", "LONGITUDE", "PRICE"]).copy()

m = folium.Map(location=location, zoom_start=11, tiles="CartoDB positron")

sample_size = 400
if len(map_df) > sample_size:
    map_df = map_df.sample(sample_size, random_state=42)

price_threshold = map_df["PRICE"].median()

for _, row in map_df.iterrows():
    marker_color = "#457B9D" if row["PRICE"] >= price_threshold else "#E9C46A"

    popup_parts = []
    if "ADDRESS" in row and pd.notna(row["ADDRESS"]):
        popup_parts.append(f"<b>{row['ADDRESS']}</b>")
    popup_parts.append(f"Price: ${row['PRICE']:,.0f}")
    if "BEDS" in row and pd.notna(row["BEDS"]):
        popup_parts.append(f"Beds: {row['BEDS']}")
    if "BATHS" in row and pd.notna(row["BATHS"]):
        popup_parts.append(f"Baths: {row['BATHS']}")
    if "SQUARE FEET" in row and pd.notna(row["SQUARE FEET"]):
        popup_parts.append(f"Sq Ft: {row['SQUARE FEET']:,.0f}")

    popup_text = "<br>".join(popup_parts)

    folium.CircleMarker(
        [row["LATITUDE"], row["LONGITUDE"]],
        radius=6,
        popup=popup_text,
        color=marker_color,
        fill=True,
        fill_color=marker_color,
        fill_opacity=0.7
    ).add_to(m)

st_folium(m, width="100%", height=500, key=f"map_{target_city}")

st.write("---")
st.subheader(f"📋 Top Performing Listings in {target_city}")

top_table = get_top_neighborhoods(df, target_city)
st.dataframe(top_table, use_container_width=True, hide_index=True)

with st.expander("View Raw Dataset Preview"):
    st.dataframe(city_df.head(50), use_container_width=True)