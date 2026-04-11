import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_folium import st_folium
import folium
import streamlit.components.v1 as components
from services.model_logic import load_model, predict

st.set_page_config(
    page_title="Mollecul | AI Real Estate Intelligence",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# DATA
# =============================================================================
BASE_DIR = os.path.dirname(__file__)
csv_path = os.path.join(BASE_DIR, "data", "dfw_real_estate.csv")

@st.cache_data(ttl=3600)
def load_real_estate_data():
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

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

    for col in ["CITY", "PROPERTY TYPE", "STATUS", "ADDRESS"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    required = [c for c in ["CITY", "PRICE", "LATITUDE", "LONGITUDE"] if c in df.columns]
    df = df.dropna(subset=required)

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
        df["PRICE_PER_SQFT_FINAL"].fillna(0) * 0.60 +
        (1 / (df["DAYS_ON_MARKET_SAFE"] + 1)) * 1000 * 0.28 +
        df["BEDS"].fillna(0) * 2.0 * 0.12
    )

    mn = df["investment_score"].min()
    mx = df["investment_score"].max()
    if mx != mn:
        df["investment_score"] = ((df["investment_score"] - mn) / (mx - mn)) * 100
    else:
        df["investment_score"] = 50

    return df

@st.cache_data(ttl=3600)
def get_price_trends(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()
    if "YEAR BUILT" in city_df.columns and city_df["YEAR BUILT"].notna().sum() > 0:
        trend_df = (
            city_df.dropna(subset=["YEAR BUILT", "PRICE"])
            .groupby("YEAR BUILT", as_index=False)["PRICE"].mean()
            .sort_values("YEAR BUILT")
        )
        trend_df = trend_df.rename(columns={"YEAR BUILT": "Period", "PRICE": "Value"})
        trend_df["Period"] = trend_df["Period"].astype(int).astype(str)
        return trend_df.tail(20)

    return pd.DataFrame({"Period": ["1", "2", "3"], "Value": [0, 0, 0]})

@st.cache_data(ttl=3600)
def get_roi_by_type(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()
    roi_df = (
        city_df.dropna(subset=["PROPERTY TYPE", "PRICE_PER_SQFT_FINAL"])
        .groupby("PROPERTY TYPE", as_index=False)["PRICE_PER_SQFT_FINAL"].mean()
        .sort_values("PRICE_PER_SQFT_FINAL", ascending=False)
        .head(8)
    )
    return roi_df.rename(columns={"PRICE_PER_SQFT_FINAL": "ROI_PROXY"})

@st.cache_data(ttl=3600)
def get_top_listings(df, city):
    city_df = df[df["CITY"].str.lower() == city.lower()].copy()

    cols = [c for c in [
        "ADDRESS", "PROPERTY TYPE", "PRICE", "BEDS", "BATHS",
        "SQUARE FEET", "DAYS ON MARKET", "investment_score"
    ] if c in city_df.columns]

    top_df = city_df.sort_values("investment_score", ascending=False)[cols].head(6).copy()

    if "PRICE" in top_df.columns:
        top_df["PRICE"] = top_df["PRICE"].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    if "BEDS" in top_df.columns:
        top_df["BEDS"] = top_df["BEDS"].apply(lambda x: int(x) if pd.notna(x) else "—")
    if "BATHS" in top_df.columns:
        top_df["BATHS"] = top_df["BATHS"].apply(lambda x: round(float(x), 1) if pd.notna(x) else "—")
    if "SQUARE FEET" in top_df.columns:
        top_df["SQUARE FEET"] = top_df["SQUARE FEET"].apply(lambda x: f"{int(x):,}" if pd.notna(x) else "—")
    if "DAYS ON MARKET" in top_df.columns:
        top_df["DAYS ON MARKET"] = top_df["DAYS ON MARKET"].apply(lambda x: int(x) if pd.notna(x) else "—")
    if "investment_score" in top_df.columns:
        top_df["investment_score"] = top_df["investment_score"].round(2)

    return top_df


@st.cache_resource
def get_valuation_model():
    model_path = os.path.join(BASE_DIR, "services", "VER4_property_valuation_model.joblib")
    return load_model(model_path)


def map_dashboard_row_to_model_input(row: pd.Series) -> pd.DataFrame:
    """
    Map the dashboard CSV row into the ATTOM-style schema expected by
    services/model_logic.py. Missing fields are allowed.
    """
    beds_val = row.get("BEDS", np.nan)
    price_val = row.get("PRICE", np.nan)

    mapped = {
        "lat": row.get("LATITUDE", np.nan),
        "lng": row.get("LONGITUDE", np.nan),
        "beds": beds_val,
        "bathsFull": row.get("BATHS", np.nan),
        "bathsTotal": row.get("BATHS", np.nan),
        "bathsHalf": np.nan,
        "sqft": row.get("SQUARE FEET", np.nan),
        "lotSqft": row.get("LOT SIZE", np.nan),
        "yearBuilt": row.get("YEAR BUILT", np.nan),
        "salePrice": price_val,
        "pricePerSqft": row.get("PRICE_PER_SQFT_FINAL", np.nan),
        "pricePerBed": (
            price_val / beds_val
            if pd.notna(price_val) and pd.notna(beds_val) and beds_val not in [0, 0.0]
            else np.nan
        ),
        "city": row.get("CITY", None),
        "propertyType": row.get("PROPERTY TYPE", None),
        "zip": row.get("ZIP OR POSTAL CODE", None),
        "saleType": row.get("SALE TYPE", None),
        "address": row.get("ADDRESS", None),
    }

    return pd.DataFrame([mapped])


@st.cache_resource
def get_forecast_artifacts():
    model_path = os.path.join(BASE_DIR, "services", "VER4_round2_forecast_model.joblib")
    scaler_path = os.path.join(BASE_DIR, "services", "VER4_round2_scaler.joblib")
    feature_cols_path = os.path.join(BASE_DIR, "services", "VER4_round2_feature_cols.json")

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    with open(feature_cols_path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)

    return model, scaler, feature_cols


def build_forecast_input(overrides: dict | None = None) -> pd.DataFrame:
    """
    Build a single-row dataframe for the forecast model.
    Uses stable baseline defaults so the UI can preview the model safely
    without requiring a live macroeconomic feed.
    """
    current_month = int(pd.Timestamp.today().month)
    defaults = {
        "mortgage_rate_30yr": 6.75,
        "fed_funds_rate": 5.25,
        "housing_starts_south": 800.0,
        "existing_home_sales": 4.10,
        "new_home_sales": 0.70,
        "yield_spread": 0.25,
        "treasury_10yr": 4.25,
        "treasury_2yr": 4.00,
        "case_shiller_dallas": 330.0,
        "cs_dallas_mom_pct": 0.40,
        "cs_dallas_3mo_pct": 1.20,
        "cpi_shelter": 3.50,
        "unemployment_texas": 4.10,
        "labor_force_part_texas": 64.0,
        "wage_growth_texas": 4.00,
        "sp500_return": 0.50,
        "homebuilder_etf_return": 0.40,
        "reit_index_return": 0.20,
        "vix": 18.0,
        "oil_wti": 75.0,
        "txn_stock_return": 0.20,
        "att_stock_return": 0.10,
        "treasury_etf_return": -0.10,
        "mortgage_rate_30yr_lag1": 6.75,
        "fed_funds_rate_lag1": 5.25,
        "sp500_return_lag1": 0.50,
        "cs_dallas_mom_pct_lag1": 0.40,
        "vix_lag1": 18.0,
        "homebuilder_etf_return_lag1": 0.40,
        "month": current_month,
        "quarter": ((current_month - 1) // 3) + 1,
        "is_spring": 1 if current_month in [3, 4, 5] else 0,
        "is_summer": 1 if current_month in [6, 7, 8] else 0,
    }

    if overrides:
        defaults.update(overrides)

    month_val = int(defaults.get("month", current_month))
    defaults["quarter"] = ((month_val - 1) // 3) + 1
    defaults["is_spring"] = 1 if month_val in [3, 4, 5] else 0
    defaults["is_summer"] = 1 if month_val in [6, 7, 8] else 0

    defaults["yield_spread"] = float(defaults.get("treasury_10yr", 0.0)) - float(defaults.get("treasury_2yr", 0.0))
    defaults["mortgage_rate_30yr_lag1"] = float(defaults.get("mortgage_rate_30yr", 0.0))
    defaults["fed_funds_rate_lag1"] = float(defaults.get("fed_funds_rate", 0.0))
    defaults["sp500_return_lag1"] = float(defaults.get("sp500_return", 0.0))
    defaults["cs_dallas_mom_pct_lag1"] = float(defaults.get("cs_dallas_mom_pct", 0.0))
    defaults["vix_lag1"] = float(defaults.get("vix", 0.0))
    defaults["homebuilder_etf_return_lag1"] = float(defaults.get("homebuilder_etf_return", 0.0))

    _, _, feature_cols = get_forecast_artifacts()
    row = {col: defaults.get(col, 0.0) for col in feature_cols}
    return pd.DataFrame([row])


def forecast_predict(input_df: pd.DataFrame) -> float:
    model, scaler, feature_cols = get_forecast_artifacts()
    X = input_df[feature_cols]
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)
    return float(pred[0])

if not os.path.exists(csv_path):
    st.error(f"CSV not found at: {csv_path}")
    st.stop()

df = load_real_estate_data()

# =============================================================================
# GLOBAL STYLES
# =============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;1,9..40,300&display=swap');

:root {
  --bg: #06111f;
  --panel: #0b1c30;
  --panel-hover: #0e2236;
  --line: rgba(110, 196, 255, 0.09);
  --line-strong: rgba(44, 228, 223, 0.28);
  --text: #ecf6ff;
  --muted: #8baec8;
  --muted-2: #567592;
  --cyan: #2ce4df;
  --blue: #5ea6ff;
  --gold: #e7c65a;
  --purple: #b89eff;
  --shadow: 0 20px 60px rgba(0,0,0,0.35);
  --glow-cyan: 0 0 24px rgba(44,228,223,0.18);
}

/* ── Reset Streamlit chrome ── */
html, body, .stApp {
  background: var(--bg) !important;
  color: var(--text) !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"],
[data-testid="stMain"], .main > div, section[data-testid="stSidebar"] {
  background: transparent !important;
}
header[data-testid="stHeader"] {
  background: rgba(6,17,31,0.90) !important;
  border-bottom: 1px solid rgba(44,228,223,0.07);
  backdrop-filter: blur(14px); -webkit-backdrop-filter: blur(14px);
}
.block-container {
  padding-top: 0 !important; padding-left: 3rem !important;
  padding-right: 3rem !important; max-width: 100% !important;
}

/* ── Subtle grid bg ── */
.stApp::before {
  content: ""; position: fixed; inset: 0;
  background-image:
    linear-gradient(rgba(44,228,223,0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(44,228,223,0.025) 1px, transparent 1px);
  background-size: 64px 64px;
  pointer-events: none; z-index: 0;
}

/* ── Layout shell ── */
.app-shell {
  position: relative; z-index: 2;
  max-width: 1460px; margin: 0 auto;
  padding: 32px 40px 60px;
}
@media (max-width: 900px) { .app-shell { padding: 18px 16px 40px; } }

/* ── Hide native Streamlit metric widget entirely ── */
[data-testid="stMetric"] { display: none !important; }

/* ── Hide native column gaps / decoration ── */
[data-testid="stHorizontalBlock"] > div { gap: 0 !important; }

/* ── Section headers ── */
.dash-section {
  display: flex; align-items: center; gap: 14px;
  margin: 36px 0 22px; padding-bottom: 16px;
  border-bottom: 1px solid var(--line);
}
.dash-section-bar {
  width: 3px; height: 24px; border-radius: 99px;
  background: linear-gradient(180deg, var(--cyan) 0%, rgba(94,166,255,0.15) 100%);
  box-shadow: var(--glow-cyan);
  flex-shrink: 0;
}
.dash-section-label {
  font-family: 'Sora', sans-serif;
  font-size: 13px; font-weight: 700;
  letter-spacing: 0.10em; text-transform: uppercase;
  color: var(--muted); margin: 0;
}
.dash-section-title {
  font-family: 'Sora', sans-serif;
  font-size: 22px; font-weight: 700; letter-spacing: -0.04em;
  color: var(--text); margin: 0;
}

/* ── City selector override ── */
[data-testid="stSelectbox"] label {
  font-size: 10px !important; text-transform: uppercase !important;
  letter-spacing: 0.14em !important; color: var(--muted-2) !important;
  font-weight: 700 !important;
}
[data-testid="stSelectbox"] > div > div {
  background: var(--panel) !important;
  border: 1px solid var(--line) !important;
  border-radius: 12px !important; color: var(--text) !important;
  font-family: 'Sora', sans-serif !important; font-weight: 600 !important;
  box-shadow: none !important;
}
[data-testid="stSelectbox"] > div > div:focus-within {
  border-color: var(--line-strong) !important;
  box-shadow: 0 0 0 2px rgba(44,228,223,0.08) !important;
}

/* ── Custom metric cards ── */
.metric-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px; margin: 20px 0 0;
}
@media (max-width: 900px) { .metric-grid { grid-template-columns: repeat(2,1fr); } }

.metric-card {
  position: relative; overflow: hidden;
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 22px 24px 20px;
  box-shadow: var(--shadow);
  transition: border-color 0.25s, transform 0.25s;
  cursor: default;
}
.metric-card:hover {
  border-color: rgba(44,228,223,0.22);
  transform: translateY(-3px);
}
.metric-card::after {
  content: ""; position: absolute;
  left: 0; right: 0; bottom: 0; height: 2px;
  background: var(--accent-bar, linear-gradient(90deg,var(--cyan),var(--blue)));
  opacity: 0.9;
}
.metric-card-icon {
  font-size: 22px; margin-bottom: 14px;
  display: block; opacity: 0.85;
}
.metric-card-label {
  font-size: 10px; text-transform: uppercase;
  letter-spacing: 0.16em; font-weight: 700;
  color: var(--muted-2); margin-bottom: 6px;
}
.metric-card-value {
  font-family: 'Sora', sans-serif;
  font-size: 28px; font-weight: 800;
  letter-spacing: -0.05em; color: var(--text);
  line-height: 1;
}
.metric-card-sub {
  font-size: 11px; color: var(--muted-2);
  margin-top: 6px; font-weight: 400;
}

/* ── Chart panel ── */
.chart-panel {
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 22px;
  box-shadow: var(--shadow);
  padding: 20px 20px 10px;
  transition: border-color 0.2s;
}
.chart-panel:hover { border-color: rgba(44,228,223,0.15); }
.chart-panel-header {
  display: flex; align-items: center; gap: 10px; margin-bottom: 4px;
}
.chart-panel-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--dot-color, var(--cyan));
  box-shadow: 0 0 8px var(--dot-color, var(--cyan));
}
.chart-panel-title {
  font-family: 'Sora', sans-serif;
  font-size: 13px; font-weight: 700;
  letter-spacing: -0.01em; color: var(--text);
}
.chart-panel-sub {
  font-size: 11px; color: var(--muted-2);
  margin-left: auto; letter-spacing: 0.04em;
}

/* ── Map & table panels ── */
.map-wrap {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 22px; box-shadow: var(--shadow);
  overflow: hidden; padding: 14px;
}
.table-wrap {
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 22px; box-shadow: var(--shadow); padding: 14px;
}

/* ── Legend pills ── */
.legend-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px; }
.legend-pill {
  display:inline-flex; align-items:center; gap:8px;
  padding:7px 14px; border-radius:999px;
  background:rgba(11,28,48,0.9); border:1px solid var(--line);
  color:var(--muted) !important; font-size:12px; font-weight:600;
}
.legend-dot { width:8px; height:8px; border-radius:50%; }

/* ── Dataframe ── */
[data-testid="stDataFrame"] { background: transparent !important; border: 0 !important; }
[data-testid="stDataFrame"] [role="table"] {
  border-radius:14px !important; overflow:hidden !important;
  border:1px solid var(--line) !important;
}
[data-testid="stDataFrame"] thead tr th {
  background:#0d2035 !important; color:var(--muted-2) !important;
  font-size:10px !important; text-transform:uppercase !important;
  letter-spacing:0.12em !important; font-weight:700 !important;
}
[data-testid="stDataFrame"] tbody tr td {
  background:#080f1c !important; color:var(--text) !important;
  font-size:13px !important;
  border-bottom:1px solid rgba(120,200,255,0.05) !important;
}
[data-testid="stDataFrame"] tbody tr:hover td { background:#0e1f33 !important; }

/* ── Footer ── */
.footer-note {
  text-align:center; padding:40px 0 12px;
  color:var(--muted-2); font-size:11px;
  letter-spacing:0.20em; text-transform:uppercase;
}

.hero-spacer { height: 4px; }
h1,h2,h3,h4,h5,h6 {
  font-family:'Sora',sans-serif !important;
  color:var(--text) !important; letter-spacing:-0.04em;
}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# HERO
# =============================================================================
HERO_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8" />
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 100%; height: 100%;
    overflow: hidden;
    background: #06111f;
    font-family: 'DM Sans', sans-serif;
  }

  .hero {
    position: relative;
    width: 100%; height: 620px;
    overflow: hidden;
    background: #06111f;
  }

  /* Liquid Ether canvas fills the whole hero */
  #liquid-canvas {
    position: absolute;
    inset: 0;
    width: 100%; height: 100%;
    display: block;
  }

  /* Dark vignette so text stays readable */
  .overlay-left {
    position: absolute; inset: 0;
    background: linear-gradient(
      105deg,
      rgba(6,17,31,0.97) 0%,
      rgba(6,17,31,0.82) 30%,
      rgba(6,17,31,0.38) 58%,
      rgba(6,17,31,0.05) 100%
    );
    z-index: 2;
  }
  .overlay-bottom {
    position: absolute; left:0; right:0; bottom:0; height:200px;
    background: linear-gradient(to bottom, transparent, rgba(6,17,31,0.88) 70%, #06111f);
    z-index: 2;
  }
  .overlay-top {
    position: absolute; top:0; left:0; right:0; height:90px;
    background: linear-gradient(to bottom, rgba(6,17,31,0.70), transparent);
    z-index: 2;
  }

  .content {
    position: relative; z-index: 3;
    height: 100%;
    display: flex; align-items: center;
    padding: 80px 60px 0;
  }

  .content-inner { max-width: 660px; }

  /* Eyebrow pill */
  .eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 14px 6px 10px;
    border-radius: 999px;
    background: rgba(44,228,223,0.08);
    border: 1px solid rgba(44,228,223,0.22);
    color: #7de8e4;
    font-size: 10px; letter-spacing: 0.18em; text-transform: uppercase; font-weight: 600;
    margin-bottom: 28px;
    animation: fadeUp 0.7s ease both;
  }
  .eyebrow-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: #2ce4df;
    box-shadow: 0 0 10px rgba(44,228,223,0.9);
    animation: pulse 2.4s ease-in-out infinite;
  }
  @keyframes pulse {
    0%,100% { opacity:1; transform:scale(1); }
    50% { opacity:0.6; transform:scale(0.8); }
  }

  /* Brand row */
  .brand {
    display: flex; align-items: center; gap: 11px;
    margin-bottom: 22px;
    animation: fadeUp 0.7s 0.08s ease both;
  }
  .brand-name {
    font-family: 'Sora', sans-serif;
    font-size: 26px; font-weight: 700;
    letter-spacing: -0.06em;
    color: #edf7ff;
  }

  /* Hero headline */
  .title {
    font-family: 'Sora', sans-serif;
    font-size: clamp(48px, 6.8vw, 86px);
    line-height: 0.95;
    letter-spacing: -0.06em;
    color: #edf7ff;
    margin-bottom: 20px;
    animation: fadeUp 0.7s 0.14s ease both;
  }
  .title .accent {
    display: block;
    background: linear-gradient(90deg, #2ce4df 0%, #5ea6ff 60%, #b89eff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .title .plain { display: block; }

  /* Subheadline */
  .sub {
    max-width: 500px;
    font-size: 16px; line-height: 1.75; font-weight: 300;
    color: rgba(220,238,252,0.72);
    margin-bottom: 30px;
    animation: fadeUp 0.7s 0.22s ease both;
  }

  /* Stat chips */
  .chip-row {
    display: flex; gap: 10px; flex-wrap: wrap;
    animation: fadeUp 0.7s 0.30s ease both;
  }
  .chip {
    padding: 11px 18px; border-radius: 12px;
    background: rgba(8, 22, 38, 0.55);
    border: 1px solid rgba(110,196,255,0.13);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: border-color 0.2s, transform 0.2s;
  }
  .chip:hover {
    border-color: rgba(44,228,223,0.30);
    transform: translateY(-2px);
  }
  .chip-label {
    font-size: 9px; text-transform: uppercase; letter-spacing: 0.16em;
    color: rgba(140,170,196,0.60); font-weight: 600; margin-bottom: 3px;
  }
  .chip-value {
    font-family: 'Sora', sans-serif;
    font-size: 14px; font-weight: 700;
    color: #edf7ff; letter-spacing: -0.03em;
  }
  .chip-value.teal {
    background: linear-gradient(90deg, #2ce4df, #5ea6ff);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(18px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  @media (max-width: 860px) {
    .hero { height: 560px; }
    .content { padding: 0 22px; }
    .sub { font-size: 14px; }
  }
</style>
</head>
<body>
<div class="hero" id="hero">

  <!-- Liquid Ether fluid simulation canvas -->
  <canvas id="liquid-canvas"></canvas>
  <div class="overlay-left"></div>
  <div class="overlay-top"></div>
  <div class="overlay-bottom"></div>

  <div class="content">
    <div class="content-inner">
      <div class="eyebrow"><span class="eyebrow-dot"></span>AI Real Estate Intelligence</div>

      <div class="brand">
        <svg width="38" height="38" viewBox="0 0 44 44" fill="none" xmlns="http://www.w3.org/2000/svg">
          <circle cx="22" cy="22" r="17" stroke="#2CE4DF" stroke-width="1.3" stroke-dasharray="3 3.5" opacity="0.5"/>
          <circle cx="36" cy="22" r="2.7" fill="#2CE4DF" opacity="0.9"/>
          <rect x="17" y="11" width="6" height="18" rx="1.4" fill="#E7C65A"/>
          <rect x="11" y="16" width="4.8" height="13" rx="1.4" fill="#2CE4DF" opacity="0.85"/>
          <rect x="25" y="14" width="4.8" height="15" rx="1.4" fill="#5EA6FF" opacity="0.8"/>
          <line x1="8" y1="29" x2="31" y2="29" stroke="#2CE4DF" stroke-width="1.0" opacity="0.3"/>
          <circle cx="22" cy="6.5" r="1.8" fill="#5EA6FF" opacity="0.7"/>
          <circle cx="6.5" cy="22" r="2.0" fill="#E7C65A" opacity="0.8"/>
        </svg>
        <div class="brand-name">mollecul</div>
      </div>

      <div class="title">
        <span class="plain">Objective.</span>
        <span class="accent">Data-Driven.</span>
        <span class="plain">Transparent.</span>
      </div>

      <div class="sub">
        Mollecul surfaces high-potential listings with cleaner market signals,
        stronger visual comparison, and faster insight into where opportunity
        actually lives.
      </div>

      <div class="chip-row">
        <div class="chip">
          <div class="chip-label">Markets Covered</div>
          <div class="chip-value teal">DFW Metro</div>
        </div>
        <div class="chip">
          <div class="chip-label">Data Source</div>
          <div class="chip-value">Live MLS</div>
        </div>
        <div class="chip">
          <div class="chip-label">Score Layer</div>
          <div class="chip-value">AI-Powered</div>
        </div>
      </div>
    </div>
  </div>
</div>

<script type="module">
// ── Liquid Ether fluid simulation (Three.js via CDN) ──────────────────────────
import * as THREE from 'https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.module.js';

const COLORS = ['#0a0f1e', '#1a3a6e', '#2ce4df', '#5ea6ff'];
const container = document.getElementById('hero');

// ── palette texture ──
function makePaletteTex(stops) {
  const data = new Uint8Array(stops.length * 4);
  stops.forEach((hex, i) => {
    const c = new THREE.Color(hex);
    data[i*4]   = Math.round(c.r*255);
    data[i*4+1] = Math.round(c.g*255);
    data[i*4+2] = Math.round(c.b*255);
    data[i*4+3] = 255;
  });
  const t = new THREE.DataTexture(data, stops.length, 1, THREE.RGBAFormat);
  t.magFilter = t.minFilter = THREE.LinearFilter;
  t.wrapS = t.wrapT = THREE.ClampToEdgeWrapping;
  t.generateMipmaps = false;
  t.needsUpdate = true;
  return t;
}

const paletteTex = makePaletteTex(COLORS);

// ── renderer ──
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('liquid-canvas'), antialias:false, alpha:false });
renderer.autoClear = false;
renderer.setClearColor(0x06111f, 1);
renderer.setPixelRatio(Math.min(window.devicePixelRatio||1, 2));

let W = container.offsetWidth, H = container.offsetHeight;
renderer.setSize(W, H);

const clock = new THREE.Clock(); clock.start();

// ── FBO helpers ──
function makeFBO(w, h) {
  const type = /iPad|iPhone|iPod/.test(navigator.userAgent) ? THREE.HalfFloatType : THREE.FloatType;
  return new THREE.WebGLRenderTarget(w, h, {
    type, depthBuffer:false, stencilBuffer:false,
    minFilter:THREE.LinearFilter, magFilter:THREE.LinearFilter,
    wrapS:THREE.ClampToEdgeWrapping, wrapT:THREE.ClampToEdgeWrapping
  });
}

const RES = 0.45;
let fw = Math.max(1, Math.round(W*RES)), fh = Math.max(1, Math.round(H*RES));
const fbos = {};
['v0','v1','vv0','vv1','div','p0','p1'].forEach(k => fbos[k] = makeFBO(fw, fh));

const cellScale = new THREE.Vector2(1/fw, 1/fh);
const fboSize   = new THREE.Vector2(fw, fh);

// ── shader source ──
const FACE_VERT = `
attribute vec3 position;
uniform vec2 boundarySpace;
varying vec2 uv;
precision highp float;
void main(){
  vec2 scale = 1.0 - boundarySpace*2.0;
  vec3 pos = position; pos.xy *= scale;
  uv = vec2(0.5)+pos.xy*0.5;
  gl_Position = vec4(pos,1.0);
}`;

const ADV_FRAG = `
precision highp float;
uniform sampler2D velocity; uniform float dt; uniform bool isBFECC;
uniform vec2 fboSize; uniform vec2 px;
varying vec2 uv;
void main(){
  vec2 ratio = max(fboSize.x,fboSize.y)/fboSize;
  if(!isBFECC){
    vec2 v=texture2D(velocity,uv).xy;
    gl_FragColor=vec4(texture2D(velocity,uv-v*dt*ratio).xy,0,0);
  } else {
    vec2 v0=texture2D(velocity,uv).xy;
    vec2 s0=uv-v0*dt*ratio;
    vec2 v1=texture2D(velocity,s0).xy;
    vec2 s2=s0+v1*dt*ratio;
    vec2 err=(s2-uv)/2.0;
    vec2 s3=uv-err;
    vec2 v2=texture2D(velocity,s3).xy;
    gl_FragColor=vec4(texture2D(velocity,s3-v2*dt*ratio).xy,0,0);
  }
}`;

const DIV_FRAG = `
precision highp float;
uniform sampler2D velocity; uniform vec2 px; uniform float dt;
varying vec2 uv;
void main(){
  float x0=texture2D(velocity,uv-vec2(px.x,0)).x;
  float x1=texture2D(velocity,uv+vec2(px.x,0)).x;
  float y0=texture2D(velocity,uv-vec2(0,px.y)).y;
  float y1=texture2D(velocity,uv+vec2(0,px.y)).y;
  gl_FragColor=vec4((x1-x0+y1-y0)/2.0/dt);
}`;

const PSN_FRAG = `
precision highp float;
uniform sampler2D pressure; uniform sampler2D divergence; uniform vec2 px;
varying vec2 uv;
void main(){
  float p0=texture2D(pressure,uv+vec2(px.x*2.0,0)).r;
  float p1=texture2D(pressure,uv-vec2(px.x*2.0,0)).r;
  float p2=texture2D(pressure,uv+vec2(0,px.y*2.0)).r;
  float p3=texture2D(pressure,uv-vec2(0,px.y*2.0)).r;
  float d=texture2D(divergence,uv).r;
  gl_FragColor=vec4((p0+p1+p2+p3)/4.0-d);
}`;

const PRESS_FRAG = `
precision highp float;
uniform sampler2D pressure; uniform sampler2D velocity; uniform vec2 px; uniform float dt;
varying vec2 uv;
void main(){
  float p0=texture2D(pressure,uv+vec2(px.x,0)).r;
  float p1=texture2D(pressure,uv-vec2(px.x,0)).r;
  float p2=texture2D(pressure,uv+vec2(0,px.y)).r;
  float p3=texture2D(pressure,uv-vec2(0,px.y)).r;
  vec2 v=texture2D(velocity,uv).xy;
  gl_FragColor=vec4(v-vec2(p0-p1,p2-p3)*0.5*dt,0,1);
}`;

const FORCE_VERT = `
precision highp float;
attribute vec3 position; attribute vec2 uv;
uniform vec2 center; uniform vec2 scale; uniform vec2 px;
varying vec2 vUv;
void main(){
  vec2 pos=position.xy*scale*2.0*px+center;
  vUv=uv; gl_Position=vec4(pos,0,1);
}`;

const FORCE_FRAG = `
precision highp float;
uniform vec2 force;
varying vec2 vUv;
void main(){
  vec2 c=(vUv-0.5)*2.0; float d=1.0-min(length(c),1.0); d*=d;
  gl_FragColor=vec4(force*d,0,1);
}`;

const COLOR_FRAG = `
precision highp float;
uniform sampler2D velocity; uniform sampler2D palette;
varying vec2 uv;
void main(){
  vec2 v=texture2D(velocity,uv).xy;
  float l=clamp(length(v),0.0,1.0);
  gl_FragColor=vec4(texture2D(palette,vec2(l,0.5)).rgb,1.0);
}`;

// ── pass factory ──
const cam = new THREE.Camera();
function makePass(vSrc, fSrc, uniforms, output) {
  const sc = new THREE.Scene();
  const mat = new THREE.RawShaderMaterial({ vertexShader:vSrc, fragmentShader:fSrc, uniforms });
  sc.add(new THREE.Mesh(new THREE.PlaneGeometry(2,2), mat));
  return { sc, mat, uniforms,
    run(tgt){ renderer.setRenderTarget(tgt||null); renderer.render(sc,cam); renderer.setRenderTarget(null); }
  };
}

const BSpace = new THREE.Vector2(1/fw, 1/fh);

const advPass = makePass(FACE_VERT, ADV_FRAG, {
  boundarySpace:{value:BSpace}, px:{value:cellScale}, fboSize:{value:fboSize},
  velocity:{value:fbos.v0.texture}, dt:{value:0.014}, isBFECC:{value:true}
});

const divPass = makePass(FACE_VERT, DIV_FRAG, {
  boundarySpace:{value:BSpace}, velocity:{value:fbos.v1.texture},
  px:{value:cellScale}, dt:{value:0.014}
});

const psnPass = makePass(FACE_VERT, PSN_FRAG, {
  boundarySpace:{value:BSpace}, px:{value:cellScale},
  pressure:{value:fbos.p0.texture}, divergence:{value:fbos.div.texture}
});

const pressPass = makePass(FACE_VERT, PRESS_FRAG, {
  boundarySpace:{value:BSpace}, px:{value:cellScale}, dt:{value:0.014},
  pressure:{value:fbos.p0.texture}, velocity:{value:fbos.v1.texture}
});

const forcePass = makePass(FORCE_VERT, FORCE_FRAG, {
  px:{value:cellScale}, force:{value:new THREE.Vector2()},
  center:{value:new THREE.Vector2()}, scale:{value:new THREE.Vector2(100,100)}
});
forcePass.mat.blending = THREE.AdditiveBlending;
forcePass.mat.depthWrite = false;

const colorPass = makePass(FACE_VERT, COLOR_FRAG, {
  boundarySpace:{value:new THREE.Vector2()},
  velocity:{value:fbos.v0.texture}, palette:{value:paletteTex}
});

// ── mouse / auto-drive ──
const mouse    = new THREE.Vector2(0,0);
const mouseOld = new THREE.Vector2(0,0);
const mouseDiff= new THREE.Vector2(0,0);
let autoT = 0;
const autoTarget = new THREE.Vector2();
const autoCurrent = new THREE.Vector2();
function pickTarget(){ autoTarget.set((Math.random()*2-1)*0.8,(Math.random()*2-1)*0.8); }
pickTarget();

container.addEventListener('mousemove', e => {
  const r = container.getBoundingClientRect();
  mouse.set((e.clientX-r.left)/r.width*2-1, -((e.clientY-r.top)/r.height*2-1));
});
container.addEventListener('touchmove', e => {
  const t = e.touches[0], r = container.getBoundingClientRect();
  mouse.set((t.clientX-r.left)/r.width*2-1, -((t.clientY-r.top)/r.height*2-1));
}, {passive:true});

// ── resize ──
function onResize() {
  W = container.offsetWidth; H = container.offsetHeight;
  renderer.setSize(W, H);
  fw = Math.max(1, Math.round(W*RES)); fh = Math.max(1, Math.round(H*RES));
  Object.values(fbos).forEach(f => f.setSize(fw, fh));
  cellScale.set(1/fw, 1/fh); fboSize.set(fw, fh);
}
window.addEventListener('resize', onResize);

// ── render loop ──
const DT = 0.014; const ITER = 24;
let frame = 0;

function render() {
  requestAnimationFrame(render);
  const dt = Math.min(clock.getDelta(), 0.04);

  // auto-drive cursor
  autoT += dt * 0.38;
  const d = new THREE.Vector2().subVectors(autoTarget, autoCurrent);
  if (d.length() < 0.02) pickTarget();
  autoCurrent.addScaledVector(d.normalize(), Math.min(d.length(), 0.38*dt));
  mouse.copy(autoCurrent);

  mouseDiff.subVectors(mouse, mouseOld);
  mouseOld.copy(mouse);

  // advection
  advPass.uniforms.velocity.value = fbos.v0.texture;
  advPass.uniforms.dt.value = DT;
  advPass.run(fbos.v1);

  // force
  const fScale = 80;
  forcePass.uniforms.force.value.set(mouseDiff.x*fScale/2, mouseDiff.y*fScale/2);
  const cx = Math.max(-0.98, Math.min(0.98, mouse.x));
  const cy = Math.max(-0.98, Math.min(0.98, mouse.y));
  forcePass.uniforms.center.value.set(cx, cy);
  forcePass.run(fbos.v1);

  // divergence
  divPass.uniforms.velocity.value = fbos.v1.texture;
  divPass.run(fbos.div);

  // poisson pressure
  for (let i=0; i<ITER; i++) {
    const pIn  = i%2===0 ? fbos.p0 : fbos.p1;
    const pOut = i%2===0 ? fbos.p1 : fbos.p0;
    psnPass.uniforms.pressure.value = pIn.texture;
    psnPass.run(pOut);
  }
  const finalP = ITER%2===0 ? fbos.p1 : fbos.p0;

  // pressure projection
  pressPass.uniforms.velocity.value = fbos.v1.texture;
  pressPass.uniforms.pressure.value = finalP.texture;
  pressPass.run(fbos.v0);

  // color output
  colorPass.uniforms.velocity.value = fbos.v0.texture;
  colorPass.run(null);

  frame++;
}
render();
</script>
</body>
</html>
"""

components.html(HERO_HTML, height=610, scrolling=False)

# =============================================================================
# HELPERS
# =============================================================================
def line_chart(trend_data):
    grid = "rgba(120,200,255,0.07)"
    font = "#7aa0bc"

    x_vals = trend_data["Period"].tolist()
    y_vals = trend_data["Value"].tolist()
    n = len(x_vals)

    if n == 0:
        return go.Figure()

    y_max = max(y_vals) * 1.15 if y_vals else 1

    fig = go.Figure(
        data=[go.Scatter(
            x=x_vals, y=y_vals,
            mode="lines+markers",
            line=dict(color="#2ce4df", width=3, shape="spline", smoothing=0.75),
            marker=dict(size=7, color="#2ce4df", line=dict(color="#06111f", width=1.5)),
            fill="tozeroy", fillcolor="rgba(44,228,223,0.07)",
            hovertemplate="<b>%{x}</b><br>Avg Price: $%{y:,.0f}<extra></extra>",
            showlegend=False,
        )]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=10, r=10, t=8, b=20),
        font=dict(family="DM Sans, sans-serif", color=font, size=11),
        hoverlabel=dict(
            bgcolor="#0b1c30", bordercolor="rgba(44,228,223,0.22)",
            font=dict(color="#ecf6ff", family="DM Sans, sans-serif", size=12),
        ),
        xaxis=dict(
            title="Year Built", gridcolor=grid, zeroline=False,
            tickfont=dict(color=font), title_font=dict(color=font),
            range=[x_vals[0], x_vals[-1]] if n > 1 else None,
        ),
        yaxis=dict(
            title="Average Price ($)", gridcolor=grid, zeroline=False,
            tickprefix="$", tickformat=",.0f",
            tickfont=dict(color=font), title_font=dict(color=font),
            range=[0, y_max],
        ),
    )
    return fig


def bar_chart(roi_data):
    grid = "rgba(120,200,255,0.07)"
    font = "#7aa0bc"

    roi_sorted = roi_data.sort_values("ROI_PROXY", ascending=True).copy()
    vals = roi_sorted["ROI_PROXY"].values

    if len(vals) == 0:
        return go.Figure()

    norm = (vals - vals.min()) / max(vals.max() - vals.min(), 1e-9)
    colors = []
    for nv in norm:
        r = int(44  + nv * 185)
        g = int(228 - nv * 30)
        b = int(223 - nv * 130)
        colors.append(f"rgb({r},{g},{b})")

    fig = go.Figure(
        data=[go.Bar(
            x=roi_sorted["ROI_PROXY"],
            y=roi_sorted["PROPERTY TYPE"],
            orientation="h",
            marker=dict(
                color=colors,
                line=dict(color="rgba(0,0,0,0)", width=0),
            ),
            text=[f"${v:,.0f}" for v in roi_sorted["ROI_PROXY"]],
            textposition="outside",
            textfont=dict(color=font, size=11),
            hovertemplate="<b>%{y}</b><br>Avg $/sq ft: $%{x:,.0f}<extra></extra>",
        )]
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
        margin=dict(l=10, r=65, t=8, b=20),
        font=dict(family="DM Sans, sans-serif", color=font, size=11),
        hoverlabel=dict(
            bgcolor="#0b1c30", bordercolor="rgba(44,228,223,0.22)",
            font=dict(color="#ecf6ff", family="DM Sans, sans-serif", size=12),
        ),
        bargap=0.28,
        xaxis=dict(
            title="Avg $ / Sq Ft", gridcolor=grid, zeroline=False,
            tickprefix="$", tickformat=",.0f",
            tickfont=dict(color=font), title_font=dict(color=font),
            range=[0, vals.max() * 1.22],
        ),
        yaxis=dict(
            gridcolor="rgba(0,0,0,0)", zeroline=False, tickfont=dict(color=font),
        ),
    )
    return fig

# =============================================================================
# APP BODY
# =============================================================================
st.markdown("<div class='app-shell'>", unsafe_allow_html=True)

available_cities = sorted(df["CITY"].dropna().unique().tolist())
default_city = "Dallas" if "Dallas" in available_cities else available_cities[0]

# ── City selector ──────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-section'>
  <div class='dash-section-bar'></div>
  <div>
    <div class='dash-section-label'>Dashboard</div>
    <div class='dash-section-title'>Real Estate Intelligence Suite</div>
  </div>
</div>
""", unsafe_allow_html=True)

c1, c2, c3 = st.columns([1.1, 1, 2.2])
with c1:
    target_city = st.selectbox(
        "Market Location",
        available_cities,
        index=available_cities.index(default_city)
    )

city_df = df[df["CITY"].str.lower() == target_city.lower()].copy()

valuation_model = get_valuation_model()

predicted_value = np.nan
selected_listing_address = "N/A"
selected_listing_price = np.nan

if not city_df.empty:
    try:
        preview_idx = city_df["investment_score"].idxmax()
        preview_row = city_df.loc[preview_idx]

        selected_listing_address = preview_row.get("ADDRESS", "N/A")
        selected_listing_price = preview_row.get("PRICE", np.nan)

        model_input_df = map_dashboard_row_to_model_input(preview_row)
        predicted_value = float(predict(valuation_model, model_input_df)[0])
    except Exception:
        predicted_value = np.nan

avg_price   = city_df["PRICE"].mean()           if "PRICE"               in city_df.columns else np.nan
avg_ppsqft  = city_df["PRICE_PER_SQFT_FINAL"].mean() if "PRICE_PER_SQFT_FINAL" in city_df.columns else np.nan
avg_dom     = city_df["DAYS ON MARKET"].mean()  if "DAYS ON MARKET"      in city_df.columns else np.nan
listing_count = len(city_df)

# ── Custom metric cards (hide native st.metric) ────────────────────────────────
avg_price_fmt  = f"${avg_price:,.0f}"   if pd.notna(avg_price)  else "N/A"
avg_ppsqft_fmt = f"${avg_ppsqft:,.0f}" if pd.notna(avg_ppsqft) else "N/A"
avg_dom_fmt    = f"{avg_dom:.1f} days"  if pd.notna(avg_dom)    else "N/A"
listing_fmt    = f"{listing_count:,}"
predicted_value_fmt = f"${predicted_value:,.0f}" if pd.notna(predicted_value) else "N/A"
model_gap = predicted_value - selected_listing_price if pd.notna(predicted_value) and pd.notna(selected_listing_price) else np.nan
model_gap_fmt = f"${model_gap:,.0f}" if pd.notna(model_gap) else "N/A"

st.markdown(f"""
<div class="metric-grid">
  <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#2ce4df,#5ea6ff)">
    <span class="metric-card-icon">💰</span>
    <div class="metric-card-label">Average Price</div>
    <div class="metric-card-value">{avg_price_fmt}</div>
    <div class="metric-card-sub">{target_city} market average</div>
  </div>
  <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#5ea6ff,#b89eff)">
    <span class="metric-card-icon">📐</span>
    <div class="metric-card-label">Avg $ / Sq Ft</div>
    <div class="metric-card-value">{avg_ppsqft_fmt}</div>
    <div class="metric-card-sub">price density index</div>
  </div>
  <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#e7c65a,#ff9f7a)">
    <span class="metric-card-icon">⏱️</span>
    <div class="metric-card-label">Avg Days on Market</div>
    <div class="metric-card-value">{avg_dom_fmt}</div>
    <div class="metric-card-sub">listing velocity signal</div>
  </div>
  <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#b89eff,#2ce4df)">
    <span class="metric-card-icon">🏘️</span>
    <div class="metric-card-label">Active Listings</div>
    <div class="metric-card-value">{listing_fmt}</div>
    <div class="metric-card-sub">properties in dataset</div>
  </div>

  <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#b89eff,#2ce4df)">
    <span class="metric-card-icon">🧠</span>
    <div class="metric-card-label">Model Estimate</div>
    <div class="metric-card-value">{predicted_value_fmt}</div>
    <div class="metric-card-sub">Top-ranked listing preview · gap {model_gap_fmt}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# keep native metrics hidden so city selector still works (they exist in DOM but are hidden)
_cols = st.columns(4)
with _cols[0]: st.metric("Average Price", avg_price_fmt)
with _cols[1]: st.metric("Avg $ / Sq Ft", avg_ppsqft_fmt)
with _cols[2]: st.metric("Avg Days on Market", avg_dom_fmt)
with _cols[3]: st.metric("Listings", listing_fmt)

# ── Forecast preview ───────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-section'>
  <div class='dash-section-bar'></div>
  <div>
    <div class='dash-section-label'>Forecasting</div>
    <div class='dash-section-title'>Forecast Model Preview</div>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("### Forecast Assumptions")

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    mortgage_rate_30yr = st.slider("Mortgage Rate (30yr %)", 3.0, 10.0, 6.75)

with col2:
    fed_funds_rate = st.slider("Fed Funds Rate %", 0.0, 10.0, 5.25)

with col3:
    treasury_10yr = st.slider("10Y Treasury %", 0.0, 10.0, 4.25)

with col4:
    treasury_2yr = st.slider("2Y Treasury %", 0.0, 10.0, 4.00)

with col5:
    case_shiller_dallas = st.slider("Case-Shiller Index", 100.0, 500.0, 320.0)

with col6:
    vix = st.slider("VIX", 10.0, 50.0, 18.0)

forecast_output = np.nan
forecast_ready = False
forecast_error = ""

implied_spread = treasury_10yr - treasury_2yr
forecast_ready = False
forecast_output = np.nan
forecast_error = ""

try:
    forecast_input_df = build_forecast_input({
        "mortgage_rate_30yr": mortgage_rate_30yr,
        "fed_funds_rate": fed_funds_rate,
        "treasury_10yr": treasury_10yr,
        "treasury_2yr": treasury_2yr,
        "case_shiller_dallas": case_shiller_dallas,
        "vix": vix
    })

    forecast_output = forecast_predict(forecast_input_df)
    forecast_ready = True

except Exception as e:
    forecast_error = str(e)
    forecast_ready = False
if forecast_ready:
    forecast_output_fmt = f"{forecast_output:,.4f}" if pd.notna(forecast_output) else "N/A"
    implied_spread = treasury_10yr - treasury_2yr
    if forecast_ready:
        st.success(f"Forecast output: {forecast_output:,.4f}")
    else:
        st.warning(f"Forecast model not ready: {forecast_error or 'Unknown error'}")
    st.markdown(f"""
    <div class="metric-grid">
      <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#2ce4df,#5ea6ff)">
        <span class="metric-card-icon">📈</span>
        <div class="metric-card-label">Forecast Output</div>
        <div class="metric-card-value">{forecast_output_fmt}</div>
        <div class="metric-card-sub">Model output based on current macro assumptions</div>
      </div>
      <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#5ea6ff,#b89eff)">
        <span class="metric-card-icon">🏦</span>
        <div class="metric-card-label">Mortgage Rate</div>
        <div class="metric-card-value">{mortgage_rate_30yr:.2f}%</div>
        <div class="metric-card-sub">30-year baseline assumption</div>
      </div>
      <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#e7c65a,#ff9f7a)">
        <span class="metric-card-icon">📊</span>
        <div class="metric-card-label">Case-Shiller Dallas</div>
        <div class="metric-card-value">{case_shiller_dallas:,.1f}</div>
        <div class="metric-card-sub">Home-price index input</div>
      </div>
      <div class="metric-card" style="--accent-bar: linear-gradient(90deg,#b89eff,#2ce4df)">
        <span class="metric-card-icon">🧮</span>
        <div class="metric-card-label">Yield Spread</div>
        <div class="metric-card-value">{implied_spread:.2f}</div>
        <div class="metric-card-sub">10Y minus 2Y treasury rate</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.caption("Forecast model uses the saved forecast .joblib, scaler, and feature-cols JSON. The displayed value is the raw model output for the chosen assumptions.")
else:
    st.info(f"Forecast model preview is unavailable right now: {forecast_error}")

# ── Charts ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-section'>
  <div class='dash-section-bar'></div>
  <div>
    <div class='dash-section-label'>Analytics</div>
    <div class='dash-section-title'>Market Performance</div>
  </div>
</div>
""", unsafe_allow_html=True)

trend_data = get_price_trends(df, target_city)
roi_data   = get_roi_by_type(df, target_city)

def _animated_line_html(trend_data):
    x_vals = trend_data["Period"].tolist()
    y_vals = trend_data["Value"].tolist()
    import json
    x_json = json.dumps(x_vals)
    y_json = json.dumps([float(v) for v in y_vals])
    return f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>* {{margin:0;padding:0;box-sizing:border-box;}} body {{background:transparent;font-family:'DM Sans',sans-serif;}}</style>
</head><body>
<div style="background:#0b1c30;border:1px solid rgba(110,196,255,0.09);border-radius:18px;padding:18px 18px 10px;height:420px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <div style="width:8px;height:8px;border-radius:50%;background:#2ce4df;box-shadow:0 0 8px rgba(44,228,223,0.7);flex-shrink:0;"></div>
    <span style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#ecf6ff;letter-spacing:-0.01em;">Average Price Trend</span>
    <span style="margin-left:auto;font-size:10px;color:#567592;letter-spacing:0.10em;text-transform:uppercase;font-weight:600;white-space:nowrap;">by year built</span>
  </div>
  <div id="linechart" style="height:370px;"></div>
</div>
<script>
(function() {{
  var xAll = {x_json};
  var yAll = {y_json};
  var n = xAll.length;
  var yMax = Math.max.apply(null, yAll) * 1.18;
  var grid = 'rgba(120,200,255,0.06)';
  var layout = {{
    paper_bgcolor:'rgba(0,0,0,0)', plot_bgcolor:'rgba(0,0,0,0)',
    height:370, margin:{{l:70,r:16,t:4,b:44}},
    font:{{family:'DM Sans, sans-serif',color:'#8baec8',size:11}},
    xaxis:{{
      title:{{text:'Year Built',font:{{family:'DM Sans, sans-serif',size:11,color:'#567592'}},standoff:10}},
      gridcolor:grid, zeroline:false,
      tickfont:{{family:'DM Sans, sans-serif',color:'#567592',size:10}},
      showline:false
    }},
    yaxis:{{
      title:{{text:'Avg Price ($)',font:{{family:'DM Sans, sans-serif',size:11,color:'#567592'}},standoff:8}},
      gridcolor:grid, zeroline:false,
      tickprefix:'$', tickformat:',.0f',
      range:[0,yMax],
      tickfont:{{family:'DM Sans, sans-serif',color:'#567592',size:10}},
      showline:false
    }},
    hoverlabel:{{bgcolor:'#0b1c30',bordercolor:'rgba(44,228,223,0.22)',font:{{color:'#ecf6ff',family:'DM Sans, sans-serif',size:12}}}},
    showlegend:false
  }};
  var config = {{displayModeBar:false,responsive:true}};
  Plotly.newPlot('linechart', [{{x:[],y:[],mode:'lines+markers',
    line:{{color:'#2ce4df',width:3,shape:'spline',smoothing:0.75}},
    marker:{{size:7,color:'#2ce4df',line:{{color:'#06111f',width:1.5}}}},
    fill:'tozeroy',fillcolor:'rgba(44,228,223,0.07)',
    hovertemplate:'<b>%{{x}}</b><br>Avg Price: $%{{y:,.0f}}<extra></extra>'
  }}], layout, config);
  var step = 0;
  var delay = Math.max(18, Math.round(900 / n));
  function drawNext() {{
    step++;
    Plotly.restyle('linechart', {{x:[xAll.slice(0,step)], y:[yAll.slice(0,step)]}}, [0]);
    if (step < n) setTimeout(drawNext, delay);
  }}
  setTimeout(drawNext, 350);
}})();
</script>
</body></html>"""

def _animated_bar_html(roi_data):
    import json
    roi_sorted = roi_data.sort_values("ROI_PROXY", ascending=True).copy()
    labels = roi_sorted["PROPERTY TYPE"].tolist()
    vals   = roi_sorted["ROI_PROXY"].tolist()
    vmin, vmax = min(vals), max(vals)
    colors = []
    for v in vals:
        nv = (v - vmin) / max(vmax - vmin, 1e-9)
        r = int(44  + nv * 185)
        g = int(228 - nv * 30)
        b = int(223 - nv * 130)
        colors.append("rgb(" + str(r) + "," + str(g) + "," + str(b) + ")")
    labels_json = json.dumps(labels)
    vals_json   = json.dumps([float(v) for v in vals])
    colors_json = json.dumps(colors)
    n_bars = len(vals)
    chart_height = max(340, n_bars * 58 + 80)
    total_height = chart_height + 60
    return f"""<!DOCTYPE html><html><head>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<style>
  * {{margin:0;padding:0;box-sizing:border-box;}}
  body {{background:transparent;font-family:'DM Sans',sans-serif;}}
</style>
</head><body>
<div style="background:#0b1c30;border:1px solid rgba(110,196,255,0.09);border-radius:18px;padding:18px 18px 10px;height:{total_height}px;">
  <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
    <div style="width:8px;height:8px;border-radius:50%;background:#e7c65a;box-shadow:0 0 8px rgba(231,198,90,0.6);flex-shrink:0;"></div>
    <span style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#ecf6ff;letter-spacing:-0.01em;">Price Strength by Property Type</span>
    <span style="margin-left:auto;font-size:10px;color:#567592;letter-spacing:0.10em;text-transform:uppercase;font-weight:600;white-space:nowrap;">avg $/sq ft</span>
  </div>
  <div id="barchart" style="height:{chart_height}px;"></div>
</div>
<script>
(function() {{
  var labels  = {labels_json};
  var valsAll = {vals_json};
  var colors  = {colors_json};
  var xMax    = Math.max.apply(null, valsAll) * 1.18;
  var grid    = 'rgba(120,200,255,0.06)';

  // Build custom y-axis tick labels as HTML annotations instead
  // so we can fully control font — we hide Plotly y ticks and use annotations
  var annotations = labels.map(function(label, i) {{
    return {{
      x: 0, y: label, xref: 'x', yref: 'y',
      text: label,
      showarrow: false,
      xanchor: 'right',
      xshift: -10,
      font: {{family: 'DM Sans, sans-serif', size: 11, color: '#8baec8'}},
    }};
  }});

  // Value labels on right side
  var valueAnnotations = valsAll.map(function(v, i) {{
    return {{
      x: v, y: labels[i], xref: 'x', yref: 'y',
      text: '<b>$' + Math.round(v).toLocaleString() + '</b>',
      showarrow: false,
      xanchor: 'left',
      xshift: 8,
      font: {{family: 'Sora, sans-serif', size: 11, color: '#ecf6ff'}},
    }};
  }});

  var layout = {{
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor:  'rgba(0,0,0,0)',
    height: {chart_height},
    margin: {{l: 170, r: 80, t: 4, b: 44}},
    bargap: 0.32,
    font: {{family: 'DM Sans, sans-serif', color: '#8baec8', size: 11}},
    annotations: valueAnnotations,
    xaxis: {{
      title: {{text: 'Avg $ / Sq Ft', font: {{family: 'DM Sans, sans-serif', size: 11, color: '#567592'}}, standoff: 10}},
      gridcolor: grid, zeroline: false,
      tickprefix: '$', tickformat: ',.0f',
      range: [0, xMax],
      tickfont: {{family: 'DM Sans, sans-serif', color: '#567592', size: 10}},
      showline: false
    }},
    yaxis: {{
      gridcolor: 'rgba(0,0,0,0)', zeroline: false,
      tickfont: {{family: 'DM Sans, sans-serif', color: '#8baec8', size: 11}},
      automargin: false,
      showticklabels: true
    }},
    hoverlabel: {{
      bgcolor: '#0b1c30', bordercolor: 'rgba(44,228,223,0.22)',
      font: {{color: '#ecf6ff', family: 'DM Sans, sans-serif', size: 12}}
    }},
    showlegend: false
  }};

  var config = {{displayModeBar: false, responsive: true}};
  var zeros  = valsAll.map(function() {{ return 0; }});

  Plotly.newPlot('barchart', [{{
    x: zeros, y: labels, orientation: 'h', type: 'bar',
    marker: {{
      color: colors,
      opacity: 0.88,
      line: {{color: 'rgba(0,0,0,0)', width: 0}}
    }},
    text: [],
    hovertemplate: '<b>%{{y}}</b><br>Avg $/sq ft: $%{{x:,.0f}}<extra></extra>'
  }}], layout, config);

  var STEPS = 36;
  var step  = 0;
  function ease(t) {{ return t * t * (3 - 2 * t); }}
  function animate() {{
    step++;
    var t = ease(step / STEPS);
    var current = valsAll.map(function(v) {{ return v * t; }});
    // Update bar x and annotation x positions together
    var updatedAnnotations = valueAnnotations.map(function(ann, i) {{
      return Object.assign({{}}, ann, {{x: current[i]}});
    }});
    Plotly.update('barchart',
      {{x: [current]}},
      {{annotations: updatedAnnotations}},
      [0]
    );
    if (step < STEPS) requestAnimationFrame(animate);
    else Plotly.update('barchart', {{x: [valsAll]}}, {{annotations: valueAnnotations}}, [0]);
  }}
  setTimeout(animate, 350);
}})();
</script>
</body></html>"""
left, right = st.columns(2, gap="medium")
with left:
    components.html(_animated_line_html(trend_data), height=428, scrolling=False)
with right:
    components.html(_animated_bar_html(roi_data), height=428, scrolling=False)



# ── Map ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='dash-section'>
  <div class='dash-section-bar'></div>
  <div>
    <div class='dash-section-label'>Explorer</div>
    <div class='dash-section-title'>Interactive Property Map</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='legend-row'>
  <div class='legend-pill'><span class='legend-dot' style='background:#2ce4df;'></span>Above median price</div>
  <div class='legend-pill'><span class='legend-dot' style='background:#e7c65a;'></span>Below median price</div>
</div>
""", unsafe_allow_html=True)

city_coords = {
    "Dallas": [32.7767, -96.7970], "Fort Worth": [32.7555, -97.3308],
    "Arlington": [32.7357, -97.1081], "Plano": [33.0198, -96.6989],
    "Frisco": [33.1507, -96.8236], "Irving": [32.8140, -96.9489],
    "Richardson": [32.9483, -96.7299], "Garland": [32.9126, -96.6389],
    "Grand Prairie": [32.7459, -96.9978], "Mansfield": [32.5632, -97.1417],
    "Austin": [30.2672, -97.7431], "Houston": [29.7604, -95.3698],
}
location = city_coords.get(target_city, [city_df["LATITUDE"].mean(), city_df["LONGITUDE"].mean()])

map_df = city_df.dropna(subset=["LATITUDE", "LONGITUDE", "PRICE"]).copy()
if len(map_df) > 400:
    map_df = map_df.sample(400, random_state=42)
price_med = map_df["PRICE"].median()

fmap = folium.Map(location=location, zoom_start=11, tiles="CartoDB dark_matter")

for _, row in map_df.iterrows():
    above = row["PRICE"] >= price_med
    mc = "#2ce4df" if above else "#e7c65a"
    parts = []
    if "ADDRESS" in row and pd.notna(row["ADDRESS"]):
        parts.append(f"<b style='color:#2ce4df'>{row['ADDRESS']}</b>")
    parts.append(f"Price: <b>${row['PRICE']:,.0f}</b>")
    if "BEDS"        in row and pd.notna(row["BEDS"]):        parts.append(f"Beds: {int(row['BEDS'])}")
    if "BATHS"       in row and pd.notna(row["BATHS"]):       parts.append(f"Baths: {row['BATHS']}")
    if "SQUARE FEET" in row and pd.notna(row["SQUARE FEET"]): parts.append(f"Sq Ft: {row['SQUARE FEET']:,.0f}")
    popup_html = (
        "<div style='font-family:DM Sans,sans-serif;font-size:13px;"
        "background:#0b1c30;color:#ecf6ff;padding:10px 14px;"
        "border-radius:10px;border:1px solid rgba(44,228,223,0.18);"
        "min-width:170px;line-height:1.7;'>"
        + "<br>".join(parts) + "</div>"
    )
    folium.CircleMarker(
        [row["LATITUDE"], row["LONGITUDE"]],
        radius=5.5,
        popup=folium.Popup(popup_html, max_width=260),
        color=mc, fill=True, fill_color=mc,
        fill_opacity=0.88 if above else 0.72, weight=1.4,
    ).add_to(fmap)

map_col, stat_col = st.columns([3, 1], gap="medium")

with map_col:
    st.markdown("<div class='map-wrap'>", unsafe_allow_html=True)
    st_folium(fmap, width="100%", height=400, key=f"map_{target_city}")
    st.markdown("</div>", unsafe_allow_html=True)

with stat_col:
    above_count = int((map_df["PRICE"] >= price_med).sum())
    below_count = int((map_df["PRICE"] < price_med).sum())
    avg_map_price = map_df["PRICE"].mean()
    max_price = map_df["PRICE"].max()
    min_price = map_df["PRICE"].min()
    st.markdown(f"""
<div style="display:flex;flex-direction:column;gap:12px;padding:4px 0;">
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:6px;">Mapped Listings</div>
    <div style="font-family:'Sora',sans-serif;font-size:26px;font-weight:800;color:#ecf6ff;letter-spacing:-0.04em;">{len(map_df):,}</div>
    <div style="font-size:11px;color:#567592;margin-top:2px;">properties plotted</div>
  </div>
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:10px;">Price Distribution</div>
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px;">
      <span style="width:8px;height:8px;border-radius:50%;background:#2ce4df;flex-shrink:0;box-shadow:0 0 6px #2ce4df;"></span>
      <span style="font-size:12px;color:#8baec8;">Above median</span>
      <span style="margin-left:auto;font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#ecf6ff;">{above_count}</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;">
      <span style="width:8px;height:8px;border-radius:50%;background:#e7c65a;flex-shrink:0;box-shadow:0 0 6px #e7c65a88;"></span>
      <span style="font-size:12px;color:#8baec8;">Below median</span>
      <span style="margin-left:auto;font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#ecf6ff;">{below_count}</span>
    </div>
  </div>
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:10px;">Price Range</div>
    <div style="font-size:11px;color:#567592;margin-bottom:3px;">Highest</div>
    <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#2ce4df;margin-bottom:8px;">${max_price:,.0f}</div>
    <div style="font-size:11px;color:#567592;margin-bottom:3px;">Lowest</div>
    <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:700;color:#e7c65a;">${min_price:,.0f}</div>
  </div>
  <div style="background:#0b1c30;border:1px solid rgba(44,228,223,0.10);border-radius:14px;padding:16px 18px;">
    <div style="font-size:9px;font-weight:700;text-transform:uppercase;letter-spacing:0.14em;color:#567592;margin-bottom:6px;">Median Price</div>
    <div style="font-family:'Sora',sans-serif;font-size:18px;font-weight:800;color:#ecf6ff;letter-spacing:-0.04em;">${price_med:,.0f}</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Top listings table ─────────────────────────────────────────────────────────
st.markdown(f"""
<div class='dash-section'>
  <div class='dash-section-bar'></div>
  <div>
    <div class='dash-section-label'>Ranked</div>
    <div class='dash-section-title'>Top Performing Listings — {target_city}</div>
  </div>
</div>
""", unsafe_allow_html=True)

top_table = get_top_listings(df, target_city)

# ── Build rich custom HTML table ──────────────────────────────────────────────
def _prop_type_pill(pt: str) -> str:
    if "Condo" in pt or "Co-op" in pt:
        color, bg = "#b89eff", "rgba(184,158,255,0.10)"
    elif "Single Family" in pt:
        color, bg = "#2ce4df", "rgba(44,228,223,0.10)"
    elif "Multi" in pt:
        color, bg = "#e7c65a", "rgba(231,198,90,0.10)"
    else:
        color, bg = "#5ea6ff", "rgba(94,166,255,0.10)"
    label = pt.replace("Residential", "").replace("Single Family", "SFR").strip()
    return (
        f'<span style="display:inline-block;padding:3px 10px;border-radius:999px;'
        f'font-size:11px;font-weight:600;letter-spacing:0.04em;'
        f'color:{color};background:{bg};border:1px solid {color}33;">{label}</span>'
    )

def _score_bar(score) -> str:
    try:
        val = float(score)
    except Exception:
        val = 0
    pct = min(max(val, 0), 100)
    if pct >= 70:
        bar_color = "linear-gradient(90deg,#2ce4df,#5ea6ff)"
        text_color = "#2ce4df"
    elif pct >= 40:
        bar_color = "linear-gradient(90deg,#5ea6ff,#b89eff)"
        text_color = "#5ea6ff"
    else:
        bar_color = "linear-gradient(90deg,#e7c65a,#ff9f7a)"
        text_color = "#e7c65a"
    return (
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="flex:1;height:5px;border-radius:99px;background:rgba(255,255,255,0.06);">'
        f'<div style="width:{pct:.0f}%;height:100%;border-radius:99px;background:{bar_color};"></div></div>'
        f'<span style="font-family:\'Sora\',sans-serif;font-size:12px;font-weight:700;color:{text_color};min-width:32px;">{pct:.2f}</span>'
        f'</div>'
    )

def _rank_badge(n):
    if n == 0: return "🥇"
    if n == 1: return "🥈"
    if n == 2: return "🥉"
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;' +
        f'width:26px;height:26px;border-radius:50%;background:rgba(86,117,146,0.20);' +
        f'border:1px solid rgba(140,174,200,0.30);font-family:Sora,sans-serif;' +
        f'font-size:11px;font-weight:700;color:#ecf6ff;">{n+1}</span>'
    )

rows_html = ""
for i, (_, row) in enumerate(top_table.iterrows()):
    badge  = _rank_badge(i)
    addr   = str(row.get("ADDRESS", "—"))
    ptype  = str(row.get("PROPERTY TYPE", "—"))
    price  = str(row.get("PRICE", "—"))
    beds   = str(row.get("BEDS", "—"))
    baths  = str(row.get("BATHS", "—"))
    sqft   = str(row.get("SQUARE FEET", "—"))
    dom    = str(row.get("DAYS ON MARKET", "—"))
    score  = row.get("investment_score", 0)

    bg = "rgba(14,34,54,0.55)" if i % 2 == 0 else "rgba(8,20,36,0.30)"
    rows_html += f"""
    <tr style="background:{bg};">
      <td style="padding:14px 16px;text-align:center;font-size:17px;opacity:0.85;">{badge}</td>
      <td style="padding:14px 16px;">
        <div style="font-family:'Sora',sans-serif;font-size:13px;font-weight:600;color:#ecf6ff;margin-bottom:3px;">{addr}</div>
        {_prop_type_pill(ptype)}
      </td>
      <td style="padding:14px 16px;font-family:'Sora',sans-serif;font-size:14px;font-weight:700;color:#ecf6ff;">{price}</td>
      <td style="padding:14px 16px;text-align:center;color:#8baec8;font-size:13px;">{beds}</td>
      <td style="padding:14px 16px;text-align:center;color:#8baec8;font-size:13px;">{baths}</td>
      <td style="padding:14px 16px;text-align:center;color:#8baec8;font-size:13px;">{sqft}</td>
      <td style="padding:14px 16px;text-align:center;">
        <span style="display:inline-block;padding:3px 10px;border-radius:8px;
          background:rgba(231,198,90,0.08);border:1px solid rgba(231,198,90,0.18);
          color:#e7c65a;font-size:12px;font-weight:600;">{dom}d</span>
      </td>
      <td style="padding:14px 20px 14px 16px;min-width:140px;">{_score_bar(score)}</td>
    </tr>
    """

custom_table_html = f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{ background: transparent; font-family: 'DM Sans', sans-serif; }}
  .wrap {{
    background: #0b1c30;
    border: 1px solid rgba(110,196,255,0.09);
    border-radius: 18px;
    overflow: hidden;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  thead tr {{ background: #081628; }}
  th {{
    padding: 11px 16px;
    font-size: 9px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: #567592; text-align: left; white-space: nowrap;
    border-bottom: 1px solid rgba(44,228,223,0.08);
  }}
  th.center {{ text-align: center; }}
  tbody tr {{ transition: background 0.18s; cursor: default; }}
  tbody tr:hover {{ background: rgba(44,228,223,0.045) !important; }}
  td {{ padding: 13px 16px; border-bottom: 1px solid rgba(120,200,255,0.04); }}
  .addr {{
    font-family: 'Sora', sans-serif;
    font-size: 13px; font-weight: 600; color: #ecf6ff; margin-bottom: 4px;
  }}
  .price {{
    font-family: 'Sora', sans-serif;
    font-size: 14px; font-weight: 700; color: #ecf6ff;
  }}
  .muted {{ color: #8baec8; font-size: 13px; text-align: center; }}
  .badge-rank {{ font-size: 17px; opacity: 0.85; text-align: center; }}
</style>
</head>
<body>
<div class="wrap">
  <table>
    <thead>
      <tr>
        <th class="center" style="width:48px;">#</th>
        <th>Property</th>
        <th>Price</th>
        <th class="center">Beds</th>
        <th class="center">Baths</th>
        <th class="center">Sq Ft</th>
        <th class="center">Days on Mkt</th>
        <th>AI Score</th>
      </tr>
    </thead>
    <tbody>
      {rows_html}
    </tbody>
  </table>
</div>
</body>
</html>
"""

num_rows = len(top_table)
table_height = 56 + (num_rows * 68) + 20
components.html(custom_table_html, height=table_height, scrolling=False)

st.markdown("""
<div class='footer-note'>MOLLECUL · AI Real Estate Intelligence · DFW Metro</div>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)
