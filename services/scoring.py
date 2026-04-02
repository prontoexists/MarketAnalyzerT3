import numpy as np
import pandas as pd


def add_investment_score(df: pd.DataFrame) -> pd.DataFrame:
    scored_df = df.copy()

    # Ensure core numeric columns exist
    if "PRICE" not in scored_df.columns:
        scored_df["PRICE"] = np.nan
    if "SQUARE FEET" not in scored_df.columns:
        scored_df["SQUARE FEET"] = np.nan
    if "DAYS ON MARKET" not in scored_df.columns:
        scored_df["DAYS ON MARKET"] = np.nan
    if "BEDS" not in scored_df.columns:
        scored_df["BEDS"] = 0

    # Compute/fill price per sqft
    if "PRICE_PER_SQFT_FINAL" not in scored_df.columns:
        scored_df["PRICE_PER_SQFT_FINAL"] = (
            scored_df["PRICE"] / scored_df["SQUARE FEET"].replace(0, np.nan)
        )

    # Safe fallback for days on market
    median_dom = scored_df["DAYS ON MARKET"].median()
    if pd.isna(median_dom):
        median_dom = 0

    scored_df["DAYS_ON_MARKET_SAFE"] = scored_df["DAYS ON MARKET"].fillna(median_dom)

    scored_df["investment_score"] = (
        scored_df["PRICE_PER_SQFT_FINAL"].fillna(0) * 0.6 +
        (1 / (scored_df["DAYS_ON_MARKET_SAFE"] + 1)) * 1000 * 0.3 +
        scored_df["BEDS"].fillna(0) * 2 * 0.1
    )

    return scored_df


def get_top_listings(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    working_df = df.copy()

    if "investment_score" not in working_df.columns:
        working_df = add_investment_score(working_df)

    return working_df.sort_values("investment_score", ascending=False).head(top_n)