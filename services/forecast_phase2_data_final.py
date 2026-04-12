
import pandas as pd
import numpy as np


def _compute_sp500_returns(sp500_df: pd.DataFrame) -> tuple[dict, dict]:
    df = sp500_df.copy()
    df.columns = [c.strip() for c in df.columns]

    date_col = None
    value_col = None
    for c in df.columns:
        cl = c.lower()
        if "date" in cl:
            date_col = c
        if c.upper() == "SP500" or "sp500" in cl:
            value_col = c

    if date_col is None or value_col is None:
        raise ValueError("SP500.csv must contain observation_date and SP500 columns.")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df.dropna(subset=[date_col, value_col]).sort_values(date_col)

    if len(df) < 3:
        raise ValueError("SP500.csv needs at least 3 valid rows to compute return and lag1 return.")

    latest = df.iloc[-1]
    prev = df.iloc[-2]
    prev2 = df.iloc[-3]

    latest_return = (latest[value_col] / prev[value_col]) - 1.0
    lag1_return = (prev[value_col] / prev2[value_col]) - 1.0

    overrides = {
        "sp500_return": float(latest_return),
        "sp500_return_lag1": float(lag1_return),
        "month": int(latest[date_col].month),
        "quarter": int(((int(latest[date_col].month) - 1) // 3) + 1),
        "is_spring": 1 if int(latest[date_col].month) in [3, 4, 5] else 0,
        "is_summer": 1 if int(latest[date_col].month) in [6, 7, 8] else 0,
    }

    context = {
        "latest_date": latest[date_col],
        "sp500_level": float(latest[value_col]),
        "sp500_return": float(latest_return),
        "sp500_return_lag1": float(lag1_return),
    }
    return overrides, context


def _load_temperature_context(temp_csv_path: str | None) -> dict:
    if not temp_csv_path:
        return {}

    df = pd.read_csv(temp_csv_path)
    df.columns = [c.strip() for c in df.columns]

    date_col = None
    max_col = None
    normal_max_col = None

    for c in df.columns:
        cl = c.lower()
        if cl == "date":
            date_col = c
        elif cl == "max":
            max_col = c
        elif cl == "normal_max":
            normal_max_col = c

    if date_col is None or max_col is None or normal_max_col is None:
        return {}

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[max_col] = pd.to_numeric(df[max_col], errors="coerce")
    df[normal_max_col] = pd.to_numeric(df[normal_max_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    if df.empty:
        return {}

    latest = df.iloc[-1]
    anomaly = np.nan
    if pd.notna(latest[max_col]) and pd.notna(latest[normal_max_col]):
        anomaly = float(latest[max_col] - latest[normal_max_col])

    return {
        "latest_temp_date": latest[date_col],
        "latest_temp_anomaly_f": anomaly,
    }


def build_phase2_forecast_overrides(sp500_csv_path: str, temp_csv_path: str | None = None) -> tuple[dict, dict]:
    """
    Phase 2 file-driven forecast data helper.

    Returns
    -------
    overrides : dict
        Forecast feature overrides driven from CSV files where possible.
    context : dict
        Extra display/context values for the UI.
    """
    sp500_df = pd.read_csv(sp500_csv_path)
    overrides, context = _compute_sp500_returns(sp500_df)

    temp_context = _load_temperature_context(temp_csv_path)
    context.update(temp_context)

    return overrides, context
