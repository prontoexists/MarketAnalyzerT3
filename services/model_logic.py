import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# ── Feature Definitions ──────────────────────────────────────────────────────
#
# All columns from the CSV are accounted for below.
#
# Excluded columns and why:
#   attomId, apn          – row identifiers, no predictive value
#   fips                  – constant (all rows same county code)
#   address               – free-text, not encodable as a category
#   state                 – constant (TX)
#   geoIdV4_N1            – 100 % null in the dataset
#   geoIdV4_ZI            – single unique value (constant per zip); adds no signal
#   ownerName             – 100 % null
#   saleTransDate,
#   saleRecDate           – redundant with saleDate (same event, different timestamps)
#   saleDocNum            – document-number identifier, not a predictive feature
#
# TARGET: avmValue  (never used as a feature)
#
# AVM-derived columns excluded to prevent data leakage:
#   avmConfidence, avmHigh, avmLow, avmValueRange,
#   avmLastMonth, avmChangeAmt, avmChangePct,
#   avmPerSqft, avmDate
#
# ─────────────────────────────────────────────────────────────────────────────

NUMERIC_FEATURES = [
    # ── Location ──────────────────────────────────────────────────────────────
    'lat', 'lng',

    # ── Age ───────────────────────────────────────────────────────────────────
    'yearBuilt', 'yearBuiltEffective',

    # ── Building size ─────────────────────────────────────────────────────────
    'sqft', 'livingArea', 'grossSqft', 'groundFloorSqft',

    # ── Lot ───────────────────────────────────────────────────────────────────
    'lotSqft', 'lotAcres',

    # ── Rooms ─────────────────────────────────────────────────────────────────
    'beds', 'bathsFull', 'bathsHalf', 'bathsTotal', 'totalRooms',

    # ── Structure ─────────────────────────────────────────────────────────────
    'stories', 'garageSize', 'garageSpaces', 'basementSqft', 'fireplaces',

    # ── Assessment / Tax ──────────────────────────────────────────────────────
    'assessedTotal', 'assessedLand', 'assessedImprov',
    'assessedPerSqft', 'assessedImprPerSqft',
    'marketTotal', 'marketLand', 'marketImprov',
    'taxAmount', 'taxYear', 'taxPerSqft',

    # ── Calculated values ─────────────────────────────────────────────────────
    'calcTotalValue', 'calcLandValue', 'calcImprValue', 'calcValuePerSqft',

    # ── Sale transaction metrics ───────────────────────────────────────────────
    'salePrice',       # last recorded sale price
    'pricePerSqft',    # $/sqft at last recorded sale
    'pricePerBed',     # $/bed  at last recorded sale
    'disclosed',       # 1 = sale price was publicly disclosed
    'saleYear',        # year of last recorded sale (derived from saleDate)
]

CATEGORICAL_FEATURES = [
    # ── Location hierarchy ────────────────────────────────────────────────────
    'zip',
    'city',
    'county',
    'municipality',
    'subdivision',      # neighborhood / platted subdivision name
    'taxCodeArea',      # tax district code(s)

    # ── Geo boundary IDs (ATTOM proprietary hash keys) ────────────────────────
    # High-cardinality but stable geographic identifiers that let XGBoost
    # learn value-area associations at multiple spatial resolutions.
    'geoIdV4_N2',       # neighbourhood-level boundary
    'geoIdV4_N4',       # sub-neighbourhood boundary
    'geoIdV4_DB',       # district boundary
    'geoIdV4_SB',       # sub-boundary

    # ── Property classification ────────────────────────────────────────────────
    'propertyType',     # SFR, DUPLEX, …
    'propSubtype',      # Residential, Commercial, …
    'propClass',        # Single Family Residence / Townhouse, …
    'propLandUse',      # ATTOM land-use code

    # ── Ownership ─────────────────────────────────────────────────────────────
    'ownerOccupied',    # OWNER OCCUPIED vs ABSENTEE

    # ── Amenities ─────────────────────────────────────────────────────────────
    'pool',
    'basement',

    # ── Building characteristics ──────────────────────────────────────────────
    'bldgType',
    'condition',
    'view',
    'floors',
    'garageType',
    'heatingType',
    'coolingType',
    'constructionType',
    'foundationType',
    'roofMaterial',
    'roofType',
    'wallType',
    'frameType',

    # ── Sale transaction characteristics ──────────────────────────────────────
    'saleType',         # Resale, New Construction, …
    'saleDocType',      # DEED, LAND CONTRACT, …
    'cashOrMortgage',   # M = mortgage, C = cash
    'newConstruction',
    'interFamily',      # inter-family transfer flag
    'sellerCarryback',
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_COL      = 'avmValue'
COMPARISON_COLS = ['avmValue', 'salePrice']

# Sentinel for missing / unknown categorical values.
# Must be a plain string so OrdinalEncoder only ever sees str arrays.
_CAT_MISSING = '__MISSING__'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_cat(series: pd.Series) -> pd.Series:
    _null_strs = {'nan', 'None', 'NaN', '<NA>', 'none', 'NULL', ''}
    s = series.astype(object).fillna(_CAT_MISSING).astype(str)
    return s.mask(s.isin(_null_strs), _CAT_MISSING)


def _extract_sale_year(series: pd.Series) -> pd.Series:
    """
    Parse a raw saleDate column (e.g. '4/29/19', '2021-08-15') to a
    4-digit float year.  Unparseable / missing values become NaN.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress pandas format-inference UserWarning
        parsed = pd.to_datetime(series, errors='coerce')
    years = parsed.dt.year.astype('float64')
    # Clamp 2-digit-year misassignments: '99' → 2099 should be 1999
    current_year = pd.Timestamp.now().year
    years = years.where(years <= current_year, years - 100)
    return years


# ── Data Preparation ─────────────────────────────────────────────────────────

def prepare_data(file_path: str):
    """
    Load and clean the property dataset.

    Returns
    -------
    X    : DataFrame of ALL_FEATURES (salePrice included as a feature)
    y    : Series of avmValue targets
    meta : DataFrame with avmValue + salePrice aligned to X / y rows
    """
    print(f"  Reading CSV from {file_path} ...")

    # We need raw feature columns (minus derived 'saleYear') + saleDate + target
    raw_needed = list(set(ALL_FEATURES) - {'saleYear'}) + ['saleDate', TARGET_COL]

    df_peek   = pd.read_csv(file_path, nrows=0)
    available = [c for c in raw_needed if c in df_peek.columns]
    missing_from_csv = [c for c in raw_needed if c not in df_peek.columns]
    if missing_from_csv:
        print(f"  Warning — absent from CSV (will default to NaN/MISSING): {missing_from_csv}")

    df = pd.read_csv(file_path, usecols=available, low_memory=False)

    # Fill completely absent columns so nothing KeyErrors downstream
    for col in raw_needed:
        if col not in df.columns:
            df[col] = np.nan

    # ── Numeric ──────────────────────────────────────────────────────────────
    for col in [c for c in NUMERIC_FEATURES if c != 'saleYear']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Derive saleYear from saleDate
    df['saleYear'] = _extract_sale_year(df['saleDate'])

    # ── Categorical → clean sentinel strings ─────────────────────────────────
    for col in CATEGORICAL_FEATURES:
        df[col] = _clean_cat(df[col])

    # ── Target column ─────────────────────────────────────────────────────────
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors='coerce')

    # ── Drop rows with no target ──────────────────────────────────────────────
    df = df.dropna(subset=[TARGET_COL])

    print(f"  Rows after cleaning:  {len(df):,}")
    print(f"  Rows with salePrice:  {df['salePrice'].notna().sum():,}")

    X    = df[ALL_FEATURES].reset_index(drop=True)
    y    = df[TARGET_COL].reset_index(drop=True)
    meta = df[[TARGET_COL, 'salePrice']].reset_index(drop=True)

    return X, y, meta


# ── Pipeline / Model ─────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    sklearn Pipeline:
      ColumnTransformer
        • passthrough     --> NUMERIC_FEATURES  (XGBoost handles NaN natively)
        • OrdinalEncoder  --> CATEGORICAL_FEATURES  (unknown/missing → -1)
      XGBRegressor
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', 'passthrough', NUMERIC_FEATURES),
            (
                'cat',
                OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,       # unseen category strings at predict time
                    dtype=np.float64,
                ),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder='drop',
    ).set_output(transform="default")   # always numpy, never DataFrame

    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.04,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.75,
        reg_alpha=0.1,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=42,
        enable_categorical=False,
    )

    return Pipeline([('preprocessor', preprocessor), ('model', model)])


def train_pipeline(X: pd.DataFrame, y: pd.Series, model_path: str) -> Pipeline:
    """Fit the pipeline on the training set and save it."""
    pipeline = build_pipeline()
    print("  Fitting pipeline …")
    pipeline.fit(X, y)
    joblib.dump(pipeline, model_path)
    print(f"  Pipeline saved → {model_path}")
    return pipeline


def load_model(model_path: str) -> Pipeline:
    """Load the saved sklearn Pipeline from disk."""
    try:
        obj = joblib.load(model_path)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Model not found at '{model_path}'.\n"
            "  → Run:  python model_logic.py  to train and save a new model."
        )
    except Exception as e:
        raise Exception(f"Error loading model: {e}")

    if not isinstance(obj, Pipeline):
        raise TypeError(
            f"'{model_path}' is a {type(obj).__name__}, not a sklearn Pipeline.\n"
            "  → Stale model from before the refactor — delete it and retrain:\n"
            "      python model_logic.py"
        )
    return obj


def predict(pipeline: Pipeline, data: pd.DataFrame) -> np.ndarray:
    """
    Make predictions.  Runs preprocessor and XGBoost explicitly so we can
    cast to numpy float64 in between - some sklearn/pandas versions return a
    DataFrame from ColumnTransformer which XGBoost then rejects.
    """
    if isinstance(data, pd.Series):
        data = data.to_frame().T

    # Guarantee every feature column is present
    for col in NUMERIC_FEATURES:
        if col not in data.columns:
            data[col] = np.nan
    for col in CATEGORICAL_FEATURES:
        if col not in data.columns:
            data[col] = _CAT_MISSING
        else:
            data[col] = _clean_cat(data[col])

    data = data[ALL_FEATURES]

    # Step 1 — encode --> guaranteed numpy float64
    preprocessor  = pipeline.named_steps['preprocessor']
    X_transformed = preprocessor.transform(data)
    if hasattr(X_transformed, 'values'):    # defensive: DataFrame → numpy
        X_transformed = X_transformed.values
    X_transformed = np.asarray(X_transformed, dtype=np.float64)

    # Step 2 — XGBoost on clean numpy array
    return pipeline.named_steps['model'].predict(X_transformed)


def get_model_info(pipeline: Pipeline) -> dict:
    """Return metadata about the loaded pipeline."""
    xgb_model = pipeline.named_steps['model']
    return {
        "n_features_in":        getattr(xgb_model, 'n_features_in_', None),
        "n_estimators":         xgb_model.n_estimators,
        "numeric_features":     NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
    }


def build_comparison(pipeline: Pipeline, X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Return predictions + avmValue + salePrice side-by-side."""
    preds = predict(pipeline, X.copy())
    comp  = X.copy()
    comp['model_estimate'] = preds
    comp['avm_value']      = meta[TARGET_COL].values
    comp['sale_price']     = meta['salePrice'].values
    comp['model_vs_avm']   = comp['model_estimate'] - comp['avm_value']
    comp['model_vs_sale']  = comp['model_estimate'] - comp['sale_price']
    comp['avm_vs_sale']    = comp['avm_value']       - comp['sale_price']
    return comp


# ── Main: Train & Evaluate ───────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    file_name  = sys.argv[1] if len(sys.argv) > 1 else 'PART1_NEW_Dallas_Properties.csv'
    model_path = 'VER4_property_valuation_model.joblib'

    # 1. Prepare
    print("\nPreparing data …")
    X, y, meta = prepare_data(file_name)

    print(f"\n  Total features:  {len(ALL_FEATURES)}")
    print(f"    Numeric:       {len(NUMERIC_FEATURES)}")
    print(f"    Categorical:   {len(CATEGORICAL_FEATURES)}")
    print(f"  Target:          {TARGET_COL}  (ATTOM AVM)")

    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, meta, test_size=0.2, random_state=42
    )

    # 2. Train
    print("\nTraining pipeline …")
    pipeline = train_pipeline(X_train, y_train, model_path)

    # 3. Evaluate
    loaded = load_model(model_path)
    y_pred = predict(loaded, X_test.copy())

    r2   = r2_score(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    print("\n--- Model Performance (vs avmValue) ---")
    print(f"R-squared:               {r2:.4f}")
    print(f"Mean Absolute Error:     ${mae:,.2f}")
    print(f"Root Mean Squared Error: ${rmse:,.2f}")

    # 4. Three-way comparison
    comparison = build_comparison(loaded, X_test.copy(), meta_test.reset_index(drop=True))
    has_sale   = comparison['sale_price'].notna()

    print(f"\n--- Three-Way Comparison (rows with salePrice: {has_sale.sum():,}) ---")
    for label, col in [
        ("Model vs AVM",  'model_vs_avm'),
        ("Model vs Sale", 'model_vs_sale'),
        ("AVM   vs Sale", 'avm_vs_sale'),
    ]:
        sub = comparison.loc[has_sale, col]
        print(f"\n  {label}")
        print(f"    Mean diff:  ${sub.mean():>12,.2f}")
        print(f"    Median:     ${sub.median():>12,.2f}")
        print(f"    MAE:        ${sub.abs().mean():>12,.2f}")

    # 5. Sample rows
    print("\n--- Sample Valuations (first 5 rows with a salePrice) ---")
    sample_cols = ['sqft', 'beds', 'bathsFull', 'yearBuilt',
                   'model_estimate', 'avm_value', 'sale_price',
                   'model_vs_avm', 'model_vs_sale', 'avm_vs_sale']
    pd.set_option('display.float_format', '${:,.0f}'.format)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 140)
    print(comparison.loc[has_sale, sample_cols].head(5).to_string(index=False))

    # 6. Feature importance — top 25
    importances = pd.Series(
        loaded.named_steps['model'].feature_importances_,
        index=ALL_FEATURES,
    )
    print("\n--- Top 25 Feature Importances ---")
    print(importances.sort_values(ascending=False).head(25).to_string())

    # 7. Save comparison CSV
    out_path = 'property_comparison.csv'
    comparison.to_csv(out_path, index=False)
    print(f"\nFull comparison saved → {out_path}")

