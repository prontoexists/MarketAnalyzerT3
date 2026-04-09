import requests
import pandas as pd
import numpy as np
from model_logic import (
    load_model, predict,
    ALL_FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
    _clean_cat, _CAT_MISSING, _extract_sale_year,
    TARGET_COL,
)

API_KEY    = "38c0c6bd6a465da7b414bc23a5df9791"
MODEL_PATH = "VER4_property_valuation_model.joblib"
ATTOM_URL  = "https://api.gateway.attomdata.com/propertyapi/v1.0.0/allevents/detail"


def get_property_data(full_address: str) -> dict:
    # hits the allevents/detail endpoint and returns a flat dict matching ALL_FEATURES
    # anything the API doesn't return comes back as None, handled downstream
    headers = {"Accept": "application/json", "apikey": API_KEY}
    params  = {"address": full_address}

    response = requests.get(ATTOM_URL, headers=headers, params=params)
    data     = response.json()

    status = data.get("status", {})
    if status.get("code") == 400 and status.get("msg") == "SuccessWithoutResult":
        raise ValueError(
            f"ATTOM couldn't find: '{full_address}'\n"
            "  Try: 4529 Wateka Dr, Dallas, TX 75209\n"
            "  Make sure street suffix is included (St, Dr, Ave, etc.)"
        )

    if response.status_code != 200:
        raise Exception(f"API Error {response.status_code}: {response.text}")

    properties = data.get("property", [])
    if not properties:
        raise ValueError("No property found for that address.")

    prop = properties[0]

    # unpack the nested mess that ATTOM returns
    addr     = prop.get("address", {})    or {}
    loc      = prop.get("location", {})   or {}
    geo_v4   = loc.get("geoIdV4", {})     or {}
    area     = prop.get("area", {})        or {}
    summ     = prop.get("summary", {})     or {}
    utils    = prop.get("utilities", {})   or {}
    bldg     = prop.get("building", {})    or {}
    bsize    = bldg.get("size", {})        or {}
    rooms    = bldg.get("rooms", {})       or {}
    interior = bldg.get("interior", {})    or {}
    bconst   = bldg.get("construction", {}) or {}
    bparking = bldg.get("parking", {})     or {}
    bsumm    = bldg.get("summary", {})     or {}
    lot      = prop.get("lot", {})         or {}
    assessment = prop.get("assessment", {}) or {}
    assessed = assessment.get("assessed", {})      or {}
    market_v = assessment.get("market", {})        or {}
    tax_data = assessment.get("tax", {})           or {}
    calc_v   = assessment.get("calculations", {})  or {}
    avm      = prop.get("avm", {})         or {}
    avmamt   = avm.get("amount", {})       or {}
    sale     = prop.get("sale", {})        or {}
    saleamt  = sale.get("amount", {})      or {}
    salecalc = sale.get("calculation", {}) or {}

    extracted = {
        'lat':                  loc.get("latitude"),
        'lng':                  loc.get("longitude"),

        'yearBuilt':            summ.get("yearbuilt"),
        'yearBuiltEffective':   bsumm.get("yearbuilteffective"),

        'sqft':                 bsize.get("universalsize"),
        'livingArea':           bsize.get("livingsize"),
        'grossSqft':            bsize.get("grosssize"),
        'groundFloorSqft':      bsize.get("groundfloorsize"),

        'lotSqft':              lot.get("lotsize2"),
        'lotAcres':             lot.get("lotsize1"),

        'beds':                 rooms.get("beds"),
        'bathsFull':            rooms.get("bathsfull"),
        'bathsHalf':            rooms.get("bathshalf"),
        'bathsTotal':           rooms.get("bathstotal"),
        'totalRooms':           rooms.get("roomstotal"),
        'fireplaces':           interior.get("fplccount"),

        'stories':              bsumm.get("levels"),
        'garageSize':           bparking.get("prkgSize"),
        'garageSpaces':         bparking.get("prkgSpaces"),
        'basementSqft':         bsize.get("basementsize"),

        # assessment keys are all lowercase in the actual API responses
        'assessedTotal':        assessed.get("assdttlvalue"),
        'assessedLand':         assessed.get("assdlandvalue"),
        'assessedImprov':       assessed.get("assdimprvalue"),
        'assessedPerSqft':      assessed.get("assdttlpersizeunit"),
        'assessedImprPerSqft':  assessed.get("assdimprpersizeunit"),
        'marketTotal':          market_v.get("mktttlvalue"),
        'marketLand':           market_v.get("mktlandvalue"),
        'marketImprov':         market_v.get("mktimprvalue"),
        'taxAmount':            tax_data.get("taxamt"),
        'taxYear':              tax_data.get("taxyear"),
        'taxPerSqft':           tax_data.get("taxpersizeunit"),

        'calcTotalValue':       calc_v.get("calcttlvalue"),
        'calcLandValue':        calc_v.get("calclandvalue"),
        'calcImprValue':        calc_v.get("calcimprvalue"),
        'calcValuePerSqft':     calc_v.get("calcvaluepersizeunit"),

        'salePrice':            saleamt.get("saleamt"),
        'pricePerSqft':         salecalc.get("pricepersizeunit"),
        'pricePerBed':          salecalc.get("priceperbed"),
        'disclosed':            saleamt.get("saledisclosuretype"),
        'saleYear':             None,  # derived below

        'zip':                  addr.get("postal1"),
        'city':                 addr.get("locality"),
        'county':               area.get("countrysecsubd"),
        'municipality':         area.get("munname"),
        'subdivision':          area.get("subdname"),
        'taxCodeArea':          area.get("taxcodearea"),

        # N4 and SB are often missing, that's fine
        'geoIdV4_N2':           geo_v4.get("N2"),
        'geoIdV4_N4':           geo_v4.get("N4"),
        'geoIdV4_DB':           geo_v4.get("DB"),
        'geoIdV4_SB':           geo_v4.get("SB"),

        'propertyType':         summ.get("proptype"),
        'propSubtype':          summ.get("propsubtype"),
        'propClass':            summ.get("propclass"),
        'propLandUse':          summ.get("propLandUse"),

        'ownerOccupied':        summ.get("absenteeInd"),

        'pool':                 lot.get("pooltype"),
        'basement':             bsize.get("bsmtsize"),

        # some of these keys have mixed case in the API, annoying but whatever
        'bldgType':             bsumm.get("bldgType"),
        'condition':            bsumm.get("condition"),
        'view':                 bsumm.get("view"),
        'floors':               interior.get("floors"),
        'garageType':           bparking.get("garagetype"),
        'heatingType':          utils.get("heatingtype"),
        'coolingType':          utils.get("coolingtype"),
        'constructionType':     bconst.get("constructiontype"),
        'foundationType':       bconst.get("foundationtype"),
        'roofMaterial':         bconst.get("roofcover"),
        'roofType':             bconst.get("roofShape"),
        'wallType':             bconst.get("wallType"),
        'frameType':            bconst.get("frameType"),

        'saleType':             saleamt.get("saletranstype"),
        'saleDocType':          saleamt.get("saledoctype"),
        'cashOrMortgage':       sale.get("cashormortgagepurchase"),
        'newConstruction':      sale.get("resaleornewconstruction"),
        'interFamily':          sale.get("interfamily"),
        'sellerCarryback':      sale.get("sellercarryback"),

        'avmValue':             avmamt.get("value"),  # not a model feature, just for display
    }

    raw_date = sale.get("salesearchdate") or sale.get("saledate")
    if raw_date:
        try:
            extracted['saleYear'] = float(
                _extract_sale_year(pd.Series([str(raw_date)])).iloc[0]
            )
        except Exception:
            extracted['saleYear'] = None

    return extracted


def build_feature_row(raw: dict) -> pd.DataFrame:
    # converts the raw dict into a single-row DataFrame
    # None -> NaN for numerics, None -> _CAT_MISSING for categoricals
    row = {}

    for col in NUMERIC_FEATURES:
        val = raw.get(col)
        try:
            row[col] = float(val) if val is not None else np.nan
        except (TypeError, ValueError):
            row[col] = np.nan

    for col in CATEGORICAL_FEATURES:
        val = raw.get(col)
        if val is None or str(val).strip() in ('', 'None', 'nan', 'NaN'):
            row[col] = _CAT_MISSING
        else:
            row[col] = str(val).strip()

    return pd.DataFrame([row])[ALL_FEATURES]


def _feature_population(df_features: pd.DataFrame) -> tuple[int, int, int]:
    num_pop   = sum(not pd.isna(df_features[c].iloc[0]) for c in NUMERIC_FEATURES)
    cat_pop   = sum(df_features[c].iloc[0] != _CAT_MISSING for c in CATEGORICAL_FEATURES)
    total     = len(ALL_FEATURES)
    populated = num_pop + cat_pop
    return populated, total, total - populated


def main():
    print("=========================================")
    print("      Property Valuation Predictor       ")
    print("=========================================")

    full_address = input(
        "Enter Full Address (e.g., 4529 Wateka Dr, Dallas, TX 75209): "
    ).strip()

    print("\nFetching from ATTOM...")
    try:
        raw_data = get_property_data(full_address)
    except Exception as e:
        print(f"\n{e}")
        return

    df_features = build_feature_row(raw_data)
    populated, total, missing = _feature_population(df_features)
    print(f"Features: {populated}/{total} populated ({missing} missing, XGBoost handles those)")

    try:
        pipeline = load_model(MODEL_PATH)
    except Exception as e:
        print(f"\nCouldn't load model: {e}")
        return

    try:
        model_estimate = predict(pipeline, df_features)[0]
    except Exception as e:
        print(f"\nPrediction failed: {e}")
        return

    avm_value  = raw_data.get('avmValue')
    sale_price = raw_data.get('salePrice')

    print("\n" + "=" * 52)
    print("              VALUATION RESULTS              ")
    print("=" * 52)
    print(f"Address:        {full_address.upper()}")
    print("-" * 52)
    print(f"Model Estimate: ${model_estimate:,.0f}")

    if avm_value:
        avm_value = float(avm_value)
        diff_avm  = model_estimate - avm_value
        pct_avm   = (diff_avm / avm_value) * 100
        print(f"ATTOM AVM:      ${avm_value:,.0f}")
        print(f"  Model vs AVM: {'+' if diff_avm >= 0 else '-'}"
              f"${abs(diff_avm):,.0f}  ({'+' if pct_avm >= 0 else ''}{pct_avm:.1f}%)")
    else:
        print("ATTOM AVM:      N/A")

    if sale_price:
        sale_price = float(sale_price)
        diff_sale  = model_estimate - sale_price
        pct_sale   = (diff_sale / sale_price) * 100
        print(f"Sale Price:     ${sale_price:,.0f}")
        print(f"  Model vs Sale:{'+' if diff_sale >= 0 else '-'}"
              f"${abs(diff_sale):,.0f}  ({'+' if pct_sale >= 0 else ''}{pct_sale:.1f}%)")
    else:
        print("Sale Price:     N/A (no recorded sale)")

    print("=" * 52)

    if input("\nShow missing features? (y/N): ").strip().lower() == 'y':
        missing_num = [c for c in NUMERIC_FEATURES if pd.isna(df_features[c].iloc[0])]
        missing_cat = [c for c in CATEGORICAL_FEATURES if df_features[c].iloc[0] == _CAT_MISSING]

        print(f"\nMissing numeric ({len(missing_num)}):")
        for c in missing_num:
            print(f"  {c}")
        print(f"\nMissing categorical ({len(missing_cat)}):")
        for c in missing_cat:
            print(f"  {c}")

    print()


if __name__ == "__main__":
    main()