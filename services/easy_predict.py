import sys
from round2_model import _get_base_estimate_from_address, predict_forward

# Your keys
ATTOM_KEY = "38c0c6bd6a465da7b414bc23a5df9791"
FRED_KEY = "3edb96e23b26e2271758689309faee32"
NOAA_KEY = "xyeXCwTZHCBtzcdEHwQpnNvsfTyiWmGq"
EPA_KEY = "aquaosprey52"

# A dictionary to translate complex economic variables into simple English
FACTOR_TRANSLATIONS = {
    "mortgage_rate_30yr": "Mortgage Interest Rates",
    "mortgage_rate_30yr_lag1": "Recent Mortgage Rate Changes",
    "vix": "Stock Market Fear / Volatility",
    "vix_lag1": "Recent Stock Market Fear",
    "case_shiller_dallas": "Overall Dallas Housing Trends",
    "fed_funds_rate": "Federal Reserve Policies",
    "is_summer": "Time of Year (Summer buying season)",
    "quarter": "Time of Year (Seasonality)",
    "housing_starts_south": "Number of New Homes Being Built",
    "yield_spread": "Bond Market Health",
    "cs_dallas_mom_pct_lag1": "Recent Dallas Price Momentum",
    "new_home_sales": "New Home Sales Volume",
    "labor_force_part_texas": "Texas Labor Market Strength",
    "wage_growth_texas": "Texas Wage Growth",
    "month": "Time of Year (Monthly Seasonality)",
}

def run_simple_prediction(address):
    print(f"🔍 Scanning property and market data for: {address}...\n")
    
    # 1. Get Base Value (Stage 1)
    base_value = _get_base_estimate_from_address(address, ATTOM_KEY)
    
    # 2. Get Forecast (Stage 2) - using verbose=False to hide the complex table
    result = predict_forward(
        base_estimate=base_value,
        fred_api_key=FRED_KEY,
        noaa_api_key=NOAA_KEY,
        epa_api_key=EPA_KEY,
        verbose=False 
    )
    
    current = base_value
    future = result.get('forward_estimate', base_value)
    diff = result.get('change_dollars', 0)
    pct = result.get('change_pct', 0)
    
    # 3. Determine the "Market Weather"
    if pct >= 2.0:
        trend = "🔥 HOT MARKET: Prices are rising fast. Great for sellers!"
    elif pct > 0:
        trend = "🌤️ WARM MARKET: Prices are slowly going up."
    elif pct > -1.5:
        trend = "☁️ FLAT MARKET: Prices are staying mostly the same."
    else:
        trend = "🌧️ COOLING MARKET: Prices are dropping. Good for buyers."
        
    # 4. Print the Simple Dashboard
    print("="*60)
    print(" 🏡  MOLLECUL SIMPLE PROPERTY FORECAST  🏡")
    print("="*60)
    print(f"📍 Address: {address}")
    print(f"\n💰 TODAY'S VALUE:  ${current:,.0f}")
    print(f"🔮 IN 6 MONTHS:    ${future:,.0f}")
    
    # Format the change string nicely
    if diff >= 0:
        print(f"📈 EXPECTED CHANGE: UP ${diff:,.0f} (+{pct:,.2f}%)")
    else:
        print(f"📉 EXPECTED CHANGE: DOWN ${abs(diff):,.0f} ({pct:,.2f}%)")
        
    print(f"\n🌡️ MARKET WEATHER: {trend}")
    print("="*60)
    
    # 5. Explain the top drivers in plain English
    factors = result.get('shap_factors', [])
    if factors:
        print("\n💡 TOP 3 REASONS DRIVING THIS PREDICTION:")
        # Get the top 3 factors
        top_factors = factors[:3]
        
        for f in top_factors:
            factor_name = f['feature']
            impact = f['shap_dollar']
            
            # Translate the scary variable name into plain English
            simple_name = FACTOR_TRANSLATIONS.get(factor_name, "General Economic Conditions")
            
            if impact > 0:
                print(f"  🟢 {simple_name} is pushing the value UP.")
            else:
                print(f"  🔴 {simple_name} is pulling the value DOWN.")
    print("\n")

if __name__ == "__main__":
    # You can change the address here
    target_address = "3205 Walker Dr, richardson, tx, 75082"
    
    if len(sys.argv) > 1:
        target_address = " ".join(sys.argv[1:])
        
    run_simple_prediction(target_address)