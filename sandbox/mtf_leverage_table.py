"""
Per-symbol MTF (Margin Trading Facility) leverage table -- Nifty 100.

Point-in-time Dhan margincalculator snapshot (one 1-share BUY MTF margin
call per symbol at live LTP). Replaces the Sandbox engine's old flat
equity_mtf_leverage placeholder (2x) with real per-symbol data for the
names it covers.

Real leverage in this snapshot ranges about 2.38x to 4.55x. Symbols not
in this table fall back to the equity_mtf_leverage Sandbox config, still
editable via the Sandbox settings page (seeded at this survey's median,
4.35x).

This is not a live feed -- broker margin requirements can change. Refresh
this table if it drifts noticeably from live data.
"""

MTF_LEVERAGE_BY_SYMBOL: dict[str, float] = {
    "ADANIENT": 3.81, "ADANIPORTS": 4.00, "APOLLOHOSP": 4.55, "BAJAJFINSV": 4.55,
    "AXISBANK": 4.55, "BHARTIARTL": 4.55, "BAJAJ-AUTO": 4.55, "BAJFINANCE": 4.35,
    "BEL": 4.35, "ASIANPAINT": 4.55, "DRREDDY": 4.55, "EICHERMOT": 4.44,
    "COALINDIA": 4.55, "CIPLA": 4.55, "HDFCBANK": 4.55, "GRASIM": 4.55,
    "HCLTECH": 4.44, "HDFCLIFE": 4.55, "HINDUNILVR": 4.55, "ICICIBANK": 4.55,
    "INFY": 4.35, "HINDALCO": 4.35, "INDIGO": 4.00, "ITC": 4.55,
    "JSWSTEEL": 4.55, "M&M": 4.35, "KOTAKBANK": 4.55, "LT": 4.44,
    "MARUTI": 4.55, "MAXHEALTH": 4.17, "NESTLEIND": 4.55, "POWERGRID": 4.55,
    "NTPC": 4.55, "ONGC": 4.55, "SBILIFE": 4.55, "SHRIRAMFIN": 3.95,
    "SBIN": 4.55, "RELIANCE": 4.55, "TATASTEEL": 4.35, "SUNPHARMA": 4.55,
    "TCS": 4.55, "TECHM": 4.54, "TATACONSUM": 4.55, "TMPV": 4.26,
    "TITAN": 4.55, "TRENT": 3.87, "ULTRACEMCO": 4.55, "WIPRO": 4.55,
    "ETERNAL": 3.94, "JIOFIN": 4.35, "ABB": 4.23, "ADANIPOWER": 3.33,
    "AMBUJACEM": 4.17, "ADANIENSOL": 3.33, "ADANIGREEN": 3.39, "BAJAJHLDNG": 4.00,
    "BANKBARODA": 4.35, "CGPOWER": 4.05, "ZYDUSLIFE": 4.55, "DLF": 4.26,
    "BPCL": 4.35, "CANBK": 4.30, "DIVISLAB": 4.55, "BRITANNIA": 4.55,
    "CHOLAFIN": 4.11, "CUMMINSIND": 4.35, "DMART": 4.55, "BOSCHLTD": 4.35,
    "HAL": 4.17, "GODREJCP": 4.55, "GAIL": 4.35, "HDFCAMC": 4.17,
    "IOC": 4.44, "HINDZINC": 3.75, "INDHOTEL": 4.35, "JINDALSTEL": 4.35,
    "LTM": 4.26, "UNITDSPR": 4.55, "MUTHOOTFIN": 3.99, "MOTHERSON": 4.05,
    "RECLTD": 4.22, "PIDILITIND": 4.55, "PFC": 4.26, "PNB": 4.35,
    "TATAPOWER": 4.55, "SOLARINDS": 4.12, "SHREECEM": 4.55, "SIEMENS": 4.17,
    "UNIONBANK": 4.03, "VEDL": 2.38, "TVSMOTOR": 4.44, "TORNTPHARM": 4.55,
    "VBL": 4.35, "MAZDOCK": 3.57, "IRFC": 4.12, "LODHA": 3.73,
    "HYUNDAI": 4.17, "ENRIN": 3.22, "TATACAP": 3.77, "TMCV": 3.34,
}
