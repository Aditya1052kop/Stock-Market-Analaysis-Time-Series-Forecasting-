"""
utils/data_loader.py
Handles loading, cleaning, and preprocessing of NSE stock CSV files.
"""

import pandas as pd
import os


# Map sidebar display names → actual CSV file prefixes
STOCK_FILE_MAP = {
    "TCS":        "Quote-Equity-TCS-EQ",
    "INFY":       "Quote-Equity-INFY-EQ",
    "HCLTECH":    "Quote-Equity-HCLTECH-EQ",
    "WIPRO":      "Quote-Equity-WIPRO-EQ",
    "TECHM":      "Quote-Equity-TECHM-EQ",
    "RELIANCE":   "Quote-Equity-RELIANCE-EQ",
    "ONGC":       "Quote-Equity-ONGC-EQ",
    "IOC":        "Quote-Equity-IOC-EQ",
    "BPCL":       "Quote-Equity-BPCL-EQ",
    "HINDPETRO":  "Quote-Equity-HINDPETRO-EQ",
}

DATA_DIR = "data"


def _clean_numeric(series: pd.Series) -> pd.Series:
    """Remove commas and convert to float."""
    return (
        series.astype(str)
              .str.replace(",", "", regex=False)
              .str.strip()
              .pipe(pd.to_numeric, errors="coerce")
    )


def load_data(stock: str) -> pd.DataFrame | None:
    """
    Load and clean a stock CSV from the data/ folder.

    Parameters
    ----------
    stock : str
        Sidebar display name, e.g. 'TCS', 'INFY'.

    Returns
    -------
    pd.DataFrame or None
    """
    file_prefix = STOCK_FILE_MAP.get(stock, stock)

    # Find the matching file (filename contains the prefix)
    matched = [
        f for f in os.listdir(DATA_DIR)
        if f.startswith(file_prefix) and f.endswith(".csv")
    ]

    if not matched:
        return None

    filepath = os.path.join(DATA_DIR, matched[0])

    try:
        df = pd.read_csv(filepath, encoding="utf-8-sig")  # handles BOM
    except Exception:
        return None

    # Standardise column names
    df.columns = [c.strip().upper() for c in df.columns]

    # Parse date
    date_col = next((c for c in df.columns if "DATE" in c), None)
    if date_col is None:
        return None

    df[date_col] = pd.to_datetime(df[date_col], dayfirst=True, errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.rename(columns={date_col: "Date"})
    df = df.set_index("Date").sort_index()

    # Clean numeric columns
    numeric_cols = ["OPEN", "HIGH", "LOW", "CLOSE", "VWAP", "VOLUME",
                    "PREV. CLOSE", "LTP"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _clean_numeric(df[col])

    # Rename for convenience
    rename_map = {
        "OPEN":  "Open",
        "HIGH":  "High",
        "LOW":   "Low",
        "CLOSE": "Close",
        "VWAP":  "VWAP",
        "VOLUME":"Volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Drop rows where Close is missing
    df = df.dropna(subset=["Close"])

    # Forward-fill any remaining gaps
    df = df.ffill()

    return df


def load_multiple(stocks: list[str]) -> dict[str, pd.DataFrame]:
    """Load several stocks at once; skips ones that fail."""
    result = {}
    for s in stocks:
        df = load_data(s)
        if df is not None:
            result[s] = df
    return result
