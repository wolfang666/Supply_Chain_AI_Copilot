"""
data_processing.py
──────────────────
Handles loading, validating, and transforming the supply chain CSV dataset.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

from config import SAMPLE_DATASET, DELAY_THRESHOLD_DAYS

REQUIRED_COLUMNS = [
    "Order_ID", "Warehouse", "Product",
    "Order_Date", "Dispatch_Date", "Ship_Date", "Destination",
]


def load_dataset(filepath: str | Path | None = None) -> pd.DataFrame:
    path = Path(filepath) if filepath else SAMPLE_DATASET
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return clean_dataset(pd.read_csv(path))


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate columns, parse dates, and compute:
      - order_processing_time  = Dispatch_Date − Order_Date   (days)
      - shipping_delay         = Ship_Date − Dispatch_Date    (days)
      - total_lead_time        = Ship_Date − Order_Date       (days)
    """
    _validate_columns(df)
    df = df.copy()

    for col in ("Order_Date", "Dispatch_Date", "Ship_Date"):
        df[col] = pd.to_datetime(df[col], errors="coerce")

    df = df.dropna(subset=["Order_Date", "Dispatch_Date", "Ship_Date"])

    df["order_processing_time"] = (df["Dispatch_Date"] - df["Order_Date"]).dt.days
    df["shipping_delay"]        = (df["Ship_Date"]     - df["Dispatch_Date"]).dt.days
    df["total_lead_time"]       = (df["Ship_Date"]     - df["Order_Date"]).dt.days

    df = df[
        (df["order_processing_time"] >= 0) &
        (df["shipping_delay"]        >= 0) &
        (df["total_lead_time"]       >= 0)
    ].reset_index(drop=True)

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    total      = len(df)
    delayed    = int((df["shipping_delay"] > DELAY_THRESHOLD_DAYS).sum())
    delay_rate = round(delayed / total * 100, 1) if total else 0.0

    return {
        "total_orders":               total,
        "delayed_orders":             delayed,
        "delay_rate_pct":             delay_rate,
        "avg_processing_time":        round(float(df["order_processing_time"].mean()), 2),
        "max_processing_time":        int(df["order_processing_time"].max()),
        "avg_shipping_delay":         round(float(df["shipping_delay"].mean()), 2),
        "max_shipping_delay":         int(df["shipping_delay"].max()),
        "avg_total_lead_time":        round(float(df["total_lead_time"].mean()), 2),
        "max_total_lead_time":        int(df["total_lead_time"].max()),
        "warehouses":                 int(df["Warehouse"].nunique()),
        "products":                   int(df["Product"].nunique()),
        "destinations":               int(df["Destination"].nunique()),
        "date_range_start":           str(df["Order_Date"].min().date()),
        "date_range_end":             str(df["Order_Date"].max().date()),
    }


def _validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Expected: {REQUIRED_COLUMNS}"
        )
