"""
analytics.py
────────────
Pure analytics functions — no Streamlit, no side-effects.
Works with the three computed columns:
  - order_processing_time  (Order_Date → Dispatch_Date)
  - shipping_delay         (Dispatch_Date → Ship_Date)
  - total_lead_time        (Order_Date → Ship_Date)
"""

from __future__ import annotations

import pandas as pd
from config import DELAY_THRESHOLD_DAYS, TOP_DESTINATIONS_N




def avg_processing_time_per_warehouse(df: pd.DataFrame) -> pd.DataFrame:
    """Avg order processing time per warehouse, descending."""
    return (
        df.groupby("Warehouse")["order_processing_time"]
        .mean().round(2).reset_index()
        .rename(columns={"order_processing_time": "avg_processing_time"})
        .sort_values("avg_processing_time", ascending=False)
        .reset_index(drop=True)
    )


def avg_shipping_delay_per_warehouse(df: pd.DataFrame) -> pd.DataFrame:
    """Avg shipping delay per warehouse, descending."""
    return (
        df.groupby("Warehouse")["shipping_delay"]
        .mean().round(2).reset_index()
        .rename(columns={"shipping_delay": "avg_shipping_delay"})
        .sort_values("avg_shipping_delay", ascending=False)
        .reset_index(drop=True)
    )


def avg_delay_per_warehouse(df: pd.DataFrame) -> pd.DataFrame:
    """Combined avg processing + shipping per warehouse (for backwards compat)."""
    proc = avg_processing_time_per_warehouse(df).set_index("Warehouse")
    ship = avg_shipping_delay_per_warehouse(df).set_index("Warehouse")
    combined = proc.join(ship)
    combined["avg_total_lead_time"] = (combined["avg_processing_time"] + combined["avg_shipping_delay"]).round(2)
    return combined.reset_index().sort_values("avg_total_lead_time", ascending=False).reset_index(drop=True)



def avg_processing_time_per_product(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Product")["order_processing_time"]
        .mean().round(2).reset_index()
        .rename(columns={"order_processing_time": "avg_processing_time"})
        .sort_values("avg_processing_time")
        .reset_index(drop=True)
    )


def avg_shipping_delay_per_product(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Product")["shipping_delay"]
        .mean().round(2).reset_index()
        .rename(columns={"shipping_delay": "avg_shipping_delay"})
        .sort_values("avg_shipping_delay")
        .reset_index(drop=True)
    )


def fastest_shipping_product(df: pd.DataFrame) -> pd.DataFrame:
    """Products ranked by avg shipping delay (fastest first)."""
    return avg_shipping_delay_per_product(df)


def orders_delayed_more_than_n(df: pd.DataFrame, n: int = DELAY_THRESHOLD_DAYS) -> pd.DataFrame:
    """Orders where shipping_delay > n days."""
    return df[df["shipping_delay"] > n].copy()


def top_delayed_destinations(df: pd.DataFrame, n: int = TOP_DESTINATIONS_N) -> pd.DataFrame:
    return (
        df.groupby("Destination")["shipping_delay"]
        .mean().round(2).reset_index()
        .rename(columns={"shipping_delay": "avg_shipping_delay"})
        .sort_values("avg_shipping_delay", ascending=False)
        .head(n).reset_index(drop=True)
    )


def monthly_order_volume(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["month"] = tmp["Order_Date"].dt.to_period("M").astype(str)
    return (
        tmp.groupby("month").size()
        .reset_index(name="order_count")
        .sort_values("month").reset_index(drop=True)
    )


def delay_distribution_per_warehouse(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("Warehouse").agg(
            avg_processing_time=("order_processing_time", "mean"),
            avg_shipping_delay=("shipping_delay", "mean"),
            avg_total_lead_time=("total_lead_time", "mean"),
            total_orders=("Order_ID", "count"),
        ).round(2).reset_index()
    )



def generate_auto_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    total = len(df)

    proc_wh = avg_processing_time_per_warehouse(df)
    insights.append(
        f"🔴 **{proc_wh.iloc[0]['Warehouse']}** has the slowest order processing: "
        f"**{proc_wh.iloc[0]['avg_processing_time']} days** avg."
    )
    insights.append(
        f"🟢 **{proc_wh.iloc[-1]['Warehouse']}** processes orders fastest: "
        f"**{proc_wh.iloc[-1]['avg_processing_time']} days** avg."
    )

    ship_wh = avg_shipping_delay_per_warehouse(df)
    insights.append(
        f"🚚 **{ship_wh.iloc[0]['Warehouse']}** has the highest shipping delay: "
        f"**{ship_wh.iloc[0]['avg_shipping_delay']} days** avg."
    )

    prod = fastest_shipping_product(df)
    insights.append(
        f"⚡ **{prod.iloc[0]['Product']}** ships fastest: **{prod.iloc[0]['avg_shipping_delay']} days** avg shipping delay."
    )
    insights.append(
        f"🐢 **{prod.iloc[-1]['Product']}** ships slowest: **{prod.iloc[-1]['avg_shipping_delay']} days** avg shipping delay."
    )

    delayed = orders_delayed_more_than_n(df)
    pct = round(len(delayed) / total * 100, 1)
    insights.append(
        f"⚠️ **{len(delayed)} orders ({pct}%)** had a shipping delay exceeding {DELAY_THRESHOLD_DAYS} days."
    )


    dest = top_delayed_destinations(df, 1).iloc[0]
    insights.append(
        f"📍 **{dest['Destination']}** is the most delayed destination: "
        f"**{dest['avg_shipping_delay']} days** avg shipping delay."
    )

    avg_proc = round(float(df["order_processing_time"].mean()), 1)
    avg_ship = round(float(df["shipping_delay"].mean()), 1)
    insights.append(
        f"📊 Overall averages — Processing: **{avg_proc}d**, Shipping: **{avg_ship}d**, "
        f"Total lead time: **{round(avg_proc + avg_ship, 1)}d**."
    )

    return insights
