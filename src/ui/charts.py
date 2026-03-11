"""
ui/charts.py
────────────
Plotly chart factories. Each function takes a DataFrame and returns a go.Figure.
Uses order_processing_time, shipping_delay, and total_lead_time columns.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

from analytics import (
    avg_processing_time_per_warehouse,
    avg_shipping_delay_per_warehouse,
    avg_shipping_delay_per_product,
    avg_processing_time_per_product,
    top_delayed_destinations,
    monthly_order_volume,
    delay_distribution_per_warehouse,
)

# ── Shared theme ──────────────────────────────────────────────────────────────
_BG      = "#111827"
_SURFACE = "#0d1117"
_ACCENT  = "#4f46e5"
_CYAN    = "#0ea5e9"
_AMBER   = "#f59e0b"
_GREEN   = "#10b981"
_TEXT    = "#f1f5f9"
_MUTED   = "#64748b"
_GRID    = "#1f2937"

_LAYOUT = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG,
    font=dict(color=_TEXT, family="Inter, sans-serif", size=12),
    margin=dict(l=48, r=24, t=48, b=40),
    xaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(color=_MUTED)),
    yaxis=dict(gridcolor=_GRID, zerolinecolor=_GRID, tickfont=dict(color=_MUTED)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_MUTED)),
)


def _title(text: str) -> dict:
    return dict(text=text, font=dict(size=14, color=_TEXT), x=0, xanchor="left", pad=dict(l=4))


# ── Charts ────────────────────────────────────────────────────────────────────

def chart_processing_vs_shipping_per_warehouse(df: pd.DataFrame) -> go.Figure:
    """Grouped bar: order processing time vs shipping delay per warehouse."""
    proc = avg_processing_time_per_warehouse(df).set_index("Warehouse")
    ship = avg_shipping_delay_per_warehouse(df).set_index("Warehouse")
    warehouses = proc.index.tolist()

    fig = go.Figure([
        go.Bar(
            name="Order Processing Time",
            x=warehouses,
            y=proc["avg_processing_time"].values,
            marker_color=_ACCENT,
            text=[f"{v}d" for v in proc["avg_processing_time"].values],
            textposition="outside",
            textfont=dict(size=11, color=_TEXT),
        ),
        go.Bar(
            name="Shipping Delay",
            x=warehouses,
            y=ship["avg_shipping_delay"].values,
            marker_color=_CYAN,
            text=[f"{v}d" for v in ship["avg_shipping_delay"].values],
            textposition="outside",
            textfont=dict(size=11, color=_TEXT),
        ),
    ])
    fig.update_layout(
        **_LAYOUT,
        title=_title("Order Processing Time vs Shipping Delay by Warehouse"),
        barmode="group",
        xaxis_title="Warehouse",
        yaxis_title="Days",
    )
    return fig


def chart_processing_vs_shipping_per_product(df: pd.DataFrame) -> go.Figure:
    """Horizontal grouped bar: processing time vs shipping delay per product."""
    proc = avg_processing_time_per_product(df).set_index("Product")
    ship = avg_shipping_delay_per_product(df).set_index("Product")
    products = proc.index.tolist()

    fig = go.Figure([
        go.Bar(
            name="Order Processing Time",
            y=products,
            x=proc["avg_processing_time"].values,
            orientation="h",
            marker_color=_ACCENT,
            text=[f"{v}d" for v in proc["avg_processing_time"].values],
            textposition="outside",
            textfont=dict(size=11, color=_TEXT),
        ),
        go.Bar(
            name="Shipping Delay",
            y=products,
            x=ship["avg_shipping_delay"].values,
            orientation="h",
            marker_color=_CYAN,
            text=[f"{v}d" for v in ship["avg_shipping_delay"].values],
            textposition="outside",
            textfont=dict(size=11, color=_TEXT),
        ),
    ])
    fig.update_layout(
        **_LAYOUT,
        title=_title("Order Processing Time vs Shipping Delay by Product"),
        barmode="group",
        xaxis_title="Days",
        yaxis_title="",
    )
    return fig


def chart_delay_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of avg shipping delay: Warehouse × Product."""
    pivot = (
        df.pivot_table(
            values="shipping_delay",
            index="Warehouse",
            columns="Product",
            aggfunc="mean",
        ).round(1)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Blues",
        text=pivot.values,
        texttemplate="%{text}d",
        hovertemplate="Warehouse: %{y}<br>Product: %{x}<br>Shipping Delay: %{z}d<extra></extra>",
        colorbar=dict(title="Days", tickfont=dict(color=_MUTED)),
    ))
    fig.update_layout(
        **_LAYOUT,
        title=_title("Shipping Delay Heatmap — Warehouse × Product"),
    )
    return fig


def chart_processing_heatmap(df: pd.DataFrame) -> go.Figure:
    """Heatmap of avg order processing time: Warehouse × Product."""
    pivot = (
        df.pivot_table(
            values="order_processing_time",
            index="Warehouse",
            columns="Product",
            aggfunc="mean",
        ).round(1)
    )
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale="Purples",
        text=pivot.values,
        texttemplate="%{text}d",
        hovertemplate="Warehouse: %{y}<br>Product: %{x}<br>Processing Time: %{z}d<extra></extra>",
        colorbar=dict(title="Days", tickfont=dict(color=_MUTED)),
    ))
    fig.update_layout(
        **_LAYOUT,
        title=_title("Order Processing Time Heatmap — Warehouse × Product"),
    )
    return fig


def chart_monthly_volume(df: pd.DataFrame) -> go.Figure:
    data = monthly_order_volume(df)
    fig = go.Figure(go.Scatter(
        x=data["month"],
        y=data["order_count"],
        mode="lines+markers",
        line=dict(color=_ACCENT, width=2.5),
        marker=dict(color=_CYAN, size=7),
        fill="tozeroy",
        fillcolor="rgba(79,70,229,0.1)",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=_title("Monthly Order Volume"),
        xaxis_title="Month",
        yaxis_title="Orders",
    )
    return fig


def chart_top_delayed_destinations(df: pd.DataFrame) -> go.Figure:
    data = top_delayed_destinations(df)
    fig = go.Figure(go.Bar(
        x=data["avg_shipping_delay"],
        y=data["Destination"],
        orientation="h",
        marker=dict(
            color=data["avg_shipping_delay"],
            colorscale="Oranges",
            showscale=True,
            colorbar=dict(title="Days", tickfont=dict(color=_MUTED)),
        ),
        hovertemplate="%{y}: %{x}d<extra></extra>",
    ))
    fig.update_layout(
        **_LAYOUT,
        title=_title("Top Destinations by Shipping Delay"),
        xaxis_title="Avg Shipping Delay (days)",
        yaxis_title="",
    )
    return fig


def chart_warehouse_breakdown(df: pd.DataFrame) -> go.Figure:
    """Stacked bar: processing time + shipping delay = total lead time per warehouse."""
    dist = delay_distribution_per_warehouse(df).sort_values("avg_total_lead_time", ascending=False)

    fig = go.Figure([
        go.Bar(
            name="Order Processing Time",
            x=dist["Warehouse"],
            y=dist["avg_processing_time"],
            marker_color=_ACCENT,
        ),
        go.Bar(
            name="Shipping Delay",
            x=dist["Warehouse"],
            y=dist["avg_shipping_delay"],
            marker_color=_CYAN,
        ),
    ])
    fig.update_layout(
        **_LAYOUT,
        title=_title("Total Lead Time Breakdown by Warehouse"),
        barmode="stack",
        xaxis_title="Warehouse",
        yaxis_title="Days",
    )
    return fig


# ── Intent detection ──────────────────────────────────────────────────────────

_INTENT_MAP: dict[str, list[str]] = {
    "warehouse_breakdown":   ["warehouse", "processing", "which warehouse", "delay by warehouse"],
    "product_breakdown":     ["product", "fastest", "slowest", "ship product"],
    "heatmap_shipping":      ["heatmap", "matrix", "breakdown", "shipping delay heat"],
    "heatmap_processing":    ["processing heatmap", "processing heat", "processing matrix"],
    "monthly":               ["monthly", "volume", "trend", "over time"],
    "destinations":          ["destination", "city", "location", "where"],
    "lead_time":             ["lead time", "total time", "stacked", "full breakdown"],
}


def detect_chart_intent(question: str) -> str | None:
    q = question.lower()
    for chart_key, keywords in _INTENT_MAP.items():
        if any(kw in q for kw in keywords):
            return chart_key
    return None


def get_chart_for_intent(intent: str, df: pd.DataFrame) -> go.Figure | None:
    dispatch = {
        "warehouse_breakdown":  chart_processing_vs_shipping_per_warehouse,
        "product_breakdown":    chart_processing_vs_shipping_per_product,
        "heatmap_shipping":     chart_delay_heatmap,
        "heatmap_processing":   chart_processing_heatmap,
        "monthly":              chart_monthly_volume,
        "destinations":         chart_top_delayed_destinations,
        "lead_time":            chart_warehouse_breakdown,
    }
    fn = dispatch.get(intent)
    return fn(df) if fn else None
