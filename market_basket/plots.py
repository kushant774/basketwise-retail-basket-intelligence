"""Interactive Plotly charts used by the app, the notebook and the report.

Colour roles: one blue for single-series magnitudes, a sequential blue ramp for lift, and a
fixed categorical order for product categories (a category keeps its colour on every chart).
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data import TransactionData, transaction_dates
from .insights import cross_sell_opportunities

BLUE = "#2a78d6"
ORANGE = "#eb6834"
TEXT = "#1f2328"
MUTED = "#5b5f66"
GRID = "#e6e7ea"
SEQUENTIAL = [[0.0, "#cde2fb"], [0.25, "#86b6ef"], [0.5, "#3987e5"], [0.75, "#256abf"], [1.0, "#0d366b"]]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
NEUTRAL = "#9aa0a6"


def short(name: str, limit: int = 30) -> str:
    """Shorten long product names for axis labels; hover text keeps the full name."""
    name = str(name)
    return name if len(name) <= limit else name[: limit - 1] + "…"


def category_colors(categories) -> dict[str, str]:
    """Stable colour per category: sorted names take the categorical slots in order."""
    names = sorted(set(categories))
    return {name: CATEGORICAL[i] if i < len(CATEGORICAL) else NEUTRAL for i, name in enumerate(names)}


def _style(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15, color=TEXT)) if title else None,
        height=height,
        margin=dict(l=8, r=8, t=48 if title else 12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, system-ui, sans-serif", size=12, color=MUTED),
        hoverlabel=dict(bgcolor="white", font_size=12, font_color=TEXT),
        legend=dict(orientation="h", yanchor="top", y=-0.16, xanchor="left", x=0, title_text=""),
    )
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True, title_standoff=8)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID, automargin=True, title_standoff=8)
    return fig


def product_bar(popularity: pd.DataFrame, top_n: int = 15, least: bool = False, title: str | None = None) -> go.Figure:
    """Horizontal bars of the share of transactions per product, coloured by category when known."""
    table = popularity.sort_values(["transactions", "product"], ascending=[least, True]).head(top_n)
    table = table.iloc[::-1]
    fig = go.Figure()
    if "category" in table:
        colors = category_colors(popularity["category"])
        for category in sorted(table["category"].unique()):
            part = table[table["category"] == category]
            fig.add_bar(y=part["product"].map(short), x=part["share of transactions"] * 100, orientation="h",
                        name=category, marker_color=colors[category],
                        customdata=part[["product", "transactions"]],
                        hovertemplate="%{customdata[0]}: %{x:.1f}% of transactions (%{customdata[1]:,})"
                                      "<extra></extra>")
        fig.update_layout(barmode="relative")
        fig.update_yaxes(categoryorder="array", categoryarray=list(table["product"].map(short)))
    else:
        fig.add_bar(y=table["product"].map(short), x=table["share of transactions"] * 100, orientation="h",
                    marker_color=BLUE, customdata=table["product"],
                    hovertemplate="%{customdata}: %{x:.1f}% of transactions<extra></extra>")
    fig.update_xaxes(title_text="Share of transactions (%)")
    default = f"{'Least' if least else 'Most'} purchased products"
    return _style(fig, title or default, height=max(260, 26 * len(table) + 90))


def category_bar(categories: pd.DataFrame) -> go.Figure:
    table = categories.sort_values("transactions")
    colors = category_colors(categories["category"])
    fig = go.Figure(go.Bar(
        y=table["category"], x=table["share of transactions"] * 100, orientation="h",
        marker_color=[colors[c] for c in table["category"]], customdata=table[["products", "transactions"]],
        hovertemplate="%{y}: in %{x:.1f}% of transactions<br>%{customdata[0]} products, "
                      "%{customdata[1]:,} transactions<extra></extra>",
    ))
    fig.update_xaxes(title_text="Share of transactions containing the category (%)")
    return _style(fig, "Category reach", height=max(260, 34 * len(table) + 90))


def basket_size_chart(data: TransactionData) -> go.Figure:
    sizes = data.items.groupby("transaction_id").size().value_counts().sort_index()
    fig = go.Figure(go.Bar(x=sizes.index, y=sizes.values, marker_color=BLUE,
                           hovertemplate="%{x} products: %{y:,} transactions<extra></extra>"))
    fig.update_xaxes(title_text="Products in the transaction", dtick=1)
    fig.update_yaxes(title_text="Transactions")
    return _style(fig, "Basket size distribution")


def weekly_transactions_chart(data: TransactionData) -> go.Figure | None:
    if "transaction_date" not in data.items:
        return None
    dates = transaction_dates(data.items.drop_duplicates("transaction_id")).dropna()
    if dates.empty:
        return None
    weekly = dates.dt.to_period("W").dt.start_time.value_counts().sort_index()
    fig = go.Figure(go.Scatter(x=weekly.index, y=weekly.values, mode="lines+markers", line=dict(color=BLUE, width=2),
                               marker=dict(size=6), hovertemplate="Week of %{x|%d %b %Y}: %{y:,} transactions<extra></extra>"))
    fig.update_yaxes(title_text="Transactions", rangemode="tozero")
    return _style(fig, "Transactions per week")


def rules_scatter(rules: pd.DataFrame) -> go.Figure:
    labels = rules["antecedents"].map(lambda s: ", ".join(sorted(s))) + " → " + \
        rules["consequents"].map(lambda s: ", ".join(sorted(s)))
    fig = go.Figure(go.Scatter(
        x=rules["support"] * 100, y=rules["confidence"] * 100, mode="markers", text=labels,
        marker=dict(size=10, color=rules["lift"], colorscale=SEQUENTIAL, line=dict(color="white", width=1),
                    colorbar=dict(title=dict(text="Lift"), thickness=12)),
        customdata=rules["lift"],
        hovertemplate="%{text}<br>support %{x:.2f}% · confidence %{y:.1f}% · lift %{customdata:.2f}<extra></extra>",
    ))
    fig.update_xaxes(title_text="Support (% of transactions)")
    fig.update_yaxes(title_text="Confidence (%)")
    return _style(fig, f"{len(rules):,} rules: support vs confidence", height=420)


def _grid_layout(graph: nx.Graph) -> dict[str, np.ndarray]:
    """Lay out each connected group of products separately and place the groups in a grid,
    so unrelated clusters do not squeeze each other together."""
    groups = sorted(nx.connected_components(graph), key=lambda nodes: (-len(nodes), sorted(nodes)))
    columns = int(np.ceil(np.sqrt(len(groups))))
    positions = {}
    for index, nodes in enumerate(groups):
        sub = graph.subgraph(sorted(nodes))
        local = nx.spring_layout(sub, seed=7, iterations=200) if len(nodes) > 2 else \
            {node: np.array([i - (len(nodes) - 1) / 2, 0.0]) for i, node in enumerate(sorted(nodes))}
        points = np.array(list(local.values()), dtype=float)
        points = points - points.mean(axis=0)
        scale = np.abs(points).max() or 1.0
        offset = np.array([index % columns, -(index // columns)], dtype=float)
        for node, point in zip(local, points, strict=True):
            positions[node] = offset + 0.36 * point / scale
    return positions


def rule_network(rules: pd.DataFrame, categories: dict[str, str] | None = None, top_n: int = 30) -> go.Figure:
    """Products linked by their strongest cross-sell pairs; thicker, darker links mean higher lift."""
    pairs = cross_sell_opportunities(rules, categories, top_n=top_n)
    fig = go.Figure()
    if pairs.empty:
        return _style(fig, "Product relationships (no pairs at these thresholds)")
    graph = nx.Graph()
    for a, b, lift in zip(pairs["when a basket has"], pairs["suggest"], pairs["lift"], strict=True):
        graph.add_edge(a, b, lift=float(lift))
    positions = _grid_layout(graph)
    lifts = pairs["lift"].to_numpy()
    low, span = lifts.min(), max(np.ptp(lifts), 1e-9)
    for a, b, data in graph.edges(data=True):
        strength = (data["lift"] - low) / span
        fig.add_scatter(x=[positions[a][0], positions[b][0]], y=[positions[a][1], positions[b][1]], mode="lines",
                        line=dict(width=1.5 + 4 * strength, color=f"rgba(37,106,191,{0.35 + 0.55 * strength:.2f})"),
                        hoverinfo="skip", showlegend=False)
    categories = categories or {}
    colors = category_colors(categories.values()) if categories else {}
    nodes = list(graph.nodes)
    groups = {}
    for node in nodes:
        groups.setdefault(categories.get(node, "Products"), []).append(node)
    for group in sorted(groups):
        members = groups[group]
        fig.add_scatter(
            x=[positions[n][0] for n in members], y=[positions[n][1] for n in members], mode="markers+text",
            text=[short(n, 22) for n in members], textposition="top center", name=group, customdata=members,
            marker=dict(size=16, color=colors.get(group, BLUE), line=dict(color="white", width=2)),
            hovertemplate="%{customdata}<extra>" + group + "</extra>",
        )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return _style(fig, f"Strongest product relationships (top {len(pairs)} pairs by lift)", height=520)


def evaluation_chart(summary: pd.DataFrame, ks: tuple[int, ...] = (3, 5)) -> go.Figure:
    fig = go.Figure()
    for k, color in zip(ks, (BLUE, ORANGE)):  # noqa: B905 - at most two cut-offs are charted
        std = summary.get(f"hit rate@{k} std", pd.Series(0, index=summary.index)).fillna(0)
        fig.add_bar(x=summary["method"], y=summary[f"hit rate@{k}"] * 100, name=f"Hit Rate@{k}", marker_color=color,
                    error_y=dict(type="data", array=std * 100, color=MUTED, thickness=1.2),
                    text=[f"{v:.1%}" for v in summary[f"hit rate@{k}"]], textposition="outside",
                    hovertemplate="%{x}<br>" + f"Hit Rate@{k}" + ": %{y:.1f}%<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.25)
    fig.update_yaxes(title_text="Hidden product found in top k (%)", rangemode="tozero")
    return _style(fig, "Hit Rate by method (mean over splits, bars show ±1 standard deviation)", height=430)
