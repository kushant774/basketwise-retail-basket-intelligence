"""Product Analytics: popularity, categories, basket sizes and purchase frequency."""
from __future__ import annotations

import streamlit as st

from market_basket import plots
from market_basket.data import summarize
from market_basket.insights import category_summary, product_popularity
from ui import components as ui
from ui import state


def _count_slider(n_products: int, default: int, key: str) -> int:
    """Slider for how many products to chart; small catalogues simply show every product."""
    if n_products <= 5:
        return n_products
    return st.slider("Number of products", 5, min(30, n_products), min(default, n_products), key=key)


def render() -> None:
    data = state.data()
    ui.page_header("Product Analytics", "Which products sell most and least, and how baskets are made up.")
    ui.data_badge(data)
    stats = summarize(data)
    popularity = product_popularity(data)
    categories = category_summary(data)
    ui.metric_row([
        ("Products", f"{stats['unique products']:,}", None),
        ("Median basket size", f"{stats['median basket size']:g}", "Half of the transactions have this many "
                                                                     "products or fewer."),
        ("Single-product transactions", ui.pct(stats["single-product share"]),
         "These baskets cannot show which products are bought together."),
        ("Best seller", popularity.iloc[0]["product"],
         f"In {popularity.iloc[0]['share of transactions']:.1%} of transactions."),
    ])

    top, least, category_tab, baskets, frequency = st.tabs(
        ["Top products", "Least purchased", "Categories", "Basket sizes", "Purchase frequency"]
    )
    with top:
        count = _count_slider(len(popularity), 15, "top_count")
        ui.chart(plots.product_bar(popularity, top_n=count))
    with least:
        count = _count_slider(len(popularity), 10, "least_count")
        ui.chart(plots.product_bar(popularity, top_n=count, least=True))
        st.caption("Rarely bought products seldom reach the minimum support, so they appear in few or no rules.")
    with category_tab:
        if categories.empty:
            st.info("The data has no category column, so category analysis is not available.",
                    icon=":material/info:")
        else:
            ui.chart(plots.category_bar(categories))
            st.dataframe(categories, hide_index=True, width="stretch", column_config=ui.column_config(categories))
    with baskets:
        ui.chart(plots.basket_size_chart(data))
        weekly = plots.weekly_transactions_chart(data)
        if weekly is not None:
            ui.chart(weekly)
    with frequency:
        search = st.text_input("Search products", placeholder="Type part of a product name").strip().lower()
        table = popularity[popularity["product"].str.lower().str.contains(search, regex=False)] if search else popularity
        st.dataframe(table, hide_index=True, width="stretch", height=420, column_config=ui.column_config(table))
        st.download_button("Download product frequency (CSV)", popularity.to_csv(index=False),
                           "product_frequency.csv", "text/csv", icon=":material/download:")
