"""Bundle Opportunities: product combinations, cross-sell pairs, merchandising groups and the relationship map."""
from __future__ import annotations

import streamlit as st

from market_basket import plots
from market_basket.insights import bundle_opportunities, cross_sell_opportunities, product_groups
from ui import components as ui
from ui import state


def render() -> None:
    data = state.data()
    ui.page_header("Bundle Opportunities",
                   "Product combinations bought together more often than chance, and how they could be used.")
    result, error = state.mining()
    if error:
        st.error(error, icon=":material/error:")
        return
    if result.rules.empty:
        st.info("No association rules pass the current settings, so there are no opportunities to show. Lower "
                "the thresholds on the Association Rules page.", icon=":material/info:")
        return

    top_n = st.slider("Number of suggestions", 5, 30, 10)
    st.subheader("Potential bundles")
    st.caption("Pairs and triples ranked by all-confidence: at least this share of the buyers of any one product in "
               "the set bought the whole set. Unlike lift, it is not inflated just because a set is larger.")
    bundles = bundle_opportunities(result, state.best_sellers(), data.categories, top_n)
    st.dataframe(bundles, hide_index=True, width="stretch", column_config=ui.column_config(bundles))
    st.download_button("Download bundles (CSV)", bundles.to_csv(index=False), "bundle_opportunities.csv", "text/csv",
                       icon=":material/download:")

    st.subheader("Cross-selling opportunities")
    st.caption("Strongest one-to-one product pairs by lift, each shown once in its more reliable direction. "
               "These are candidates for 'customers also bought' prompts, to be tested before rollout.")
    cross_sell = cross_sell_opportunities(result.rules, data.categories, top_n)
    st.dataframe(cross_sell, hide_index=True, width="stretch", column_config=ui.column_config(cross_sell))
    st.download_button("Download cross-sell pairs (CSV)", cross_sell.to_csv(index=False), "cross_sell.csv",
                       "text/csv", icon=":material/download:")

    st.subheader("Merchandising groups")
    min_lift = st.slider("Minimum lift for products to be linked", 1.2, 5.0, 2.0, 0.1,
                         help="Higher values give smaller, tighter groups.")
    groups = product_groups(result.rules, min_lift)
    if not groups:
        st.info("No product pairs reach this lift.", icon=":material/info:")
    columns = st.columns(3)
    for index, group in enumerate(groups):
        categories = sorted({data.categories.get(product, "Uncategorised") for product in group})
        with columns[index % 3].container(border=True):
            st.markdown(f'<div class="bw-card-title">Group {index + 1}</div>', unsafe_allow_html=True)
            ui.chips(group)
            st.caption("Categories: " + ", ".join(categories))
    if groups:
        st.caption("Potential uses: shelf adjacency, store-layout reviews or themed promotions. Groups come from "
                   "community detection on the rule graph and change with the lift threshold.")
    ui.chart(plots.rule_network(result.rules, data.categories))
