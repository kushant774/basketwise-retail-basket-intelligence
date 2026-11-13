"""Page registry and sidebar for the BasketWise app."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from ui import state
from ui.views import (
    association_rules,
    bundles,
    data_explorer,
    evaluation_lab,
    overview,
    product_analytics,
    recommendations,
    settings,
)

LOGO = Path(__file__).parent / "assets" / "basketwise_logo.svg"

OVERVIEW = st.Page(overview.render, title="Overview", icon=":material/dashboard:", url_path="overview", default=True)
DATA = st.Page(data_explorer.render, title="Data Explorer", icon=":material/table_view:", url_path="data")
PRODUCTS = st.Page(product_analytics.render, title="Product Analytics", icon=":material/bar_chart:", url_path="products")
RULES = st.Page(association_rules.render, title="Association Rules", icon=":material/hub:", url_path="rules")
RECOMMEND = st.Page(recommendations.render, title="Smart Recommendations", icon=":material/shopping_cart:",
                    url_path="recommendations")
BUNDLES = st.Page(bundles.render, title="Bundle Opportunities", icon=":material/inventory_2:", url_path="bundles")
EVALUATION = st.Page(evaluation_lab.render, title="Evaluation Lab", icon=":material/science:", url_path="evaluation")
SETTINGS = st.Page(settings.render, title="Settings & Data", icon=":material/settings:", url_path="settings")

PAGES = {
    "Understand": [OVERVIEW, DATA, PRODUCTS],
    "Discover": [RULES, RECOMMEND, BUNDLES],
    "Validate": [EVALUATION],
    "Manage": [SETTINGS],
}


def sidebar() -> None:
    st.logo(str(LOGO), size="large")
    data = state.data()
    config = state.config()
    result, error = state.mining()
    with st.sidebar:
        st.caption("CURRENT ANALYSIS")
        if data.is_synthetic:
            st.badge("Synthetic data", icon=":material/science:", color="blue")
        else:
            st.badge("Uploaded data", icon=":material/upload_file:", color="green")
        st.markdown(f"**{data.report.valid_transactions:,}** transactions · **{data.report.unique_products:,}** products")
        if error:
            st.markdown(":red[Settings need attention]")
        else:
            st.markdown(f"**{len(result.rules):,}** association rules")
        st.caption(f"Support ≥ {config.min_support:.1%} · confidence ≥ {config.min_confidence:.0%} · "
                   f"lift ≥ {config.min_lift:.1f} · {config.algorithm}")
