"""Smart Recommendations: an interactive basket builder with explained suggestions."""
from __future__ import annotations

import streamlit as st

from market_basket.config import RECOMMENDATION_RANK_METRICS
from market_basket.mining import as_text, empty_rules
from market_basket.recommend import Recommendation, Recommender, compare_rankings, recommendations_table
from ui import components as ui
from ui import state

EXAMPLES = {"Milk + Bread": ["Milk", "Bread"], "Pasta": ["Pasta"], "Shampoo": ["Shampoo"],
            "Potato Chips": ["Potato Chips"], "Tea": ["Tea"]}
RANK_HELP = ("Confidence: the products most often bought with this basket (the default; it found the hidden "
             "product most often in the Evaluation Lab on the demo data). Lift: the most unusual pairings. "
             "Support: the most common combinations overall.")


def _use_example() -> None:
    choice = st.session_state.get("example")
    if choice:
        state.set_basket(EXAMPLES[choice])
        st.session_state.example = None


def _clear_basket() -> None:
    state.set_basket([])


def _sync_basket() -> None:
    """Copy the widget's value into the persistent basket (the widget's own value is deleted by
    Streamlit when the user opens another page)."""
    state.set_basket(st.session_state.basket_widget)


def _nothing_to_suggest(fallback: bool) -> None:
    if fallback:
        st.info("Nothing is left to suggest: no association rule applies, and every best seller is already in the "
                "basket or ignored.", icon=":material/info:")
    else:
        st.info("No association rule applies to this basket. Turn on the best-seller fallback, or lower the "
                "thresholds on the Association Rules page.", icon=":material/info:")


def _recommendation_card(position: int, rec: Recommendation) -> None:
    with st.container(border=True):
        title, badge = st.columns([3, 1])
        title.markdown(f'<div class="bw-card-title">{position}. {ui.escape(rec.product)}</div>',
                       unsafe_allow_html=True)
        with badge:
            if rec.source == "association rule":
                st.badge("Association rule", icon=":material/hub:", color="blue")
            else:
                st.badge("Best seller", icon=":material/trending_up:", color="gray")
        st.markdown(f'<div class="bw-muted">{ui.escape(rec.reason)}</div>', unsafe_allow_html=True)
        if rec.source == "association rule":
            st.caption(f"Rule: {ui.escape(as_text(rec.based_on))} → {ui.escape(rec.product)} · support "
                       f"{rec.support:.1%} · confidence {rec.confidence:.1%} · lift {rec.lift:.2f}")


def render() -> None:
    data = state.data()
    config = state.config()
    ui.page_header("Smart Recommendations",
                   "Build a basket and see which products customers often add, with the reason for each suggestion.")
    result, error = state.mining()
    rules = result.rules if result is not None else empty_rules()
    basket = state.basket()
    st.session_state.basket_widget = basket  # show the persistent basket in the widget
    left, right = st.columns([2, 3], gap="large")
    with left.container(border=True):
        st.markdown('<div class="bw-card-title">Basket</div>', unsafe_allow_html=True)
        st.multiselect("Search and add products", data.catalog, key="basket_widget", on_change=_sync_basket,
                       placeholder="Type a product name")
        examples = [name for name, items in EXAMPLES.items() if set(items) <= set(data.catalog)]
        if examples:
            st.pills("Quick examples", examples, key="example", on_change=_use_example)
        st.button("Clear basket", on_click=_clear_basket, icon=":material/delete:")
        st.divider()
        top_n = st.slider("Number of suggestions", 1, 10, 5)
        rank_by = st.radio("Rank suggestions by", RECOMMENDATION_RANK_METRICS, horizontal=True, help=RANK_HELP,
                           format_func=str.capitalize)
        fallback = st.toggle("Fill free slots with best sellers", value=True,
                             help="When fewer association rules apply than the number of suggestions requested.")

    with right:
        if error:
            st.error(f"Association rules are unavailable: {error}", icon=":material/error:")
        if not basket:
            st.info("The basket is empty. "
                    + ("Showing best sellers; add products to get suggestions based on what is bought together."
                       if fallback else "Add products to get suggestions."), icon=":material/info:")
        else:
            st.markdown("**Current basket**")
            ui.chips(basket)
        if config.exclude:
            st.caption("Ignored products are never suggested: " + ui.escape(", ".join(config.exclude)) + ".")
        recommender = Recommender(rules, rank_by, state.best_sellers() if fallback else None, config.exclude)
        recommendations = recommender.recommend(basket, top_n)
        if not recommendations and basket:
            _nothing_to_suggest(fallback)
        for position, rec in enumerate(recommendations, start=1):
            _recommendation_card(position, rec)
        if recommendations:
            st.caption("Suggestions describe what customers bought together in this data. They are not a "
                       "guarantee of additional sales.")
            with st.expander("Compare ranking methods", icon=":material/compare_arrows:"):
                st.markdown("The same basket ranked three ways. Best sellers fill slots that no rule covers.")
                st.dataframe(compare_rankings(rules, basket, state.best_sellers(), top_n, config.exclude),
                             width="stretch")
            with st.expander("Show as a table", icon=":material/table:"):
                table = recommendations_table(recommendations)
                st.dataframe(table, hide_index=True, width="stretch", column_config=ui.column_config(table))
                st.download_button("Download recommendations (CSV)", table.to_csv(index=False),
                                   "recommendations.csv", "text/csv", icon=":material/download:")
