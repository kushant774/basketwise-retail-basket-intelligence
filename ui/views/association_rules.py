"""Association Rules: mining settings, the rule table, rule explanations and the rule map."""
from __future__ import annotations

import streamlit as st

from market_basket import plots
from market_basket.config import ALGORITHM_NAMES, ALGORITHMS, RULE_RANK_METRICS, MiningConfig
from market_basket.mining import as_text, explain_rule, rank_rules, readable
from ui import components as ui
from ui import state

SIZE_OPTIONS = [2, 3, 4, None]


def _settings_form(catalog: list[str]) -> None:
    current = state.config()
    with st.form("mining_settings", border=True):
        st.markdown('<div class="bw-card-title">Mining settings</div>', unsafe_allow_html=True)
        left, middle, right = st.columns(3, gap="large")
        algorithm = left.radio("Mining algorithm", ALGORITHMS, index=ALGORITHMS.index(current.algorithm),
                               format_func=ALGORITHM_NAMES.get, horizontal=True,
                               help="Both find exactly the same itemsets (both come from the mlxtend library). "
                                    "In BasketWise's benchmark both took under a second at the default settings, "
                                    "while at very low minimum support with no size limit, when there are many "
                                    "itemsets, FP-Growth was 2-7 times faster.")
        max_len = left.selectbox("Maximum products in a rule", SIZE_OPTIONS, index=SIZE_OPTIONS.index(current.max_len),
                                 format_func=lambda v: "No limit" if v is None else str(v))
        min_support = middle.slider("Minimum support", 0.002, 0.10, float(current.min_support), 0.001, format="%.3f",
                                    help=ui.METRIC_HELP["support"])
        min_confidence = middle.slider("Minimum confidence", 0.0, 1.0, float(current.min_confidence), 0.05,
                                       help=ui.METRIC_HELP["confidence"])
        min_lift = right.slider("Minimum lift", 1.0, 5.0, float(current.min_lift), 0.1, help=ui.METRIC_HELP["lift"])
        exclude = right.multiselect("Ignore products", catalog, default=[p for p in current.exclude if p in catalog],
                                    help="Leave these products out of rules, recommendations (including best-seller "
                                         "suggestions), bundles and the evaluation, for example a carrier bag. Sales "
                                         "statistics still include them.")
        submitted = st.form_submit_button("Apply settings", type="primary", icon=":material/tune:")
    if submitted:
        try:
            new_config = MiningConfig(min_support, min_confidence, min_lift, max_len, algorithm, tuple(exclude))
        except ValueError as error:
            st.error(str(error))
            return
        state.set_config(new_config)
        ui.flash("Mining settings applied.")
        # Rerun at once: the form widgets take their defaults from the settings, so they must be rebuilt
        # from the new settings before the next submission, and the sidebar (drawn before this page)
        # must show the new rule count.
        st.rerun()


def _rule_detail(rule) -> None:
    with st.container(border=True):
        title = f"{as_text(rule['antecedents'])} → {as_text(rule['consequents'])}"
        st.markdown(f'<div class="bw-card-title">{ui.escape(title)}</div>', unsafe_allow_html=True)
        st.markdown(ui.escape(explain_rule(rule["antecedents"], rule["consequents"], rule["support"],
                                           rule["confidence"], rule["lift"], rule["consequent support"])))
        a, b, c, d = st.columns(4)
        a.metric("Support", ui.pct(rule["support"], 1))
        b.metric("Confidence", ui.pct(rule["confidence"], 1))
        c.metric("Lift", f"{rule['lift']:.2f}")
        d.metric("Conviction", "∞" if rule["conviction"] == float("inf") else f"{rule['conviction']:.2f}")
        st.caption("An association describes what was bought together in this data. It does not prove that "
                   "one product causes the purchase of the other.")


def render() -> None:
    data = state.data()
    ui.page_header("Association Rules", "Find products that are bought together, and how strong each pattern is.")
    _settings_form(data.catalog)
    result, error = state.mining()
    if error:
        st.error(error, icon=":material/error:")
        return
    rules = result.rules
    ui.metric_row([
        ("Frequent itemsets", f"{len(result.frequent_itemsets):,}", "Product sets that reach the minimum support."),
        ("Association rules", f"{len(rules):,}", "Rules that pass the confidence and lift thresholds."),
        ("Highest lift", f"{rules['lift'].max():.2f}" if len(rules) else "None", None),
        ("Transactions analysed", f"{result.n_transactions:,}", None),
    ])
    ui.explain_metrics()
    if rules.empty:
        st.info("No rules pass these settings. Lower the minimum support, confidence or lift, or allow more "
                "products per rule.", icon=":material/info:")
        return

    search_column, sort_column = st.columns([3, 1])
    search = search_column.text_input("Find rules mentioning a product", placeholder="e.g. Pasta").strip().lower()
    sort_by = sort_column.selectbox("Sort by", RULE_RANK_METRICS)
    ranked = rank_rules(rules, sort_by)
    if search:
        mentions = (ranked["antecedents"].map(as_text).str.lower().str.contains(search, regex=False)
                    | ranked["consequents"].map(as_text).str.lower().str.contains(search, regex=False))
        ranked = ranked[mentions].reset_index(drop=True)
        if ranked.empty:
            st.info(f"No rule mentions “{ui.escape(search)}”. Check the spelling, try part of the name, or clear "
                    "the search to see all rules.", icon=":material/search_off:")
            return
    st.caption(f"{len(ranked):,} rules shown. Select a row to see it explained in plain words.")
    event = st.dataframe(readable(ranked), hide_index=True, width="stretch", height=380,
                         column_config=ui.RULE_COLUMNS, on_select="rerun", selection_mode="single-row",
                         key="rules_table")
    selected = event.selection.rows if event and event.selection else []
    if selected and selected[0] < len(ranked):
        _rule_detail(ranked.iloc[selected[0]])
    elif len(ranked):
        _rule_detail(ranked.iloc[0])
    st.download_button("Download these rules (CSV)", readable(ranked).to_csv(index=False), "association_rules.csv",
                       "text/csv", icon=":material/download:")
    ui.chart(plots.rules_scatter(ranked))
