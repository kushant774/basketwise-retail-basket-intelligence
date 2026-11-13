"""Evaluation Lab: offline comparison of recommendation methods against a best-seller baseline."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from market_basket import plots
from market_basket.config import ALGORITHM_NAMES, MiningConfig
from market_basket.evaluation import METHODS
from ui import components as ui
from ui import state

BASELINE = METHODS[0]

METHOD_NOTES = {
    "Best-seller baseline": "Recommends the most popular products that are not already in the basket.",
    "Rules ranked by confidence": "Association rules only, strongest confidence first.",
    "Rules ranked by lift": "Association rules only, highest lift first.",
    "Rules by confidence + best-seller fallback": "Confidence-ranked rules; best sellers fill any free slots.",
}


def _methodology() -> None:
    with st.expander("How the evaluation works", icon=":material/menu_book:"):
        st.markdown(
            "1. The transactions are shuffled with a fixed random seed and split into a training set and a test set.\n"
            "2. Association rules and best sellers are calculated from the **training set only**, so nothing from "
            "the test set is used to make recommendations.\n"
            "3. For every test transaction with at least two products, each product is **hidden in turn**. The "
            "remaining products form the basket, and the hidden product is the answer we hope to recommend.\n"
            "4. **Hit Rate@k** is the share of these cases where the hidden product appears in the top k "
            "suggestions. **Availability** is the share of cases with at least one suggestion. **Coverage** is "
            "the share of products that a method ever recommends.\n"
            "5. This is repeated on several random splits; the table shows the average and the spread.\n"
            "6. Every method uses the same mining settings, and products you chose to ignore are treated the same "
            "way by every method (never recommended and never hidden).\n"
            "7. Three recommendation methods are compared with the best-seller baseline. Within one run, the test "
            "transactions are never used to build recommendations. However, BasketWise's default settings and "
            "ranking (confidence) were chosen after looking at results like these on the demo data, so the "
            "default method's lead may be slightly optimistic. This selection effect is small compared with the "
            "gap to the baseline on the demo data, but on new data treat the comparison as a first check, not proof."
        )
        for method, note in METHOD_NOTES.items():
            st.markdown(f"- **{method}**: {note}")


def _limitations(is_synthetic: bool) -> None:
    st.subheader("Limitations")
    notes = [
        "Offline hit rate measures whether suggestions match products customers bought anyway. It does not "
        "measure extra sales caused by the suggestion.",
        "Proving business impact needs a controlled experiment (an A/B test) that compares sales with and "
        "without recommendations.",
        "Transactions are split at random, so the test does not check how well rules hold up as preferences "
        "change over time.",
    ]
    if is_synthetic:
        notes.insert(0, "This dataset is synthetic and was generated with deliberately planted product "
                        "associations, so association rules are expected to do well here. Real retail data is "
                        "noisier, and the gap to the baseline may be much smaller or reversed.")
    for note in notes:
        st.markdown(f"- {note}")


def _settings_text(config: MiningConfig) -> str:
    text = (f"support ≥ {config.min_support:.1%}, confidence ≥ {config.min_confidence:.0%}, lift ≥ "
            f"{config.min_lift:.1f}, at most {config.max_len or 'any number of'} products per rule, "
            f"{ALGORITHM_NAMES[config.algorithm]}")
    if config.exclude:
        text += f", ignoring {', '.join(config.exclude)}"
    return text


def render() -> None:
    data = state.data()
    ui.page_header("Evaluation Lab", "Do the recommendations find what customers actually bought? "
                                     "Three methods compared against simply recommending best sellers.")
    _methodology()
    config = state.config()
    with st.form("evaluation_settings", border=True):
        left, middle, right = st.columns(3, gap="large")
        n_splits = left.slider("Random splits", 1, 10, 5, help="More splits give a more reliable average.")
        test_size = middle.select_slider("Test share", [0.1, 0.2, 0.3, 0.4], value=0.2, format_func=ui.pct)
        seed = right.number_input("Random seed", 0, 99_999, 42)
        st.caption(f"Uses the current mining settings: {_settings_text(config)}.")
        run = st.form_submit_button("Run evaluation", type="primary", icon=":material/play_arrow:")
    if run:
        try:
            with st.spinner("Mining rules on each training split and scoring the test baskets..."):
                state.run_evaluation(n_splits, test_size, int(seed))
        except ValueError as error:
            st.error(f"The evaluation could not run: {error}", icon=":material/error:")

    result, stale = state.latest_evaluation()
    if result is None:
        st.info("Run the evaluation to compare the methods.", icon=":material/info:")
        _limitations(data.is_synthetic)
        return
    if stale:
        st.warning("The data or mining settings changed after this evaluation. Run it again to update the results.",
                   icon=":material/warning:")
    used = result.settings
    st.caption(f"Result shown: {used['n_splits']} random split{'s' if used['n_splits'] != 1 else ''}, "
               f"{used['test_size']:.0%} of transactions held out, seeds {used['seed']}–"
               f"{used['seed'] + used['n_splits'] - 1}, {used['n_transactions']:,} transactions; "
               f"{ui.escape(_settings_text(used['config']))}.")

    summary = result.summary.set_index("method")
    baseline = summary.loc[BASELINE]
    challengers = summary.drop(index=BASELINE)
    best_name = challengers["hit rate@3"].idxmax()
    best = challengers.loc[best_name]
    n = result.settings["n_splits"]
    ui.metric_row([
        ("Best-seller Hit Rate@3", ui.pct(baseline["hit rate@3"], 1), METHOD_NOTES.get(BASELINE)),
        ("Best method Hit Rate@3", ui.pct(best["hit rate@3"], 1), best_name),
        ("Difference@3", f"{(best['hit rate@3'] - baseline['hit rate@3']) * 100:+.1f} pts",
         "Percentage points versus the best-seller baseline."),
        ("Cases per split (average)", f"{int(baseline['cases']):,}",
         "Hidden-product cases in each test set, averaged over the splits."),
    ])
    wins = int(best["splits beating baseline@3"])
    if best["hit rate@3"] > baseline["hit rate@3"]:
        st.success(f"In this data, **{best_name}** found the hidden product in the top 3 in "
                   f"{best['hit rate@3']:.1%} of cases, compared with {baseline['hit rate@3']:.1%} for best sellers. "
                   f"It beat the baseline in {wins} of {n} splits.", icon=":material/insights:")
    else:
        st.warning(f"In this data, no association-rule method beat the best-seller baseline at Hit Rate@3 "
                   f"({baseline['hit rate@3']:.1%}). Best sellers may be the better default here.",
                   icon=":material/insights:")
    ui.chart(plots.evaluation_chart(result.summary))
    display = result.summary.copy()
    for column in [c for c in display.columns if c.startswith("splits beating")]:
        display[column] = display[column].map(lambda v: "-" if pd.isna(v) else f"{int(v)} of {n}")
    st.dataframe(display, hide_index=True, width="stretch", column_config=ui.column_config(display))
    with st.expander("Results for each split", icon=":material/table:"):
        st.dataframe(result.per_split, hide_index=True, width="stretch",
                     column_config=ui.column_config(result.per_split))
    left, right = st.columns(2)
    left.download_button("Download summary (CSV)", result.summary.to_csv(index=False), "evaluation_summary.csv",
                         "text/csv", icon=":material/download:")
    right.download_button("Download per-split results (CSV)", result.per_split.to_csv(index=False),
                          "evaluation_per_split.csv", "text/csv", icon=":material/download:")
    _limitations(data.is_synthetic)
