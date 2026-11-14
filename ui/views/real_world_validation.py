"""Evaluation Lab, second tab: chronological validation on the real UCI Online Retail data.

Kept separate from the random-split benchmark on purpose: different data, a different split and
different settings, so the two results are never shown as one benchmark. Nothing heavy runs until
the user clicks, and the prepared data and results are cached for the running app.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from market_basket import plots, real_world, uci
from market_basket.temporal import BASELINE, PRIMARY, TemporalResult
from ui import components as ui

SUPPORT_OPTIONS = [0.005, 0.0075, 0.01, 0.015, 0.02, 0.03, 0.05]
CONFIDENCE_OPTIONS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
LIFT_OPTIONS = [1.0, 1.2, 1.5, 2.0]
SHARE_OPTIONS = [0.1, 0.15, 0.2, 0.25, 0.3]


def dataset_path() -> Path:
    """The UCI file to use: the download script's CSV unless a test or user set another path."""
    return Path(st.session_state.get("uci_path", uci.DEFAULT_CSV_PATH))


@st.cache_resource(show_spinner=False, max_entries=2)
def _prepare(path: str, modified: float, size: int, require_customer_id: bool) -> real_world.PreparedSource:
    return real_world.prepare(path, require_customer_id, progress=lambda _: None)


@st.cache_resource(show_spinner=False, max_entries=4)
def _evaluate(source_key: tuple, settings: tuple, _source: real_world.PreparedSource,
              _progress) -> TemporalResult:
    options = dict(settings)
    return real_world.evaluate(_source, progress=_progress, **options)


@st.cache_resource(show_spinner=False, max_entries=4)
def _exports(source_key: tuple, settings: tuple, _source: real_world.PreparedSource,
             _result: TemporalResult) -> dict:
    """Report files and the ZIP for one result, built once instead of on every rerun of the page."""
    files = real_world.evaluation_files(_source, _result)
    return {"json": files["real_world_evaluation.json"], "markdown": files["real_world_evaluation.md"],
            "zip": real_world.export_zip(_source, _result),
            "data": json.loads(files["real_world_evaluation.json"])}


def _percent_config(table: pd.DataFrame, columns=None) -> dict:
    columns = [c for c in table.columns if table[c].dtype.kind == "f"] if columns is None else columns
    return {c: st.column_config.NumberColumn(str(c)[:1].upper() + str(c)[1:], format="percent") for c in columns}


def _file_key(path: Path, require_customer_id: bool) -> tuple:
    stat = path.stat()
    return str(path), stat.st_mtime, stat.st_size, require_customer_id


def _intro() -> None:
    st.markdown(
        "This tab tests BasketWise on **real transactions** instead of the synthetic demo data, and **by time** "
        "instead of at random: rules and best sellers are learned from earlier invoices only, and each later "
        "invoice is used to ask *if one product is hidden, do the suggestions bring it back?*"
    )
    with st.expander("How this differs from the random-split benchmark", icon=":material/compare_arrows:"):
        st.markdown(
            "- **Data:** the UCI Online Retail dataset (a UK online gift retailer, Dec 2010 – Dec 2011), not the "
            "synthetic demo data with planted associations.\n"
            "- **Split:** a chronological cutoff. Every training invoice is earlier than every test invoice, so "
            "the test is closer to using past patterns on future baskets. The random-split benchmark mixes "
            "periods.\n"
            "- **Settings:** thresholds are chosen on a validation period at the end of the training period; the "
            "test period is never used to choose them.\n"
            "- **The two results are separate experiments.** Their numbers must not be combined or compared as one "
            "benchmark.\n"
            "- **What it does not show:** next-purchase prediction, extra sales, revenue, or real customer adoption."
        )
    st.caption(f"Data: {ui.escape(uci.CITATION)} Licensed under {uci.LICENCE}. The data is not included in this "
               "repository; it is downloaded from the UCI Machine Learning Repository.")


def _quality(source: real_world.PreparedSource) -> None:
    quality, summary = source.quality, source.prepared.summary
    st.subheader("Data quality and preprocessing")
    ui.metric_row([
        ("Raw rows", f"{quality['rows']:,}", "Rows in the downloaded file."),
        ("Valid invoices", f"{summary['valid_invoices']:,}", "Purchase invoices left after cleaning."),
        ("Products", f"{summary['unique_products']:,}", "Distinct stock codes in the valid invoices."),
        ("Customers", f"{summary['unique_customers']:,}", "Distinct customer IDs; invoices without one are kept."),
    ])
    st.caption(f"Raw data: {quality['dates']['first'][:10]} to {quality['dates']['last'][:10]}, "
               f"{quality['invoices']['total']:,} invoices including "
               f"{quality['invoices']['cancellation_invoices']:,} cancellation invoices, "
               f"{quality['customers']['rows_without_customer_id']:,} rows without a customer ID, "
               f"{quality['exact_duplicate_rows']:,} exact duplicate rows, {quality['countries']['count']} countries.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        table = uci.preprocessing_table(source.prepared)
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"step": st.column_config.TextColumn("Step", width="large"),
                                    "count": st.column_config.NumberColumn("Rows / count", format="localized")})
    with right:
        st.markdown("**Preprocessing decisions**")
        st.markdown(
            "- A basket is an invoice; products are identified by stock code.\n"
            "- Cancellation invoices (number starts with C) and zero or negative quantities are never purchases.\n"
            "- Postage, carriage, fees, discounts, manual entries, samples and vouchers are not merchandise.\n"
            "- Zero or negative prices are removed; missing customer IDs are kept.\n"
            "- Every removed row keeps its reason (download below)."
        )
    with st.expander("Full data-quality report", icon=":material/fact_check:"):
        st.json(real_world.json_safe(quality), expanded=1)
    left, right = st.columns(2)
    left.download_button("Download preprocessing summary (CSV)", table.to_csv(index=False),
                         "preprocessing_summary.csv", "text/csv", icon=":material/download:")
    right.download_button("Download removed rows (CSV)", source.prepared.removed.to_csv(index=False),
                          "removed_rows.csv", "text/csv", icon=":material/download:")


def _settings_form(source: real_world.PreparedSource) -> dict | None:
    times = source.prepared.invoices["time"]
    first_day, last_day = times.min().date(), times.max().date()
    with st.form("real_world_settings", border=True):
        st.markdown("**Evaluation settings** (the defaults are the published configuration)")
        left, middle, right = st.columns(3, gap="large")
        split_mode = left.radio("Test period", ["Latest share of invoices", "Fixed start date"],
                                help="The cutoff is always a midnight, so a day is never split.")
        test_share = left.select_slider("Share of invoices held out", SHARE_OPTIONS,
                                        value=real_world.DEFAULT_TEST_SHARE, format_func=ui.pct)
        cutoff = left.date_input("Test period starts on", value=None, min_value=first_day, max_value=last_day,
                                 help="Used only with 'Fixed start date'.")
        supports = middle.multiselect("Candidate minimum supports", SUPPORT_OPTIONS,
                                      default=list(real_world.DEFAULT_SUPPORTS), format_func=lambda v: f"{v:.2%}")
        confidences = middle.multiselect("Candidate minimum confidences", CONFIDENCE_OPTIONS,
                                         default=list(real_world.DEFAULT_CONFIDENCES), format_func=ui.pct)
        lifts = middle.multiselect("Candidate minimum lifts", LIFT_OPTIONS, default=list(real_world.DEFAULT_LIFTS))
        top_k = right.number_input("Recommendations per case (K)", 1, 10, real_world.DEFAULT_TOP_K)
        validation_share = right.select_slider("Validation share of the training period", SHARE_OPTIONS,
                                               value=real_world.DEFAULT_VALIDATION_SHARE, format_func=ui.pct,
                                               help="Used to choose among several candidate values.")
        st.caption("With several candidate values, the best combination is chosen on the validation period (the "
                   "end of the training period), never on the test period. Rules have at most 2 products, the "
                   "default of this experiment; the limitations below the results say whether longer rules would "
                   "have exceeded the mining safety limit on this training period.")
        run = st.form_submit_button("Run chronological evaluation", type="primary", icon=":material/play_arrow:")
    if not run:
        return None
    if split_mode == "Fixed start date" and cutoff is None:
        st.error("Choose the date on which the test period starts, or use 'Latest share of invoices'.",
                 icon=":material/error:")
        return None
    return {"cutoff": pd.Timestamp(cutoff).isoformat() if split_mode == "Fixed start date" else None,
            "test_share": test_share, "validation_share": validation_share, "supports": tuple(supports),
            "confidences": tuple(confidences), "lifts": tuple(lifts), "max_len": real_world.DEFAULT_MAX_LEN,
            "top_k": int(top_k), "n_bootstrap": real_world.DEFAULT_BOOTSTRAP, "seed": real_world.DEFAULT_SEED}


def _results(source: real_world.PreparedSource, result: TemporalResult, source_key: tuple,
             settings_key: tuple) -> None:
    k = result.settings["top_k"]
    hr = f"hit rate@{k}"
    split, config = result.split, result.config
    summary = result.summary.set_index("method")
    base, main = summary.loc[BASELINE], summary.loc[PRIMARY]
    st.subheader("Results on the test period")
    st.caption(f"Training: {split['training']['first_invoice'][:10]} to {split['training']['last_invoice'][:10]} "
               f"({split['training']['invoices']:,} invoices). Test: {split['test']['first_invoice'][:10]} to "
               f"{split['test']['last_invoice'][:10]} ({split['test']['invoices']:,} invoices). Thresholds: support "
               f"{config.min_support:.2%}, confidence {config.min_confidence:.0%}, lift {config.min_lift:g}"
               + (" (chosen on the validation period)" if not result.selection.empty else "")
               + f"; {len(result.rules):,} rules learned from the training period.")
    interval = result.intervals.get("difference_vs_baseline", {}).get(PRIMARY, {}).get("interval") \
        if result.intervals else None
    ui.metric_row([
        (f"Best-seller Hit Rate@{k}", ui.pct(base[hr], 1), real_world.METHOD_NOTES[BASELINE]),
        (f"Association rules Hit Rate@{k}", ui.pct(main[hr], 1), real_world.METHOD_NOTES[PRIMARY]),
        ("Difference", f"{(main[hr] - base[hr]) * 100:+.1f} pts",
         f"95% interval: {real_world.format_points(interval)}" if interval else "No interval computed."),
        ("Evaluated cases", f"{int(base['cases']):,}", f"From {int(base['invoices']):,} test invoices; the same cases "
                                                       "for every method."),
    ])
    if interval and interval[0] > 0:
        st.success(f"On this test period, association rules learned from earlier invoices found the hidden product "
                   f"in the top {k} more often than best sellers did. This is offline basket completion on one "
                   "retailer's data; it does not show extra sales.", icon=":material/insights:")
    elif interval and interval[1] < 0:
        st.warning("On this test period, best sellers found the hidden product more often than the association "
                   "rules.", icon=":material/insights:")
    else:
        st.info("On this test period, the difference between the methods is not clearly different from zero.",
                icon=":material/insights:")
    st.dataframe(real_world.comparison_table(result, k), hide_index=True, width="stretch")
    left, right = st.columns(2, gap="large")
    with left:
        ui.chart(plots.real_world_chart(result.summary, result.intervals, k, BASELINE, PRIMARY))
    with right:
        ui.chart(plots.weekly_hit_rate_chart(result.weekly, k, BASELINE, PRIMARY))
    with st.expander("All methods", icon=":material/table:"):
        table = real_world.methods_table(result, k)
        st.dataframe(table, hide_index=True, width="stretch", column_config=_percent_config(table))
    if not result.selection.empty:
        with st.expander("Threshold selection on the validation period", icon=":material/tune:"):
            v = result.validation
            st.caption(f"Validation period {v['validation_part']['first_invoice'][:10]} to "
                       f"{v['validation_part']['last_invoice'][:10]} ({v['cases']:,} cases); rules learned from "
                       f"{v['training_part']['first_invoice'][:10]} to {v['training_part']['last_invoice'][:10]}. "
                       f"Best-seller baseline on validation: {ui.pct(v[f'baseline {hr}'], 1)}.")
            table = real_world.selection_table(result.selection, hr)
            st.dataframe(table, hide_index=True, width="stretch",
                         column_config=_percent_config(table, [c for c in table if str(c).startswith("validation")]))
    exports = _exports(source_key, settings_key, source, result)
    data = exports["data"]
    with st.expander("Evaluation configuration", icon=":material/settings:"):
        st.json({"split": data["split"], "settings": data["settings"], "dataset": data["dataset"]}, expanded=1)
    left, middle, right = st.columns(3)
    left.download_button("Download results (JSON)", exports["json"], "real_world_evaluation.json", "application/json",
                         icon=":material/download:")
    middle.download_button("Download report (Markdown)", exports["markdown"], "real_world_evaluation.md",
                           "text/markdown", icon=":material/download:")
    right.download_button("Download all results (ZIP)", exports["zip"], "real_world_validation.zip",
                          "application/zip", icon=":material/download:")
    st.subheader("Limitations")
    for note in data["limitations"]:
        st.markdown(f"- {ui.escape(note)}")


def render() -> None:
    _intro()
    path = dataset_path()
    if not path.is_file():
        st.info("The UCI Online Retail dataset has not been downloaded yet. From the project folder, run the command "
                "below (about 23 MB from archive.ics.uci.edu), then come back to this tab.", icon=":material/info:")
        st.code("python scripts/download_uci_online_retail.py", language="bash")
        return
    require_customer_id = st.toggle("Require a customer ID (drop invoices without one)", value=False,
                                    key="uci_require_customer_id")
    if not st.session_state.get("uci_loaded"):
        if not st.button("Load and check the dataset", icon=":material/upload_file:"):
            st.caption(f"File: {ui.escape(path)}")
            return
        st.session_state.uci_loaded = True
    key = _file_key(path, require_customer_id)
    try:
        with st.spinner("Reading, checking and preprocessing the UCI data (about 10 seconds)..."):
            source = _prepare(*key)
    except (ValueError, OSError) as error:  # DatasetError is a ValueError
        st.error(f"The dataset could not be used: {error}", icon=":material/error:")
        return
    _quality(source)
    settings = _settings_form(source)
    if settings is not None:
        settings_key = tuple(sorted(settings.items()))
        status = st.status("Running the chronological evaluation (one to two minutes with the default grid)...",
                           expanded=True)
        try:
            result = _evaluate(key, settings_key, source, status.write)
        except ValueError as error:
            status.update(label="The evaluation could not run", state="error")
            st.error(f"The evaluation could not run: {error}", icon=":material/error:")
        else:
            status.update(label="Evaluation finished", state="complete", expanded=False)
            st.session_state.real_world = {"key": key, "settings": settings_key, "result": result}
    stored = st.session_state.get("real_world")
    if stored and stored["key"] == key:
        _results(source, stored["result"], key, stored["settings"])
    elif stored:
        st.info("The dataset or the customer-ID option changed after the last run. Run the evaluation again.",
                icon=":material/info:")
