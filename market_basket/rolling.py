"""Multi-cutoff (rolling-origin) chronological evaluation on the UCI Online Retail data.

The single split in temporal.py has one test period. This module repeats that protocol at several
cutoffs, to see whether the result holds at other times of the year.

Protocol (docs/PRODUCT_DECISIONS.md, decision 25)
-------------------------------------------------
1. Cutoffs are the first day of every calendar month, starting `min_training_months` after the month of
   the first invoice and ending with the last month that has an invoice. A list of dates can be given
   instead.
2. Fold i trains on every invoice before cutoff i (an expanding window) and tests on the invoices from
   cutoff i up to cutoff i + 1 (the last fold: up to the last invoice). Every invoice is tested at most once.
3. Each fold runs temporal.evaluate_chronological on its own. Thresholds are chosen on the latest
   `validation_share` of that fold's training invoices. Rules and best sellers are learned from that
   fold's training invoices, and its test invoices are scored by leave-one-out.
4. Reported: every fold's result, with a paired bootstrap interval for the difference; how many folds
   have that interval above zero, below zero or including zero; and pooled results over all folds and
   over the folds whose test period ends on or before the single-split cutoff (they share no test
   invoice with the single split).

Outputs (written by `python -m market_basket uci-rolling`):

    rolling_evaluation.json         machine-readable results, settings and provenance
    rolling_evaluation.md           the human-readable report
    rolling_folds.csv               one row per fold
    rolling_pooled_summary.csv      pooled results per method
    rolling_threshold_selection.csv every candidate setting of every fold
    rolling_cases.csv.gz            every scored case, with its fold
"""
from __future__ import annotations

import json
import math
import platform
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from . import real_world, uci
from .config import ALGORITHM_NAMES, MiningConfig
from .reporting import markdown_table
from .temporal import (
    BASELINE,
    PRIMARY,
    Progress,
    TemporalResult,
    bootstrap_intervals,
    candidate_settings,
    cutoff_for_share,
    evaluate_chronological,
    summarise,
)

OUTPUT_DIR = real_world.OUTPUT_DIR / "rolling"
DEFAULT_MIN_TRAINING_MONTHS = 3
ALL_FOLDS = "all folds"
EARLY_FOLDS = "folds before the single-split test period"
QUESTION = ("Does the single-split result hold at other times? At every monthly cutoff, do association rules "
            "learned only from earlier invoices complete the next month's baskets more often than best sellers?")


def _quiet(_: str) -> None:
    pass


@dataclass(frozen=True)
class RollingResult:
    settings: dict                  # every setting used, as plain values
    reference_cutoff: pd.Timestamp  # the single-split cutoff (latest test_share of all invoices)
    folds: pd.DataFrame             # one row per fold
    selection: pd.DataFrame         # every candidate setting of every fold (empty if no selection)
    pooled: pd.DataFrame            # one row per (group of folds, method)
    intervals: dict                 # {group of folds: bootstrap intervals}
    fold_results: list[TemporalResult] = field(repr=False)
    cases: pd.DataFrame = field(repr=False)  # every fold's cases, with a "fold" column


def monthly_cutoffs(times: pd.Series, min_training_months: int = DEFAULT_MIN_TRAINING_MONTHS) -> list[pd.Timestamp]:
    """The first day of every month from `min_training_months` after the month of the first invoice
    up to the month of the last invoice."""
    if isinstance(min_training_months, bool) or not isinstance(min_training_months, int) or min_training_months < 1:
        raise ValueError(f"The training period must be at least 1 month, got {min_training_months!r}.")
    times = pd.to_datetime(times)
    if times.empty or times.isna().any():
        raise ValueError("Every invoice needs a valid date and time for a chronological split.")
    first, last = times.min(), times.max()
    start = first.to_period("M").start_time + pd.DateOffset(months=min_training_months)
    cutoffs = list(pd.date_range(start, last, freq="MS"))
    if not cutoffs:
        raise ValueError(f"The invoices run from {first:%Y-%m-%d} to {last:%Y-%m-%d}, so no month is left to test "
                         f"after {min_training_months} months of training. Use fewer training months or give "
                         "the cutoffs.")
    return cutoffs


def _verdict(interval) -> str:
    if not interval:
        return "not computed"
    if interval[0] > 0:
        return "recommender higher"
    if interval[1] < 0:
        return "baseline higher"
    return "no clear difference"


def _fold_row(number: int, cutoff: pd.Timestamp, end: pd.Timestamp | None, result: TemporalResult,
              reference_cutoff: pd.Timestamp, top_k: int) -> dict:
    hr = f"hit rate@{top_k}"
    summary = result.summary.set_index("method")
    base, main = summary.loc[BASELINE], summary.loc[PRIMARY]
    interval = (result.intervals["difference_vs_baseline"][PRIMARY]["interval"] if result.intervals else None)
    config, split = result.config, result.split
    return {
        "fold": number,
        "cutoff": f"{cutoff:%Y-%m-%d}",
        "test_end_exclusive": "" if end is None else f"{end:%Y-%m-%d}",
        "training_first_invoice": split["training"]["first_invoice"],
        "test_first_invoice": split["test"]["first_invoice"],
        "test_last_invoice": split["test"]["last_invoice"],
        "training_invoices": split["training"]["invoices"],
        "test_invoices": split["test"]["invoices"],
        "test_invoices_with_two_or_more_products": split["test"]["invoices_with_two_or_more_products"],
        "cases": int(base["cases"]),
        "min_support": config.min_support, "min_confidence": config.min_confidence, "min_lift": config.min_lift,
        "max_len": config.max_len,
        "rules": len(result.rules),
        f"{hr}: {BASELINE}": float(base[hr]),
        f"{hr}: {PRIMARY}": float(main[hr]),
        "difference": float(main[hr] - base[hr]),
        "difference 95% low": interval[0] if interval else float("nan"),
        "difference 95% high": interval[1] if interval else float("nan"),
        "verdict": _verdict(interval),
        f"invoice-averaged {hr}: {BASELINE}": float(base[f"invoice-averaged {hr}"]),
        f"invoice-averaged {hr}: {PRIMARY}": float(main[f"invoice-averaged {hr}"]),
        f"coverage: {PRIMARY}": float(main["coverage"]),
        f"availability: {PRIMARY}": float(main["availability"]),
        "hidden product unseen in training": float(1 - result.cases["hidden_product_in_training"].mean()),
        "validation share of training": result.settings["validation_share"],
        "before the single-split test period": end is not None and end <= reference_cutoff,
    }


def pooled_summary(cases: pd.DataFrame, names: list[str], top_k: int, group: str) -> pd.DataFrame:
    """Metrics per method over the cases of several folds (coverage is per fold only: each fold has its own
    candidate universe)."""
    table = summarise(cases, names, top_k, 0, {name: set() for name in names}).drop(columns="coverage")
    table.insert(0, "folds", cases["fold"].nunique())
    table.insert(0, "group", group)
    return table


def evaluate_rolling(invoices: pd.DataFrame, candidates: list[MiningConfig], cutoffs: list | None = None,
                     min_training_months: int = DEFAULT_MIN_TRAINING_MONTHS,
                     validation_share: float = real_world.DEFAULT_VALIDATION_SHARE,
                     top_k: int = real_world.DEFAULT_TOP_K, n_bootstrap: int = real_world.DEFAULT_BOOTSTRAP,
                     seed: int = real_world.DEFAULT_SEED, reference_share: float = real_world.DEFAULT_TEST_SHARE,
                     progress: Progress = _quiet) -> RollingResult:
    """Run the protocol in the module docstring on an invoice table made by uci.preprocess."""
    if not candidates:
        raise ValueError("Give at least one threshold setting.")
    times = pd.to_datetime(invoices["time"])
    if cutoffs is None:
        cuts = monthly_cutoffs(times, min_training_months)
        rule = (f"first day of every month, from {min_training_months} months after the month of the first "
                "invoice")
    else:
        cuts = sorted({pd.Timestamp(c) for c in cutoffs})
        if not cuts:
            raise ValueError("Give at least one cutoff date.")
        rule = "dates given by the user"
    reference = cutoff_for_share(times, reference_share)
    ends = [*cuts[1:], None]
    results, rows = [], []
    for number, (cut, end) in enumerate(zip(cuts, ends, strict=True), start=1):
        label = f"Fold {number} of {len(cuts)} (test from {cut:%Y-%m-%d}"
        label += ")" if end is None else f" to {end - pd.Timedelta(days=1):%Y-%m-%d})"
        progress(label)
        window = invoices if end is None else invoices[times < end]
        try:
            result = evaluate_chronological(window, candidates, cutoff=cut, validation_share=validation_share,
                                            top_k=top_k, n_bootstrap=n_bootstrap, seed=seed,
                                            progress=lambda text, n=number: progress(f"  fold {n}: {text}"))
        except ValueError as error:
            raise ValueError(f"Fold {number} (test period from {cut:%Y-%m-%d}): {error}") from error
        results.append(result)
        rows.append(_fold_row(number, cut, end, result, reference, top_k))
    folds = pd.DataFrame(rows)

    names = list(results[0].summary["method"])
    cases = pd.concat([r.cases.assign(fold=n) for n, r in enumerate(results, start=1)], ignore_index=True)
    selection = pd.concat([r.selection.assign(fold=n) for n, r in enumerate(results, start=1) if not r.selection.empty],
                          ignore_index=True) if len(candidates) > 1 else pd.DataFrame()
    if not selection.empty:
        selection = selection[["fold", *[c for c in selection.columns if c != "fold"]]]
    groups = {ALL_FOLDS: cases}
    early = folds.loc[folds["before the single-split test period"], "fold"]
    if len(early):
        groups[EARLY_FOLDS] = cases[cases["fold"].isin(early)]
    pooled = pd.concat([pooled_summary(subset, names, top_k, group) for group, subset in groups.items()],
                       ignore_index=True)
    intervals = {group: bootstrap_intervals(subset, names, BASELINE, n_bootstrap, seed)
                 for group, subset in groups.items()}
    settings = {
        "top_k": top_k, "cutoff_rule": rule, "cutoffs": [f"{c:%Y-%m-%d}" for c in cuts],
        "min_training_months": min_training_months if cutoffs is None else None,
        "validation_share": validation_share if len(candidates) > 1 else None,
        "candidate_settings": len(candidates),
        "candidates": {"min_support": sorted({c.min_support for c in candidates}),
                       "min_confidence": sorted({c.min_confidence for c in candidates}),
                       "min_lift": sorted({c.min_lift for c in candidates})},
        "max_len": candidates[0].max_len, "algorithm": ALGORITHM_NAMES[candidates[0].algorithm],
        "bootstrap_resamples": n_bootstrap, "seed": seed,
        "reference_cutoff": reference.isoformat(),
        "reference_rule": f"the single-split cutoff: start of the day containing the first invoice of the latest "
                          f"{reference_share:.0%} of invoices",
        "window": "expanding: every invoice before the cutoff",
    }
    return RollingResult(settings, reference, folds, selection, pooled, intervals, results, cases)


def evaluate(source: real_world.PreparedSource, cutoffs: list | None = None,
             min_training_months: int = DEFAULT_MIN_TRAINING_MONTHS,
             validation_share: float = real_world.DEFAULT_VALIDATION_SHARE, supports=real_world.DEFAULT_SUPPORTS,
             confidences=real_world.DEFAULT_CONFIDENCES, lifts=real_world.DEFAULT_LIFTS,
             max_len: int | None = real_world.DEFAULT_MAX_LEN, top_k: int = real_world.DEFAULT_TOP_K,
             n_bootstrap: int = real_world.DEFAULT_BOOTSTRAP, seed: int = real_world.DEFAULT_SEED,
             progress: Progress = print) -> RollingResult:
    """Multi-cutoff evaluation of the prepared UCI baskets."""
    candidates = candidate_settings(supports, confidences, lifts, max_len)
    return evaluate_rolling(source.prepared.invoices, candidates, cutoffs=cutoffs,
                            min_training_months=min_training_months, validation_share=validation_share,
                            top_k=top_k, n_bootstrap=n_bootstrap, seed=seed, progress=progress)


# ---------------------------------------------------------------- results and report


def uses_default_protocol(source: real_world.PreparedSource, result: RollingResult) -> bool:
    """True only for the configuration fixed in decision 25: the official file and every default setting."""
    s = result.settings
    return (real_world.is_official_file(source) and not source.prepared.summary["require_customer_id"]
            and s["min_training_months"] == DEFAULT_MIN_TRAINING_MONTHS and s["top_k"] == real_world.DEFAULT_TOP_K
            and s["validation_share"] == real_world.DEFAULT_VALIDATION_SHARE
            and s["candidates"] == {"min_support": list(real_world.DEFAULT_SUPPORTS),
                                    "min_confidence": list(real_world.DEFAULT_CONFIDENCES),
                                    "min_lift": list(real_world.DEFAULT_LIFTS)}
            and s["max_len"] == real_world.DEFAULT_MAX_LEN and s["bootstrap_resamples"] == real_world.DEFAULT_BOOTSTRAP
            and s["seed"] == real_world.DEFAULT_SEED
            and result.fold_results[0].config.exclude == ())


def verdict_counts(result: RollingResult) -> dict:
    counts = result.folds["verdict"].value_counts()
    return {verdict: int(counts.get(verdict, 0)) for verdict in
            ("recommender higher", "no clear difference", "baseline higher", "not computed")}


def _pooled_row(result: RollingResult, group: str, method: str) -> pd.Series:
    table = result.pooled
    return table[(table["group"] == group) & (table["method"] == method)].iloc[0]


def _folds(count: int) -> str:
    return f"{count} fold" + ("" if count == 1 else "s")


def _fold_list(numbers) -> str:
    """'fold 4', 'folds 1 and 2' or 'folds 1, 2 and 3'."""
    numbers = [str(n) for n in sorted({int(n) for n in numbers})]
    if len(numbers) == 1:
        return f"fold {numbers[0]}"
    return f"folds {', '.join(numbers[:-1])} and {numbers[-1]}"


def answer(result: RollingResult) -> str:
    """The report's one-paragraph answer, computed from the results."""
    k = result.settings["top_k"]
    hr = f"hit rate@{k}"
    folds, n = result.folds, len(result.folds)
    counts = verdict_counts(result)
    if counts["not computed"] == n:
        parts = [f"Over {n} folds, the per-fold differences were not given intervals (no bootstrap)."]
    elif counts["recommender higher"] == n:
        parts = [(f"In all {n} folds the association-rule recommender found the hidden product in the top {k} more "
                 "often than the best sellers, and every fold's 95% interval for the difference is above zero.")]
    else:
        parts = [(f"Of {n} folds, the recommender was clearly higher in {counts['recommender higher']}, clearly lower "
                 f"in {counts['baseline higher']}, and the 95% interval included zero in "
                 f"{counts['no clear difference']}.")]
    worst, best = folds.loc[folds["difference"].idxmin()], folds.loc[folds["difference"].idxmax()]
    parts.append(f"The difference ranged from {worst['difference'] * 100:+.1f} points (test period from "
                 f"{worst['cutoff']}) to {best['difference'] * 100:+.1f} points (from {best['cutoff']}).")
    for group in result.intervals:
        base, main = _pooled_row(result, group, BASELINE), _pooled_row(result, group, PRIMARY)
        interval = result.intervals[group].get("difference_vs_baseline", {}).get(PRIMARY, {}).get("interval")
        subset = folds if group == ALL_FOLDS else folds[folds["before the single-split test period"]]
        span = f"test periods from {subset['cutoff'].iloc[0]} to {subset['test_last_invoice'].iloc[-1][:10]}"
        parts.append(f"Pooled over {group} ({_folds(len(subset))}, {span}): {main[hr]:.1%} against {base[hr]:.1%}, "
                     f"{(main[hr] - base[hr]) * 100:+.1f} points"
                     + (f" (95% interval {real_world.format_points(interval)})." if interval else "."))
    return " ".join(parts)


def limitations(source: real_world.PreparedSource, result: RollingResult) -> list[str]:
    folds = result.folds
    settings = folds[["min_support", "min_confidence", "min_lift"]].drop_duplicates()
    smallest = folds.loc[folds["training_invoices"].idxmin()]
    notes = [
        ("One retailer: every fold comes from the same online retailer and the same year, so the folds say nothing "
        "certain about other shops."),
        ("The folds are not independent. Their training periods overlap (each one contains the previous one), "
        "neighbouring months share customers and seasons, and the pooled intervals resample test invoices, not "
        "periods. Each fold's interval is reported separately and is not corrected for making several comparisons: "
        "with many folds, one interval can exclude zero by chance."),
        ("Like the single split, this is offline basket completion: the hidden product comes from the same invoice, "
        "and a hit means the suggestion matched a product the customer bought anyway. It shows no extra sales, "
        "revenue or causal effect."),
        ("The thresholds were fixed by the user (a single value each), not chosen on validation periods."
         if result.selection.empty else
         "Each fold chose its own thresholds on its own validation period, so every fold uses only data before its "
         "cutoff. " + ("Every fold chose the same setting." if len(settings) == 1 else
                       f"{len(settings)} different settings were chosen, so the folds do not all use the same "
                       "rules settings.")),
        (f"The earliest folds learn from little data: the smallest training period has "
        f"{int(smallest['training_invoices']):,} invoices (fold {int(smallest['fold'])})."),
        "Rules have at most "
        + (f"{result.settings['max_len']} products" if result.settings["max_len"] else "any number of products")
        + " and the threshold grid is small, as in the single split.",
    ]
    if not result.selection.empty:
        refused = result.selection[result.selection["status"].str.startswith("refused")]
        if len(refused):
            notes.append(f"In {_fold_list(refused['fold'])}, {len(refused)} candidate settings were refused by "
                         "BasketWise's safety limits (the reasons are in rolling_threshold_selection.csv), so those "
                         "folds chose among the remaining settings.")
        grid = sorted(result.selection["min_support"].unique())
        for edge, value in (("lowest", grid[0]), ("highest", grid[-1])):
            at_edge = folds[folds["min_support"] == value]
            if len(grid) > 1 and len(at_edge):
                notes.append(f"In {_fold_list(at_edge['fold'])}, the chosen minimum support ({value:.2%}) is the "
                             f"{edge} value in the grid, so a value outside the grid might have scored better on "
                             "validation.")
    last = folds.iloc[-1]
    days = (pd.Timestamp(last["test_last_invoice"]).normalize() - pd.Timestamp(last["cutoff"])).days + 1
    if days < 28:
        notes.append(f"The last fold's test period is short: it runs from {last['cutoff']} to "
                     f"{last['test_last_invoice'][:10]} ({days} days), where the data ends.")
    overlap = folds[~folds["before the single-split test period"]]
    if len(overlap):
        notes.append(f"The test periods of {_folds(len(overlap))} (from {overlap['cutoff'].iloc[0]}) run past the "
                     f"single-split cutoff ({result.reference_cutoff:%Y-%m-%d}), so they reuse test invoices that the "
                     "single split had already scored. The pooled result over the earlier folds avoids them.")
    if uses_default_protocol(source, result):
        notes.append(f"This run matches the multi-cutoff protocol of BasketWise {real_world.version_string()} on the "
                     "official UCI file. That protocol (monthly cutoffs from three months after the first invoice, "
                     "expanding training windows, the 3 x 3 threshold grid chosen per fold, at most 2 products per "
                     "rule, K = 3, 1,000 bootstrap resamples with seed 42, and the summaries reported here) was "
                     "written into docs/PRODUCT_DECISIONS.md (decision 25) before it was first run on the real "
                     "data. It was added after the single-split test period had been scored.")
    else:
        notes.append("This run differs from BasketWise's default multi-cutoff protocol or does not use the official "
                     "UCI file. If its settings were chosen after looking at results, its test periods are no "
                     "longer untouched hold-outs.")
    return notes


def results_dict(source: real_world.PreparedSource, result: RollingResult, generated: datetime | None = None) -> dict:
    generated = generated or datetime.now(UTC)
    return real_world.json_safe({
        "title": "BasketWise multi-cutoff chronological validation (UCI Online Retail)",
        "generated_utc": generated.replace(microsecond=0).isoformat(),
        "basketwise_version": real_world.version_string(),
        "question": QUESTION,
        "answer": answer(result),
        "not_measured": real_world.NOT_MEASURED,
        "dataset": real_world.dataset_info(source),
        "preprocessing": {key: source.prepared.summary[key] for key in source.prepared.summary
                          if key != "removed_by_reason"},
        "settings": result.settings,
        "verdict_counts": verdict_counts(result),
        "folds": result.folds.to_dict("records"),
        "pooled": result.pooled.to_dict("records"),
        "bootstrap": {group: intervals for group, intervals in result.intervals.items()} if any(
            result.intervals.values()) else None,
        "threshold_selection": result.selection.to_dict("records"),
        "limitations": limitations(source, result),
        "software": {"python": platform.python_version(),
                     **{name: version(name) for name in ("pandas", "numpy", "mlxtend", "openpyxl")}},
    })


def _pct(value, digits: int = 1) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}%}"


def folds_table(result: RollingResult) -> pd.DataFrame:
    k = result.settings["top_k"]
    hr = f"hit rate@{k}"
    rows = []
    for fold in result.folds.to_dict("records"):
        interval = None if math.isnan(fold["difference 95% low"]) else [fold["difference 95% low"],
                                                                         fold["difference 95% high"]]
        rows.append({
            "fold": fold["fold"],
            "test period": f"{fold['cutoff']} to {fold['test_last_invoice'][:10]}",
            "training invoices": f"{fold['training_invoices']:,}",
            "test invoices": f"{fold['test_invoices']:,}",
            "cases": f"{fold['cases']:,}",
            "support / confidence": f"{fold['min_support']:.1%} / {fold['min_confidence']:.0%}",
            "rules": f"{fold['rules']:,}",
            "best-seller baseline": _pct(fold[f"{hr}: {BASELINE}"]),
            "association-rule recommender": _pct(fold[f"{hr}: {PRIMARY}"]),
            "difference (95% interval)": f"{fold['difference'] * 100:+.1f} ("
                                         + (real_world.format_points(interval) if interval else "not computed") + ")",
            "before single-split test": "yes" if fold["before the single-split test period"] else "",
        })
    return pd.DataFrame(rows)


def pooled_table(result: RollingResult) -> pd.DataFrame:
    k = result.settings["top_k"]
    hr = f"hit rate@{k}"
    rows = []
    for row in result.pooled.to_dict("records"):
        interval = result.intervals.get(row["group"], {}).get("difference_vs_baseline", {}).get(row["method"], {})
        rows.append({
            "folds": f"{row['group']} ({row['folds']})", "method": row["method"], "cases": f"{row['cases']:,}",
            f"Hit Rate@{k}": _pct(row[hr]),
            "95% interval": real_world.format_interval(result.intervals.get(row["group"], {}).get("hit_rate", {})
                                                        .get(row["method"])),
            "difference vs baseline (95% interval)": "-" if row["method"] == BASELINE else
            (real_world.format_points(interval["interval"]) if interval else "not computed"),
            f"invoice-averaged Hit Rate@{k}": _pct(row[f"invoice-averaged {hr}"]),
            "availability": _pct(row["availability"]),
        })
    return pd.DataFrame(rows)


def selection_summary(result: RollingResult) -> pd.DataFrame:
    if result.selection.empty:
        return pd.DataFrame()
    k = result.settings["top_k"]
    rows = []
    for fold, table in result.selection.groupby("fold", sort=True):
        chosen = table[table["chosen"]].iloc[0]
        compared = (table["status"] == "compared").sum()
        validation = result.fold_results[fold - 1].validation
        rows.append({
            "fold": fold,
            "validation period": f"{validation['validation_part']['first_invoice'][:10]} to "
                                 f"{validation['validation_part']['last_invoice'][:10]}",
            "settings compared": f"{compared} of {len(table)}",
            "chosen support / confidence / lift": f"{chosen['min_support']:.1%} / {chosen['min_confidence']:.0%} / "
                                                  f"{chosen['min_lift']:g}",
            f"validation Hit Rate@{k} (chosen)": _pct(chosen[f"validation hit rate@{k}"]),
            f"validation Hit Rate@{k} (baseline)": _pct(validation[f"baseline hit rate@{k}"]),
        })
    return pd.DataFrame(rows)


def build_markdown(source: real_world.PreparedSource, result: RollingResult, generated: datetime | None = None) -> str:
    generated = generated or datetime.now(UTC)
    data = results_dict(source, result, generated)
    s, k = result.settings, result.settings["top_k"]
    dataset = data["dataset"]
    lines = [
        "# BasketWise multi-cutoff chronological validation (UCI Online Retail)",
        "",
        (f"Generated {generated:%Y-%m-%d %H:%M} UTC by BasketWise {data['basketwise_version']}. Every number in this "
        "report was computed by the code in this repository from the file named in section 1; none was typed by "
        "hand."),
        "",
        f"**Question.** {QUESTION}",
        "",
        f"**Answer for this data.** {data['answer']}",
        "",
        "**What this does not show:** " + "; ".join(real_world.NOT_MEASURED) + ".",
        "",
        ("This report extends the single-split report (`real_world_evaluation.md`), which has one test period. It "
        "uses the same data, preprocessing, methods and metrics, but its numbers come from different training and "
        "test periods, so they must not be added to the single split's numbers. The synthetic benchmark in "
        "`docs/EVALUATION_REPORT.md` is a separate experiment again."),
        "",
        "## 1. Dataset",
        "",
        markdown_table(pd.DataFrame([
            ("Dataset", "Online Retail, UCI Machine Learning Repository"),
            ("Citation", uci.CITATION),
            ("Licence", f"{uci.LICENCE} — {uci.LICENCE_URL}"),
            ("Retrieved (UTC)", dataset["retrieved_utc"] or "unknown"),
            ("File used", f"{dataset['file']} (SHA-256 {dataset['file_sha256']})"),
            ("Same file as the published results", "yes" if dataset["same_file_as_published_results"] else "no"),
            ("Valid invoices after preprocessing", f"{source.prepared.summary['valid_invoices']:,}"),
        ], columns=["item", "value"])),
        "",
        "Data quality and preprocessing follow the single-split pipeline; `uci-prepare` with the same options writes "
        "their reports. Rows without a customer ID were "
        + ("removed in this run." if source.prepared.summary["require_customer_id"] else "kept in this run."),
        "",
        "## 2. Protocol",
        "",
        f"1. **Cutoffs:** {s['cutoff_rule']}: " + ", ".join(s["cutoffs"]) + ".",
        ("2. **Folds:** each fold learns from every invoice before its cutoff (an expanding window) and tests on the "
        "invoices up to the next cutoff; the last fold tests up to the last invoice. No invoice is tested twice."),
        "3. **Inside each fold** the single-split protocol runs unchanged: "
        + (f"the {s['candidate_settings']} candidate settings are compared on the latest "
           f"{s['validation_share']:.0%} of that fold's training invoices, " if s["validation_share"] else
           "the one given setting is used, ")
        + f"rules and best sellers are learned from that fold's training invoices, and every product of every test "
        f"invoice with at least two products is hidden once. K = {k}.",
        f"4. **Intervals:** {s['bootstrap_resamples']:,} bootstrap resamples of whole test invoices (seed "
        f"{s['seed']}), the same resamples for both methods, per fold and for each pooled group."
        if s["bootstrap_resamples"] else "4. **Intervals:** not computed in this run.",
        (f"5. **Pooled groups:** all folds, and the folds whose test period ends on or before the single-split "
        f"cutoff ({result.reference_cutoff:%Y-%m-%d}), which share no test invoice with the single split."),
        "",
        ("The recommender is BasketWise's default (rules ranked by confidence, best sellers filling free slots). "
        "The baseline suggests the fold's training-period best sellers that are not already in the basket."),
        "",
        "## 3. Results per fold",
        "",
        markdown_table(folds_table(result)),
        "",
        (f"Columns: Hit Rate@{k} for each method on that fold's test period, and the difference in percentage "
        "points. 'support / confidence' is the setting chosen on that fold's validation period."),
        "",
        "## 4. Summary over the folds",
        "",
        "- " + "; ".join(f"{verdict}: {count}" for verdict, count in data["verdict_counts"].items()
                         if count or verdict != "not computed") + " (out of " + f"{len(result.folds)} folds).",
        (f"- Recommender Hit Rate@{k} per fold: from {result.folds[f'hit rate@{k}: {PRIMARY}'].min():.1%} to "
        f"{result.folds[f'hit rate@{k}: {PRIMARY}'].max():.1%}; best sellers: from "
        f"{result.folds[f'hit rate@{k}: {BASELINE}'].min():.1%} to "
        f"{result.folds[f'hit rate@{k}: {BASELINE}'].max():.1%}."),
        "",
        "Pooled results (every case of the folds in the group, each method on the same cases):",
        "",
        markdown_table(pooled_table(result)),
        "",
        "## 5. Threshold selection per fold",
        "",
    ]
    choice = selection_summary(result)
    if choice.empty:
        lines += ["A single threshold setting was given, so no selection was made.", ""]
    else:
        lines += [markdown_table(choice), "",
                  "Every candidate of every fold is in `rolling_threshold_selection.csv`.", ""]
    lines += [
        "## 6. Limitations",
        "",
        *[f"- {note}" for note in data["limitations"]],
        "",
        "## 7. How to reproduce",
        "",
        "```",
        "python scripts/download_uci_online_retail.py   # official ZIP -> Data/external/uci_online_retail/",
        "python -m market_basket uci-rolling            # this report and the rest of outputs/real_world/rolling/",
        "```",
        "",
        "With the same file and settings the results are identical on every run; only the generation time changes.",
        "",
    ]
    return "\n".join(lines)


def evaluation_files(source: real_world.PreparedSource, result: RollingResult,
                     generated: datetime | None = None) -> dict:
    generated = generated or datetime.now(UTC)

    def csv(frame: pd.DataFrame) -> str:
        return frame.to_csv(index=False, lineterminator="\n")

    return {
        "rolling_evaluation.json": json.dumps(results_dict(source, result, generated), indent=2,
                                              ensure_ascii=False) + "\n",
        "rolling_evaluation.md": build_markdown(source, result, generated),
        "rolling_folds.csv": csv(result.folds),
        "rolling_pooled_summary.csv": csv(result.pooled),
        "rolling_threshold_selection.csv": csv(result.selection),
    }


def write_outputs(source: real_world.PreparedSource, result: RollingResult, output_dir: str | Path,
                  generated: datetime | None = None) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, text in evaluation_files(source, result, generated).items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        paths.append(out / name)
    cases = out / "rolling_cases.csv.gz"
    table = result.cases.drop(columns="customer_id").assign(time=result.cases["time"].dt.strftime("%Y-%m-%d %H:%M:%S"))
    table = table[["fold", *[c for c in table.columns if c != "fold"]]]
    table.to_csv(cases, index=False, compression={"method": "gzip", "mtime": 0}, lineterminator="\n")
    return [*paths, cases]
