"""Real-world validation on the UCI Online Retail data: preparation, evaluation and reports.

This ties together uci.py (download, data quality, preprocessing) and temporal.py (chronological
evaluation) and writes the outputs used by the CLI and the Evaluation Lab:

    data_quality_report.json      facts about the raw table
    preprocessing_summary.csv     rows removed by reason and the resulting baskets
    removed_rows.csv.gz           every removed raw row with its reason (for auditing)
    real_world_evaluation.json    machine-readable results, settings and provenance
    real_world_evaluation.md      the human-readable report
    evaluation_summary.csv, weekly_results.csv, threshold_selection.csv, training_rules.csv,
    evaluation_cases.csv.gz       per-method metrics, per-week results, validation results,
                                  the rules learned from the training period, and every scored case
"""
from __future__ import annotations

import io
import json
import math
import platform
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from . import uci
from .config import ALGORITHM_NAMES, MAX_FREQUENT_ITEMSETS, PROJECT_ROOT
from .data import encode
from .mining import as_text, estimate_itemsets
from .reporting import markdown_table
from .temporal import (
    BASELINE,
    PRIMARY,
    Progress,
    TemporalResult,
    candidate_settings,
    evaluate_chronological,
    split_by_time,
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_world"
DEFAULT_SUPPORTS = (0.005, 0.01, 0.02)
DEFAULT_CONFIDENCES = (0.1, 0.2, 0.3)
DEFAULT_LIFTS = (1.2,)
DEFAULT_MAX_LEN = 2
DEFAULT_TOP_K = 3
DEFAULT_TEST_SHARE = 0.2
DEFAULT_VALIDATION_SHARE = 0.2
DEFAULT_BOOTSTRAP = 1000
DEFAULT_SEED = 42

METHOD_NOTES = {
    "Best-seller baseline": "The training period's best-selling products that are not already in the basket.",
    "Rules ranked by confidence": "Association rules learned from the training period, strongest confidence first; "
                                  "no fallback, so some cases get fewer than K suggestions.",
    "Rules ranked by lift": "The same rules, highest lift first; no fallback.",
    "Rules by confidence + best-seller fallback": "BasketWise's default recommender: confidence-ranked rules, with "
                                                  "training-period best sellers filling free slots.",
}
QUESTION = ("When part of a later invoice is known, do association rules learned only from earlier invoices "
            "suggest the invoice's remaining product more often than simply suggesting best sellers?")
NOT_MEASURED = [
    "next-purchase or next-order prediction (the hidden product comes from the same invoice)",
    "extra sales, revenue or any causal effect of showing recommendations (that needs an A/B test)",
    "real customer adoption (no recommendation was shown to any customer)",
    "how other shops or other periods would behave (the data comes from one online retailer)",
]


@dataclass(frozen=True)
class PreparedSource:
    path: str
    file_name: str
    sha256: str
    manifest: dict
    quality: dict
    prepared: uci.PreparedRetail


def prepare(path: str | Path = uci.DEFAULT_CSV_PATH, require_customer_id: bool = False,
            progress: Progress = print) -> PreparedSource:
    """Read the UCI file, measure its quality and preprocess it into baskets."""
    path = Path(path)
    progress(f"Reading {path}")
    raw = uci.read_raw(path)
    progress("Checking data quality")
    quality = uci.data_quality_report(raw)
    progress("Preprocessing into baskets")
    prepared = uci.preprocess(raw, require_customer_id=require_customer_id)
    return PreparedSource(str(path), path.name, uci.sha256(path), uci.read_manifest(path), quality, prepared)


def evaluate(source: PreparedSource, cutoff: str | None = None, test_share: float = DEFAULT_TEST_SHARE,
             validation_share: float = DEFAULT_VALIDATION_SHARE, supports=DEFAULT_SUPPORTS,
             confidences=DEFAULT_CONFIDENCES, lifts=DEFAULT_LIFTS, max_len: int | None = DEFAULT_MAX_LEN,
             top_k: int = DEFAULT_TOP_K, n_bootstrap: int = DEFAULT_BOOTSTRAP, seed: int = DEFAULT_SEED,
             progress: Progress = print) -> TemporalResult:
    """Chronological evaluation of the prepared baskets (see temporal.evaluate_chronological)."""
    candidates = candidate_settings(supports, confidences, lifts, max_len)
    result = evaluate_chronological(source.prepared.invoices, candidates, cutoff=cutoff, test_share=test_share,
                                    validation_share=validation_share, top_k=top_k, n_bootstrap=n_bootstrap,
                                    seed=seed, progress=progress)
    if result.config.max_len == 2:
        # Would three-product rules have been possible? Checked on the TRAINING period with the same guaranteed
        # upper bound that mining uses before it starts (see mining.estimate_itemsets).
        train, _ = split_by_time(source.prepared.invoices, result.split["cutoff"])
        bound = estimate_itemsets(encode([list(items) for items in train["items"]]), result.config.min_support, 3,
                                  stop_above=MAX_FREQUENT_ITEMSETS)
        result.settings["three_product_itemsets_exceed_limit"] = bool(bound > MAX_FREQUENT_ITEMSETS)
    return result


# ---------------------------------------------------------------- machine-readable results


def json_safe(value):
    """JSON-safe values: NaN becomes null, NumPy and pandas scalars become Python numbers."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    return value


def dataset_info(source: PreparedSource) -> dict:
    manifest = source.manifest
    return {
        "name": "Online Retail",
        "source_page": uci.SOURCE_PAGE, "download_url": uci.DOWNLOAD_URL, "doi": f"https://doi.org/{uci.DOI}",
        "citation": uci.CITATION, "licence": uci.LICENCE, "licence_url": uci.LICENCE_URL,
        "retrieved_utc": manifest.get("retrieved_utc"),
        "file": source.file_name, "file_sha256": source.sha256,
        "official_spreadsheet_sha256": manifest.get("files", {}).get("spreadsheet", {}).get("sha256"),
        "same_file_as_published_results": is_official_file(source),
        "rows": source.quality["rows"],
    }


def _top_rules(result: TemporalResult, labels: dict[str, str], n: int = 10) -> list[dict]:
    rows = []
    for rule in result.rules.head(n).itertuples(index=False):
        rows.append({"if the basket has": as_text(labels.get(p, p) for p in rule.antecedents),
                     "suggest": as_text(labels.get(p, p) for p in rule.consequents),
                     "support": float(rule.support), "confidence": float(rule.confidence), "lift": float(rule.lift)})
    return rows


def results_dict(source: PreparedSource, result: TemporalResult, generated: datetime | None = None) -> dict:
    generated = generated or datetime.now(UTC)
    k = result.settings["top_k"]
    config = result.config
    labels = source.prepared.labels()
    intervals = result.intervals
    methods = []
    for row in result.summary.to_dict("records"):
        name = row["method"]
        entry = {"method": name, "description": METHOD_NOTES.get(name, ""), **{key: row[key] for key in row
                                                                               if key != "method"}}
        if intervals:
            entry[f"hit rate@{k} 95% interval"] = intervals["hit_rate"][name]
            if name in intervals["difference_vs_baseline"]:
                entry["difference vs baseline 95% interval"] = intervals["difference_vs_baseline"][name]["interval"]
        methods.append(entry)
    summary = result.summary.set_index("method")
    return json_safe({
        "title": "BasketWise real-world chronological validation (UCI Online Retail)",
        "generated_utc": generated.replace(microsecond=0).isoformat(),
        "basketwise_version": version_string(),
        "question": QUESTION,
        "not_measured": NOT_MEASURED,
        "dataset": dataset_info(source),
        "data_quality": {key: source.quality[key] for key in source.quality if key != "countries"}
        | {"countries": {"count": source.quality["countries"]["count"],
                         "top_10_by_rows": dict(list(source.quality["countries"]["rows_by_country"].items())[:10])}},
        "preprocessing": source.prepared.summary,
        "split": result.split,
        "settings": {
            **result.settings,
            "candidates": {"min_support": sorted({c for c in result.selection.get("min_support",
                                                                                  [config.min_support])}),
                           "min_confidence": sorted({c for c in result.selection.get("min_confidence",
                                                                                     [config.min_confidence])}),
                           "min_lift": sorted({c for c in result.selection.get("min_lift", [config.min_lift])})},
            "chosen": {"min_support": config.min_support, "min_confidence": config.min_confidence,
                       "min_lift": config.min_lift, "max_len": config.max_len,
                       "algorithm": ALGORITHM_NAMES[config.algorithm]},
            "leave_one_out": "every product of every test invoice with two or more products is hidden once",
            "candidate_universe": "products bought in the training period",
            "denominator": "all leave-one-out cases of the test period, including hidden products never sold in "
                           "the training period (a miss for every method)",
        },
        "threshold_selection": {"validation": result.validation,
                                "candidates": result.selection.to_dict("records")},
        "training_rules": {"frequent_itemsets": result.n_itemsets, "rules": len(result.rules),
                           "top_rules_by_confidence": _top_rules(result, labels)},
        "primary_comparison": {"baseline": BASELINE, "recommender": PRIMARY,
                               f"hit rate@{k} difference (points)": 100 * (summary.loc[PRIMARY, f"hit rate@{k}"]
                                                                           - summary.loc[BASELINE, f"hit rate@{k}"])},
        "results": methods,
        "bootstrap": {key: intervals[key] for key in ("resamples", "level", "seed", "unit")} if intervals else None,
        "weekly": result.weekly.assign(**{"week starting": result.weekly["week starting"].dt.strftime("%Y-%m-%d")})
        .to_dict("records"),
        "limitations": limitations(source, result),
        "software": {"python": platform.python_version(),
                     **{name: version(name) for name in ("pandas", "numpy", "mlxtend", "openpyxl")}},
    })


def is_official_file(source: PreparedSource) -> bool:
    """True when the file is the CSV the download script made from the official spreadsheet whose
    checksum the published results were computed from."""
    files = source.manifest.get("files", {})
    return (files.get("spreadsheet", {}).get("sha256") == uci.OFFICIAL_SPREADSHEET_SHA256
            and files.get("csv", {}).get("sha256") == source.sha256)


def uses_default_protocol(source: PreparedSource, result: TemporalResult) -> bool:
    """True only for the published configuration: the official file and every default setting."""
    s, selection = result.settings, result.selection
    grid = (sorted(selection["min_support"].unique()), sorted(selection["min_confidence"].unique()),
            sorted(selection["min_lift"].unique())) if not selection.empty else None
    return (is_official_file(source) and not source.prepared.summary["require_customer_id"]
            and s["top_k"] == DEFAULT_TOP_K and s["fixed_cutoff"] is None and s["test_share"] == DEFAULT_TEST_SHARE
            and s["validation_share"] == DEFAULT_VALIDATION_SHARE and result.config.max_len == DEFAULT_MAX_LEN
            and grid == (list(DEFAULT_SUPPORTS), list(DEFAULT_CONFIDENCES), list(DEFAULT_LIFTS))
            and s["bootstrap_resamples"] == DEFAULT_BOOTSTRAP and s["seed"] == DEFAULT_SEED
            and not result.config.exclude)


def version_string() -> str:
    try:
        return version("basketwise")
    except PackageNotFoundError:  # not installed as a package: read pyproject.toml
        text = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        return next((line.split("=")[1].strip().strip('"') for line in text.splitlines()
                     if line.startswith("version")), "unknown")


# ---------------------------------------------------------------- human-readable report


def _minutes(value: float) -> str:
    return f"{value:g} minute" + ("" if value == 1 else "s")


def _pct(value, digits: int = 1) -> str:
    return "n/a" if value is None or (isinstance(value, float) and math.isnan(value)) else f"{value:.{digits}%}"


def format_interval(values) -> str:
    return f"{values[0]:.1%} to {values[1]:.1%}" if values else "not computed"


def format_points(values) -> str:
    return f"{values[0] * 100:+.1f} to {values[1] * 100:+.1f} points" if values else "not computed"


def limitations(source: PreparedSource, result: TemporalResult) -> list[str]:
    k = result.settings["top_k"]
    summary = result.summary.set_index("method")
    unseen = 1 - result.cases["hidden_product_in_training"].mean()
    sizes = result.cases.groupby("invoice").size()
    first, last = source.quality["dates"]["first"][:10], source.quality["dates"]["last"][:10]
    removed_customers = source.prepared.summary["removed_by_reason"].get(uci.R_NO_CUSTOMER, 0)
    notes = [
        f"One retailer, one split: the data covers {first} to {last} for one online retailer (UCI describes it as a "
        "UK-based online gift retailer with many wholesale customers), evaluated on a single chronological split. "
        "The results say nothing certain about other shops or other periods.",
        "This is offline basket completion, not next-purchase prediction: the hidden product comes from the same "
        "invoice, and the rest of that invoice is used as context. In a live shop the order in which products are "
        "added is unknown, so less context may be available.",
        "A hit means the suggestion matched a product the customer bought anyway. It does not show extra sales, "
        "revenue or any causal effect; that needs a controlled experiment such as an A/B test.",
        f"Large invoices contribute many cases (the largest test invoice gives {sizes.max():,} cases; the median "
        f"gives {int(sizes.median()):,}). The invoice-averaged hit rate, which weights every invoice equally, is "
        "reported as a check.",
        f"{unseen:.1%} of the test cases hide a product that was never sold in the training period. No method can "
        "suggest such a product; these cases stay in the denominator for every method.",
        "Cancellations are removed as rows, but the data does not link a cancellation to its original order, so an "
        "order that was later cancelled still counts as a basket.",
        ("The thresholds were chosen from a small grid on a validation period at the end of the training period. "
         "Other values, longer rules or other ranking methods were not tried, so better settings may exist."
         if not result.selection.empty else
         "The thresholds were fixed by the user (a single value each), not chosen on a validation period."),
        (f"Rows without a customer ID were removed in this run ({removed_customers:,} rows), so the baskets are "
         "only those of identified customers." if source.prepared.require_customer_id else
         f"{source.prepared.summary['invoices_without_customer_id']:,} valid invoices have no customer ID. They are "
         "kept, because a basket is an invoice; customer-level questions cannot be answered with them."),
        "The confidence intervals resample test invoices only. They do not cover changes between periods, such "
        "as seasonal peaks inside or outside the test period.",
    ]
    if result.config.max_len == 2:
        refused = result.settings.get("three_product_itemsets_exceed_limit")
        notes.insert(6, "Rules have one product on each side (at most 2 products per rule). "
                     + ("On this training period, allowing three products at the chosen support would exceed "
                        f"BasketWise's pre-mining safety limit of {MAX_FREQUENT_ITEMSETS:,} itemsets, so longer rules "
                        "were not possible with these settings." if refused else
                        "Longer rules were not tried and might find patterns that pairs miss."))
    if uses_default_protocol(source, result):
        notes.append(f"This run matches the default protocol of BasketWise {version_string()} on the official UCI file "
                     "(latest 20% of invoices as the test period, 20% of the training invoices for validation, the "
                     "3 x 3 threshold grid, at most 2 products per rule, K = 3, 1,000 bootstrap resamples with "
                     "seed 42). "
                     "In this project that protocol was fixed before the test period was first scored and was not "
                     "changed after seeing test results (docs/PRODUCT_DECISIONS.md, decisions 20 to 22).")
    else:
        notes.append("This run differs from BasketWise's default protocol or does not use the official UCI file. If "
                     "its settings were chosen after looking at test results, the test period is no longer an "
                     "untouched hold-out.")
    if not result.selection.empty:
        chosen = result.selection[result.selection["chosen"]].iloc[0]
        grid = sorted(result.selection["min_support"].unique())
        if chosen["min_support"] in (grid[0], grid[-1]) and len(grid) > 1:
            edge = "lowest" if chosen["min_support"] == grid[0] else "highest"
            notes.insert(7, f"The chosen minimum support ({chosen['min_support']:.2%}) is the {edge} value in the "
                            "grid, so a value outside the grid might have scored better on validation.")
    if summary.loc[PRIMARY, f"hit rate@{k}"] <= summary.loc[BASELINE, f"hit rate@{k}"]:
        notes.insert(0, "On this test period the association-rule recommender did not beat the best-seller baseline.")
    return notes


def build_markdown(source: PreparedSource, result: TemporalResult, generated: datetime | None = None) -> str:
    generated = generated or datetime.now(UTC)
    data = results_dict(source, result, generated)
    k = result.settings["top_k"]
    hr = f"hit rate@{k}"
    quality, prep, split = source.quality, source.prepared.summary, result.split
    summary = result.summary.set_index("method")
    base, main = summary.loc[BASELINE], summary.loc[PRIMARY]
    intervals = result.intervals
    difference = (main[hr] - base[hr]) * 100
    diff_interval = intervals.get("difference_vs_baseline", {}).get(PRIMARY, {}).get("interval") if intervals else None
    config = result.config
    dataset = data["dataset"]

    if diff_interval and diff_interval[0] > 0:
        verdict = (f"On this test period the association-rule recommender found the hidden product in the top {k} "
                   f"more often than the best sellers did ({_pct(main[hr])} against {_pct(base[hr])}, "
                   f"{difference:+.1f} points; 95% interval {format_points(diff_interval)}).")
    elif diff_interval and diff_interval[1] < 0:
        verdict = (f"On this test period the association-rule recommender found the hidden product LESS often than "
                   f"the best sellers ({_pct(main[hr])} against {_pct(base[hr])}; 95% interval "
                   f"{format_points(diff_interval)}).")
    else:
        verdict = (f"On this test period the difference between the association-rule recommender ({_pct(main[hr])}) "
                   f"and the best sellers ({_pct(base[hr])}) is {difference:+.1f} points"
                   + (f"; the 95% interval ({format_points(diff_interval)}) includes zero." if diff_interval else "."))
    verdict += (f" In absolute terms, the hidden product was not among the {k} suggestions in {_pct(1 - main[hr])} of "
                f"cases for the association-rule recommender and in {_pct(1 - base[hr])} for the best sellers.")

    lines = [
        "# BasketWise real-world chronological validation (UCI Online Retail)",
        "",
        f"Generated {generated:%Y-%m-%d %H:%M} UTC by BasketWise {data['basketwise_version']}. Every number in this "
        "report was computed by the code in this repository from the file named in section 1; none was typed by "
        "hand.",
        "",
        f"**Question.** {QUESTION}",
        "",
        f"**Answer for this data.** {verdict}",
        "",
        "**What this does not show:** " + "; ".join(NOT_MEASURED) + ".",
        "",
        "This is a separate experiment from the synthetic benchmark in `docs/EVALUATION_REPORT.md`: the data, the "
        "split (by time instead of random) and the settings differ, so the two sets of numbers must not be combined "
        "or compared as one benchmark.",
        "",
        "## 1. Dataset source and attribution",
        "",
        markdown_table(pd.DataFrame([
            ("Dataset", "Online Retail, UCI Machine Learning Repository"),
            ("Citation", uci.CITATION),
            ("Licence", f"{uci.LICENCE} — {uci.LICENCE_URL}"),
            ("Source page", uci.SOURCE_PAGE),
            ("Download URL", uci.DOWNLOAD_URL),
            ("Retrieved (UTC)", dataset["retrieved_utc"] or ("unknown (the ZIP was placed in the folder by hand)"
                                                 if source.manifest else "unknown (no download manifest found)")),
            ("File used", f"{dataset['file']} (SHA-256 {dataset['file_sha256']})"),
            ("Official spreadsheet SHA-256", dataset["official_spreadsheet_sha256"] or "unknown"),
        ], columns=["item", "value"])),
        "",
        "The raw data is not stored in this repository. "
        + ("This file was made by `scripts/download_uci_online_retail.py`, which records the source, retrieval time "
           "and checksums in `download_manifest.json`"
           + (", and it is the same file the published results in `docs/real_world/` were computed from."
              if dataset["same_file_as_published_results"] else
              ". Its spreadsheet checksum differs from the one the published results were computed from, so the "
              "numbers may differ from `docs/real_world/`.")
           if source.manifest else
           "No download manifest was found next to this file, so its origin cannot be confirmed. Use "
           "`scripts/download_uci_online_retail.py` to get the official file."),
        "",
        "## 2. Data-quality findings (raw table)",
        "",
        markdown_table(pd.DataFrame([
            ("Rows", f"{quality['rows']:,}"),
            ("Invoices (all)", f"{quality['invoices']['total']:,}"),
            ("Cancellation invoices (number starts with C)", f"{quality['invoices']['cancellation_invoices']:,} "
             f"({quality['cancellations']['rows']:,} rows, of which "
             f"{quality['cancellations']['rows_with_negative_quantity']:,} have a negative quantity)"),
            ("Other non-numeric invoices", f"{quality['invoices']['other_non_numeric_invoices']:,}"
             + (f" ({', '.join(quality['invoices']['other_non_numeric_examples'])})"
                if quality['invoices']['other_non_numeric_examples'] else "")),
            ("Stock codes (as written / ignoring case)", f"{quality['products']['stock_codes']:,} / "
             f"{quality['products']['stock_codes_ignoring_case']:,}"),
            ("Distinct descriptions", f"{quality['products']['descriptions']:,}"),
            ("Stock codes with more than one description",
             f"{quality['products']['stock_codes_with_several_descriptions']:,}"),
            ("Descriptions shared by several stock codes",
             f"{quality['products']['descriptions_shared_by_several_stock_codes']:,}"),
            ("Customers (distinct CustomerID)", f"{quality['customers']['unique_customer_ids']:,}"),
            ("Rows without CustomerID", f"{quality['customers']['rows_without_customer_id']:,}"),
            ("Missing or placeholder descriptions", f"{quality['descriptions']['missing_rows']:,}"),
            ("Note-like descriptions (lower case, no word in capitals)",
             f"{quality['descriptions']['note_like_rows']:,}"
             + (" (most common: " + ", ".join(f"'{e}'" for e in quality["descriptions"]["note_like_examples"][:5])
                + ")" if quality["descriptions"]["note_like_examples"] else "")),
            ("Exact duplicate rows", f"{quality['exact_duplicate_rows']:,}"),
            ("Negative quantities (all / outside cancellations)", f"{quality['quantity']['negative_rows']:,} / "
             f"{quality['quantity']['negative_rows_outside_cancellations']:,}"),
            ("Zero quantities", f"{quality['quantity']['zero_rows']:,}"),
            ("Zero / negative unit prices", f"{quality['unit_price']['zero_rows']:,} / "
             f"{quality['unit_price']['negative_rows']:,}"),
            ("Invalid or missing dates", f"{quality['dates']['invalid_or_missing_rows']:,}"),
            ("Date range", f"{quality['dates']['first']} to {quality['dates']['last']} "
             f"({quality['dates']['days_with_invoices']} days with invoices)"),
            ("Service and accounting lines (postage, fees, ...)", f"{quality['products']['non_merchandise_rows']:,}"),
            ("Countries", f"{quality['countries']['count']}"),
        ], columns=["check", "observed"])),
        "",
        "Rows per country (top 10): " + ", ".join(f"{country} {count:,}" for country, count in
                                                 list(quality["countries"]["rows_by_country"].items())[:10]) + ".",
        "",
        "Regular invoices per month: " + ", ".join(f"{month} {count:,}" for month, count in
                                                  quality["invoices"]["regular_invoices_per_month"].items()) + ".",
        "",
        "## 3. Preprocessing decisions",
        "",
        "- **Baskets are invoices.** Products are identified by stock code, trimmed and in upper case: "
        f"{quality['products']['lower_case_stock_codes_with_upper_case_twin']} of the "
        f"{quality['products']['lower_case_stock_codes']} lower-case codes (such as 85123a) also appear in upper "
        f"case, and {quality['products']['lower_case_twins_sharing_a_description']} of those share a description "
        "with their upper-case twin. Descriptions are display names only (the most frequent description of a "
        f"code is used), because {quality['products']['descriptions_shared_by_several_stock_codes']} descriptions "
        "belong to several different codes.",
        "- **Returns and cancellations are never purchases.** Every row of a cancellation invoice is removed, and so "
        "is every row with a zero or negative quantity elsewhere (returns or stock adjustments).",
        "- **Non-sale invoices and service lines are removed:** invoice numbers with another letter (bad-debt "
        "adjustments), and stock codes for postage, carriage, fees, commissions, manual entries, discounts, samples "
        "and gift vouchers (" + ", ".join(sorted(uci.NON_MERCHANDISE_CODES)) + ", GIFT_0001_*).",
        "- **Rows with a missing, zero or negative unit price are removed.** The raw table has "
        f"{quality['unit_price']['zero_rows']:,} zero-price rows "
        f"({quality['unit_price']['zero_price_rows_with_customer_id']:,}"
        f" of them with a customer ID) and {quality['unit_price']['negative_rows']:,} negative prices; rows already "
        "removed for an earlier reason are counted only once.",
        "- **Missing CustomerID is kept**" + (" — except in this run, which required customer IDs" if
                                               prep["require_customer_id"] else "") + ": a basket is an invoice, "
        "and customer IDs are not used by the methods.",
        "- **Duplicates:** exact duplicate rows are removed; other repeated lines of the same product in one invoice "
        "are merged into one basket line (quantities added). Basket analysis only uses whether a product is in "
        "the basket.",
        "- **Dates** must be ISO date-times (as written by the download script); anything else is reported as "
        "invalid, never guessed. An invoice's time is its earliest line "
        f"({quality['invoices']['regular_invoices_with_several_timestamps']:,} invoices have lines with more than one "
        f"timestamp, at most {_minutes(quality['invoices']['longest_time_span_within_an_invoice_minutes'])} "
        "apart).",
        "- Each removed row keeps its first applicable reason in `removed_rows.csv.gz`, with its line number in "
        "the CSV.",
        "",
        markdown_table(uci.preprocessing_table(source.prepared).assign(
            count=lambda t: t["count"].map(lambda v: f"{v:,}"))),
        "",
        "## 4. Training and test periods",
        "",
        markdown_table(pd.DataFrame([
            ("Cutoff (first moment of the test period)", split["cutoff"]),
            ("How the cutoff was chosen", split["rule"]),
            ("Training period", f"{split['training']['first_invoice']} to {split['training']['last_invoice']}"),
            ("Training invoices", f"{split['training']['invoices']:,}"),
            ("Test period", f"{split['test']['first_invoice']} to {split['test']['last_invoice']}"),
            ("Test invoices (with 2+ products)", f"{split['test']['invoices']:,} "
             f"({split['test']['invoices_with_two_or_more_products']:,})"),
            ("Share of invoices in the test period", _pct(split["test_share_of_invoices"])),
        ], columns=["item", "value"])),
        "",
        "Every training invoice is earlier than every test invoice, and no invoice is split between the periods.",
        "",
        "## 5. Evaluation method",
        "",
        "1. Rules and best sellers are learned from training-period invoices only.",
        "2. In every test invoice with at least two products, each product is hidden once. The other products of "
        f"that invoice are the query, and each method suggests up to {k} products that are not already in the "
        "query.",
        f"3. All methods are scored on exactly the same {int(base['cases']):,} cases from "
        f"{int(base['invoices']):,} test invoices ({int(base['customers']):,} distinct customers among invoices "
        "with a customer ID).",
        f"4. **Candidate universe:** the {result.settings['candidate_universe']:,} products bought in the training "
        "period; nothing else can be suggested. **Denominator:** every case above, including cases whose hidden "
        "product was never sold in training (a miss for every method).",
        "",
        "Metrics in simple words:",
        "",
        f"- **Hit Rate@{k}**: the share of cases where the hidden product is among the {k} suggestions.",
        f"- **Precision@{k}**: the share of the {k} suggestion slots that hold the hidden product, averaged over "
        f"cases. With one hidden product per case it is Hit Rate@{k} divided by {k}, so it can be at most "
        f"{1 / k:.1%}.",
        f"- **Recall@{k}**: the share of hidden products that were suggested. With one hidden product per case it "
        f"equals Hit Rate@{k}.",
        "- **Coverage**: the share of the candidate universe that a method suggested at least once.",
        "- **Availability**: the share of cases with at least one suggestion.",
        f"- **Invoice-averaged Hit Rate@{k}**: the hit rate of each invoice, averaged over invoices, so that large "
        "invoices do not dominate.",
        f"- **95% intervals**: from {intervals.get('resamples', 0):,} bootstrap resamples of whole test invoices "
        f"(seed {intervals.get('seed', '-')}), the same resamples for every method." if intervals else
        "- No bootstrap intervals were computed in this run.",
        "",
        "## 6. Threshold selection (validation period inside the training period)",
        "",
    ]
    if result.selection.empty:
        lines += ["A single threshold setting was given, so no selection was made.", ""]
    else:
        v = result.validation
        lines += [
            f"Candidate settings were compared on a validation period from {v['validation_part']['first_invoice']} "
            f"to {v['validation_part']['last_invoice']} ({v['validation_part']['invoices']:,} invoices, "
            f"{v['cases']:,} cases), with rules learned from {v['training_part']['first_invoice']} to "
            f"{v['training_part']['last_invoice']}. The test period was not used. On validation, the best-seller "
            f"baseline scored {_pct(v[f'baseline {hr}'])}.",
            "",
            markdown_table(_percent_text(selection_table(result.selection, hr), digits=2)),
            "",
        ]
    lines += [
        "## 7. Results on the test period",
        "",
        f"Chosen setting: minimum support {config.min_support:.2%}, minimum confidence {config.min_confidence:.0%}, "
        f"minimum lift {config.min_lift:g}, at most {config.max_len or 'any number of'} products per rule "
        f"({ALGORITHM_NAMES[config.algorithm]}). Learned from the training period: {result.n_itemsets:,} frequent "
        f"itemsets and {len(result.rules):,} rules.",
        "",
        "The association-rule recommender below is BasketWise's default: rules ranked by confidence, with "
        "training-period best sellers filling any free slots. The best-seller baseline suggests the training "
        "period's most popular products that are not already in the basket.",
        "",
        markdown_table(comparison_table(result, k)),
        "",
        "All methods (rules-only methods leave a slot empty when no rule applies):",
        "",
        markdown_table(_percent_text(methods_table(result, k))),
        "",
        f"Hit Rate@{k} per test week (the first and last weeks can be partial):",
        "",
        markdown_table(_percent_text(weekly_table(result, k))),
        "",
        "Examples of rules learned from the training period (strongest confidence first):",
        "",
        markdown_table(pd.DataFrame(data["training_rules"]["top_rules_by_confidence"]), digits=3),
        "",
        "## 8. Evaluation settings",
        "",
        markdown_table(pd.DataFrame([
            ("Recommendations per case (K)", k),
            ("Test share / fixed cutoff", result.settings["fixed_cutoff"] or _pct(result.settings["test_share"], 0)),
            ("Validation share of the training period", _pct(result.settings["validation_share"], 0)
             if result.settings["validation_share"] else "no selection"),
            ("Candidate minimum supports",
             ", ".join(f"{s:.2%}" for s in data["settings"]["candidates"]["min_support"])),
            ("Candidate minimum confidences", ", ".join(f"{c:.0%}" for c in
                                                        data["settings"]["candidates"]["min_confidence"])),
            ("Candidate minimum lifts", ", ".join(f"{x:g}" for x in data["settings"]["candidates"]["min_lift"])),
            ("Chosen minimum support / confidence / lift", f"{config.min_support:.2%} / {config.min_confidence:.0%} "
                                                           f"/ {config.min_lift:g}"),
            ("Maximum products per rule", config.max_len or "no limit"),
            ("Algorithm", ALGORITHM_NAMES[config.algorithm] + " (mlxtend)"),
            ("Recommender ranking", "confidence, with best-seller fallback (BasketWise default)"),
            ("Bootstrap resamples / seed", f"{result.settings['bootstrap_resamples']:,} / {result.settings['seed']}"),
            ("Customer IDs required", "yes" if prep["require_customer_id"] else "no"),
        ], columns=["setting", "value"])),
        "",
        "## 9. Limitations",
        "",
        *[f"- {note}" for note in data["limitations"]],
        "",
        "## 10. How to reproduce",
        "",
        "```",
        "python scripts/download_uci_online_retail.py   # official ZIP -> Data/external/uci_online_retail/",
        "python -m market_basket uci-prepare            # data-quality report and preprocessing summary",
        "python -m market_basket uci-evaluate           # this report and the other files in outputs/real_world/",
        "```",
        "",
        "The same evaluation can be run from the Evaluation Lab page of the app (tab 'Real-world validation'). "
        "With the same file and settings the results are identical on every run; only the generation time "
        "changes.",
        "",
    ]
    return "\n".join(lines)


def _percent_text(table: pd.DataFrame, digits: int = 1) -> pd.DataFrame:
    """Markdown copy of a table with every share shown as a percentage and counts with thousands separators."""
    table = table.copy()
    shares = ("hit rate", "precision", "recall", "availability", "coverage", "baseline", "recommender")
    for column in table.columns:
        if table[column].dtype.kind == "f" and any(word in str(column) for word in shares):
            table[column] = table[column].map(lambda v: _pct(v, digits))
        elif table[column].dtype.kind in "iu" and str(column) in ("invoices", "cases", "rules", "itemsets"):
            table[column] = table[column].map(lambda v: f"{v:,}")
    return table


def selection_table(selection: pd.DataFrame, hr: str) -> pd.DataFrame:
    table = selection.copy()
    table["min_support"] = table["min_support"].map(lambda v: f"{v:.2%}")
    table["min_confidence"] = table["min_confidence"].map(lambda v: f"{v:.0%}")
    table["min_lift"] = table["min_lift"].map(lambda v: f"{v:g}")
    table["chosen"] = table["chosen"].map(lambda v: "yes" if v else "")
    keep = ["min_support", "min_confidence", "min_lift", "rules", f"validation {hr}", "validation coverage",
            "validation rank", "status", "chosen"]
    return table[[c for c in keep if c in table]]


def comparison_table(result: TemporalResult, k: int) -> pd.DataFrame:
    summary = result.summary.set_index("method")
    intervals = result.intervals
    rows = []
    for metric, label in ((f"hit rate@{k}", f"Hit Rate@{k}"), (f"precision@{k}", f"Precision@{k}"),
                          (f"recall@{k}", f"Recall@{k}"), ("coverage", "Coverage"), ("availability", "Availability"),
                          (f"invoice-averaged hit rate@{k}", f"Invoice-averaged Hit Rate@{k}")):
        rows.append((label, _pct(summary.loc[BASELINE, metric]), _pct(summary.loc[PRIMARY, metric])))
    if intervals:
        rows.append((f"Hit Rate@{k} 95% interval", format_interval(intervals["hit_rate"][BASELINE]),
                     format_interval(intervals["hit_rate"][PRIMARY])))
    rows.append(("Evaluated cases", f"{int(summary.loc[BASELINE, 'cases']):,}",
                 f"{int(summary.loc[PRIMARY, 'cases']):,}"))
    return pd.DataFrame(rows, columns=["Metric", "Best-seller baseline", "Association-rule recommender"])


def methods_table(result: TemporalResult, k: int) -> pd.DataFrame:
    table = result.summary[["method", f"hit rate@{k}", f"precision@{k}", f"recall@{k}", "availability", "coverage",
                            f"invoice-averaged hit rate@{k}", f"hit rate@{k}, hidden product seen in training"]].copy()
    if result.intervals:
        table["difference vs baseline (95% interval)"] = [
            "-" if name == BASELINE else format_points(result.intervals["difference_vs_baseline"][name]["interval"])
            for name in table["method"]]
    return table


def weekly_table(result: TemporalResult, k: int) -> pd.DataFrame:
    weekly = result.weekly
    columns = ["week starting", "invoices", "cases", f"hit rate@{k}: {BASELINE}", f"hit rate@{k}: {PRIMARY}"]
    table = weekly[columns].rename(columns={f"hit rate@{k}: {BASELINE}": "best-seller baseline",
                                            f"hit rate@{k}: {PRIMARY}": "association-rule recommender"})
    return table.assign(**{"week starting": table["week starting"].dt.strftime("%Y-%m-%d")})


# ---------------------------------------------------------------- files


def write_preparation_outputs(source: PreparedSource, output_dir: str | Path) -> list[Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    quality = {"dataset": dataset_info(source), **source.quality}
    files = {
        "data_quality_report.json": json.dumps(json_safe(quality), indent=2, ensure_ascii=False) + "\n",
        "preprocessing_summary.csv": uci.preprocessing_table(source.prepared).to_csv(index=False, lineterminator="\n"),
    }
    paths = []
    for name, text in files.items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        paths.append(out / name)
    removed = out / "removed_rows.csv.gz"
    source.prepared.removed.to_csv(removed, index=False, compression={"method": "gzip", "mtime": 0},
                                   lineterminator="\n")
    return [*paths, removed]


def evaluation_files(source: PreparedSource, result: TemporalResult, generated: datetime | None = None) -> dict:
    """Every evaluation output as {file name: text}, used for the output folder and the app's ZIP download."""
    generated = generated or datetime.now(UTC)
    labels = source.prepared.labels()
    rules = result.rules.assign(**{"if the basket has": result.rules["antecedents"].map(
        lambda s: as_text(labels.get(p, p) for p in s)), "suggest": result.rules["consequents"].map(
        lambda s: as_text(labels.get(p, p) for p in s))})
    rules = rules[["if the basket has", "suggest", "support", "confidence", "lift", "leverage", "conviction"]]
    weekly = result.weekly.assign(**{"week starting": result.weekly["week starting"].dt.strftime("%Y-%m-%d")})

    def csv(frame: pd.DataFrame) -> str:
        return frame.to_csv(index=False, lineterminator="\n")

    return {
        "real_world_evaluation.json": json.dumps(results_dict(source, result, generated), indent=2,
                                                 ensure_ascii=False) + "\n",
        "real_world_evaluation.md": build_markdown(source, result, generated),
        "evaluation_summary.csv": csv(result.summary),
        "weekly_results.csv": csv(weekly),
        "threshold_selection.csv": csv(result.selection),
        "training_rules.csv": csv(rules),
    }


def cases_csv(result: TemporalResult) -> pd.DataFrame:
    """The scored cases without customer IDs (they are not needed to audit the scores)."""
    return result.cases.drop(columns="customer_id").assign(time=result.cases["time"].dt.strftime("%Y-%m-%d %H:%M:%S"))


def write_evaluation_outputs(source: PreparedSource, result: TemporalResult, output_dir: str | Path,
                             generated: datetime | None = None) -> list[Path]:
    out = Path(output_dir)
    paths = write_preparation_outputs(source, out)
    for name, text in evaluation_files(source, result, generated).items():
        (out / name).write_text(text, encoding="utf-8", newline="\n")
        paths.append(out / name)
    cases = out / "evaluation_cases.csv.gz"
    cases_csv(result).to_csv(cases, index=False, compression={"method": "gzip", "mtime": 0}, lineterminator="\n")
    return [*paths, cases]


def export_zip(source: PreparedSource, result: TemporalResult, generated: datetime | None = None) -> bytes:
    """All evaluation outputs plus the data-quality and preprocessing files, as one ZIP (for the app)."""
    files = evaluation_files(source, result, generated)
    files["data_quality_report.json"] = json.dumps(json_safe({"dataset": dataset_info(source), **source.quality}),
                                                   indent=2, ensure_ascii=False) + "\n"
    files["preprocessing_summary.csv"] = uci.preprocessing_table(source.prepared).to_csv(index=False,
                                                                                        lineterminator="\n")
    files["evaluation_cases.csv"] = cases_csv(result).to_csv(index=False, lineterminator="\n")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, text in files.items():
            archive.writestr(name, text)
    return buffer.getvalue()
