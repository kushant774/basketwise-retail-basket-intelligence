"""Downloadable outputs: a Markdown analysis report and a ZIP of CSV exports."""
from __future__ import annotations

import io
import zipfile
from datetime import datetime

import pandas as pd

from .config import ALGORITHM_NAMES
from .data import TransactionData, item_support, summarize
from .evaluation import EvaluationResult
from .insights import (
    bundle_opportunities,
    category_summary,
    cross_sell_opportunities,
    product_groups,
    product_popularity,
)
from .mining import MiningResult, explain_rule, readable


def markdown_table(frame: pd.DataFrame, digits: int = 3) -> str:
    """Render a small DataFrame as a Markdown table without extra dependencies."""
    if frame.empty:
        return "_No rows._"

    def cell(value) -> str:
        if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NA:
            return ""
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value).replace("|", "/")

    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    divider = "|" + "---|" * len(frame.columns)
    body = ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *body])


def build_report(data: TransactionData, result: MiningResult, evaluation: EvaluationResult | None = None,
                 generated: datetime | None = None, top_n: int = 10) -> str:
    generated = generated or datetime.now()
    config = result.config
    support = item_support(data.transactions)
    summary = summarize(data)
    lines = [
        "# BasketWise analysis report",
        "",
        f"Generated {generated:%Y-%m-%d %H:%M}. Data source: {data.source}"
        + (" (synthetic data, not real customers or sales)." if data.is_synthetic else "."),
        "",
        "Associations describe products that were bought together in this data. They do not prove that one "
        "product causes the purchase of another, and the opportunities below should be tested before rollout.",
        "",
        "## Dataset",
        "",
        markdown_table(pd.DataFrame([(k, str(v)) for k, v in summary.items()], columns=["measure", "value"])),
        "",
        "## Data validation",
        "",
        markdown_table(data.report.summary_table()),
        "",
        *[f"- {warning}" for warning in data.report.warnings],
        "",
        "## Mining settings",
        "",
        f"Algorithm {ALGORITHM_NAMES[config.algorithm]} (mlxtend); minimum support {config.min_support:.2%}; "
        f"minimum confidence {config.min_confidence:.0%}; minimum lift {config.min_lift:.2f}; maximum products per "
        f"rule {config.max_len or 'no limit'}"
        + (f"; ignored products (left out of rules, recommendations and bundles): {', '.join(config.exclude)}"
           if config.exclude else "") + ".",
        f"Result: {len(result.frequent_itemsets):,} frequent itemsets and {len(result.rules):,} rules.",
        "",
        f"## Top {top_n} rules by lift",
        "",
    ]
    top_rules = result.rules.head(top_n)
    if top_rules.empty:
        lines.append("_No rules pass the current thresholds._")
    else:
        lines.append(markdown_table(readable(top_rules)[["antecedents", "consequents", "support", "confidence", "lift"]]))
        lines += ["", "In plain words:", ""]
        for rule in top_rules.head(3).to_dict("records"):
            lines.append("- " + explain_rule(rule["antecedents"], rule["consequents"], rule["support"],
                                             rule["confidence"], rule["lift"], rule["consequent support"]))
    lines += [
        "",
        "## Potential cross-selling opportunities",
        "",
        markdown_table(cross_sell_opportunities(result.rules, data.categories, top_n)),
        "",
        "## Potential bundles",
        "",
        markdown_table(bundle_opportunities(result, support, data.categories, top_n)),
        "",
        "## Products often bought together (merchandising groups, lift >= 2)",
        "",
        *([f"- {', '.join(group)}" for group in product_groups(result.rules)] or ["_No groups at this lift._"]),
        "",
        "## Most and least purchased products",
        "",
    ]
    popularity = product_popularity(data)
    keep = [c for c in ("product", "category", "transactions", "share of transactions") if c in popularity]
    lines += [markdown_table(popularity[keep].head(5)), "", markdown_table(popularity[keep].tail(5)), ""]
    categories = category_summary(data)
    if len(categories):
        lines += ["## Categories", "", markdown_table(categories), ""]
    if evaluation is not None:
        s = evaluation.settings
        lines += [
            "## Recommendation evaluation (offline)",
            "",
            f"{s['n_splits']} random splits, {1 - s['test_size']:.0%} train / {s['test_size']:.0%} test, seeds "
            f"{s['seed']}-{s['seed'] + s['n_splits'] - 1}. In every test transaction with two or more products, "
            "each product is hidden in turn and the methods recommend from the rest; a hit means the hidden "
            "product appears in the top k. Rules and best sellers come from the training split only. Three "
            "recommendation methods are compared with a best-seller baseline.",
            "",
            markdown_table(evaluation.summary, digits=4),
            "",
            "Offline hit rate shows whether recommendations match what customers bought anyway. It does not "
            "measure extra sales; that needs a controlled experiment such as an A/B test.",
            "",
        ]
    return "\n".join(lines)


def export_zip(data: TransactionData, result: MiningResult, evaluation: EvaluationResult | None = None,
               generated: datetime | None = None) -> bytes:
    """ZIP file with the report and CSV exports of the data, rules and insights."""
    support = item_support(data.transactions)
    files = {
        "report.md": build_report(data, result, evaluation, generated),
        "cleaned_transactions.csv": data.items.to_csv(index=False),
        "removed_rows.csv": data.removed.to_csv(index=False),
        "rules.csv": readable(result.rules).to_csv(index=False),
        "frequent_itemsets.csv": readable(result.frequent_itemsets).to_csv(index=False),
        "cross_sell_opportunities.csv": cross_sell_opportunities(result.rules, data.categories, 50).to_csv(index=False),
        "bundle_opportunities.csv": bundle_opportunities(result, support, data.categories, 50).to_csv(index=False),
        "product_popularity.csv": product_popularity(data).to_csv(index=False),
    }
    if evaluation is not None:
        files["evaluation_summary.csv"] = evaluation.summary.to_csv(index=False)
        files["evaluation_per_split.csv"] = evaluation.per_split.to_csv(index=False)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    return buffer.getvalue()
