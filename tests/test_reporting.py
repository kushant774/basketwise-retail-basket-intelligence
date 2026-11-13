import io
import zipfile
from datetime import datetime

import pandas as pd

from market_basket.config import MiningConfig
from market_basket.evaluation import evaluate
from market_basket.mining import mine_rules
from market_basket.reporting import build_report, export_zip, markdown_table

FIXED_TIME = datetime(2026, 1, 15, 9, 30)


def test_report_has_all_sections_and_honest_wording(demo):
    result = mine_rules(demo.transactions)
    report = build_report(demo, result, evaluate(demo.transactions, n_splits=1), generated=FIXED_TIME)
    for heading in ("# BasketWise analysis report", "## Dataset", "## Data validation", "## Mining settings",
                    "## Top 10 rules by lift", "## Potential cross-selling opportunities", "## Potential bundles",
                    "## Recommendation evaluation (offline)"):
        assert heading in report
    assert "Generated 2026-01-15 09:30" in report
    assert "synthetic data, not real customers or sales" in report
    assert "do not prove that one product causes" in report
    assert f"{len(result.rules):,} rules" in report
    assert "A/B test" in report


def test_report_with_no_rules(demo):
    result = mine_rules(demo.transactions, MiningConfig(min_support=0.9))
    report = build_report(demo, result, generated=FIXED_TIME)
    assert "_No rules pass the current thresholds._" in report
    assert "## Recommendation evaluation" not in report


def test_zip_export_contains_consistent_csv_files(demo):
    result = mine_rules(demo.transactions)
    archive = zipfile.ZipFile(io.BytesIO(export_zip(demo, result, evaluate(demo.transactions, n_splits=1),
                                                    generated=FIXED_TIME)))
    names = set(archive.namelist())
    assert {"report.md", "cleaned_transactions.csv", "removed_rows.csv", "rules.csv", "frequent_itemsets.csv",
            "cross_sell_opportunities.csv", "bundle_opportunities.csv", "product_popularity.csv",
            "evaluation_summary.csv", "evaluation_per_split.csv"} == names
    rules = pd.read_csv(archive.open("rules.csv"))
    assert len(rules) == len(result.rules)
    cleaned = pd.read_csv(archive.open("cleaned_transactions.csv"))
    assert cleaned["transaction_id"].nunique() == 8000


def test_markdown_table_formatting():
    table = markdown_table(pd.DataFrame({"name": ["a|b", None], "value": [0.12345, float("nan")]}), digits=2)
    assert table.splitlines() == ["| name | value |", "|---|---|", "| a/b | 0.12 |", "|  |  |"]
    assert markdown_table(pd.DataFrame()) == "_No rows._"


def test_zip_export_works_when_no_rules_are_found(demo):
    result = mine_rules(demo.transactions, MiningConfig(min_support=0.9))
    archive = zipfile.ZipFile(io.BytesIO(export_zip(demo, result, generated=FIXED_TIME)))
    assert pd.read_csv(archive.open("rules.csv")).empty
    assert "_No rules pass the current thresholds._" in archive.read("report.md").decode("utf-8")
