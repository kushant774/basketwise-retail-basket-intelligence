"""Command-line interface: python -m market_basket <command> [options]."""
from __future__ import annotations

import argparse
import contextlib
import math
import sys
from pathlib import Path

import pandas as pd

from .config import ALGORITHMS, DEMO_DATA_PATH, RECOMMENDATION_RANK_METRICS, RULE_RANK_METRICS, MiningConfig
from .data import (
    DataValidationError,
    TransactionData,
    item_support,
    load_demo_data,
    load_transactions,
    summarize,
    validate_products,
)
from .evaluation import evaluate
from .insights import bundle_opportunities, cross_sell_opportunities
from .mining import mine_rules, rank_rules, readable
from .recommend import Recommender, recommendations_table
from .reporting import build_report, export_zip

PRINTED_RULE_COLUMNS = ["antecedents", "consequents", "support", "confidence", "lift", "leverage", "conviction"]
DEFAULTS = MiningConfig()


def _finite_float(text: str) -> float:
    try:
        value = float(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from error
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError(f"must be a finite number, got {text!r}")
    return value


def _count(minimum: int):
    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError as error:
            raise argparse.ArgumentTypeError(f"not a whole number: {text!r}") from error
        if value < minimum:
            raise argparse.ArgumentTypeError(f"must be at least {minimum}, got {value}")
        return value
    return parse


def _table(frame: pd.DataFrame) -> str:
    return frame.to_string(index=False, float_format=lambda value: f"{value:.4f}")


def _load(args: argparse.Namespace) -> TransactionData:
    if args.data is None or Path(args.data).resolve() == DEMO_DATA_PATH.resolve():
        return load_demo_data()
    if Path(args.data).is_dir():
        raise DataValidationError(f"'{args.data}' is a folder; give the path of a CSV file.")
    return load_transactions(args.data)


def _config(args: argparse.Namespace, catalog: list[str]) -> MiningConfig:
    return MiningConfig(args.min_support, args.min_confidence, args.min_lift,
                        None if args.max_len == 0 else args.max_len, args.algorithm,
                        tuple(validate_products(args.exclude or [], catalog)))


def cmd_summary(args, data) -> None:
    print(f"Source: {data.source}" + (" (synthetic)" if data.is_synthetic else ""))
    for key, value in summarize(data).items():
        print(f"  {key:<22} {value}")
    print(f"\nTop {args.top} products by share of transactions")
    print(_table(item_support(data.transactions).head(args.top).reset_index()))


def cmd_validate(args, data) -> None:
    print(f"Validation summary for {data.source}")
    print(_table(data.report.summary_table()))
    for warning in data.report.warnings:
        print(f"  warning: {warning}")
    if args.removed_output:
        Path(args.removed_output).parent.mkdir(parents=True, exist_ok=True)
        data.removed.to_csv(args.removed_output, index=False)
        if len(data.removed):
            print(f"{len(data.removed):,} removed rows written to {args.removed_output}")
        else:
            print(f"No rows were removed; {args.removed_output} contains only the header.")


def cmd_rules(args, data) -> None:
    result = mine_rules(data.transactions, _config(args, data.catalog))
    print(f"{len(result.frequent_itemsets):,} frequent itemsets, {len(result.rules):,} rules")
    if result.rules.empty:
        print("No rules pass these thresholds. Try a lower --min-support, --min-confidence or --min-lift.")
        return
    rules = rank_rules(result.rules, args.rank_by)
    table = readable(rules)
    print(_table(table[PRINTED_RULE_COLUMNS].head(args.top)))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\nAll {len(rules):,} rules written to {args.output}")


def cmd_recommend(args, data) -> None:
    basket = validate_products(args.products, data.catalog)
    config = _config(args, data.catalog)
    result = mine_rules(data.transactions, config)
    best_sellers = None if args.no_fallback else item_support(data.transactions)
    recs = Recommender(result.rules, args.rank_by, best_sellers, config.exclude).recommend(basket, top_n=args.top)
    print(f"Basket: {', '.join(basket) or '(empty)'}   ranked by {args.rank_by}"
          + (f"   ignoring: {', '.join(config.exclude)}" if config.exclude else ""))
    if not recs:
        print("No recommendation: no association rule applies to this basket"
              + ("." if args.no_fallback else ", and no best seller is left to suggest."))
        return
    table = recommendations_table(recs)
    print(_table(table.drop(columns="reason")))
    for rec in recs:
        print(f"  {rec.product}: {rec.reason}")


def cmd_insights(args, data) -> None:
    result = mine_rules(data.transactions, _config(args, data.catalog))
    print("Potential cross-selling opportunities")
    print(_table(cross_sell_opportunities(result.rules, data.categories, args.top)))
    print("\nPotential bundles")
    print(_table(bundle_opportunities(result, item_support(data.transactions), data.categories, args.top)))


def cmd_evaluate(args, data) -> None:
    result = evaluate(data.transactions, _config(args, data.catalog), n_splits=args.splits,
                      test_size=args.test_size, seed=args.seed)
    print(f"Leave-one-out evaluation over {args.splits} random splits "
          f"({1 - args.test_size:.0%} train / {args.test_size:.0%} test)")
    print(_table(result.summary))


def cmd_report(args, data) -> None:
    config = _config(args, data.catalog)
    result = mine_rules(data.transactions, config)
    evaluation = evaluate(data.transactions, config, n_splits=args.splits) if args.splits else None
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(build_report(data, result, evaluation), encoding="utf-8")
    (out / "basketwise_export.zip").write_bytes(export_zip(data, result, evaluation))
    print(f"Wrote {out / 'report.md'} and {out / 'basketwise_export.zip'}")


def build_parser() -> argparse.ArgumentParser:
    data_options = argparse.ArgumentParser(add_help=False)
    data_options.add_argument("--data", default=None,
                              help="transaction CSV (long or wide format); default: the synthetic demo data")

    mining_options = argparse.ArgumentParser(add_help=False)
    mining_options.add_argument("--min-support", type=_finite_float, default=DEFAULTS.min_support,
                                help=f"share of transactions, above 0 and at most 1 (default {DEFAULTS.min_support})")
    mining_options.add_argument("--min-confidence", type=_finite_float, default=DEFAULTS.min_confidence,
                                help=f"between 0 and 1 (default {DEFAULTS.min_confidence})")
    mining_options.add_argument("--min-lift", type=_finite_float, default=DEFAULTS.min_lift,
                                help=f"0 or more (default {DEFAULTS.min_lift})")
    mining_options.add_argument("--max-len", type=_count(0), default=DEFAULTS.max_len,
                                help=f"largest itemset size, at least 2; 0 means no limit (default {DEFAULTS.max_len})")
    mining_options.add_argument("--algorithm", choices=ALGORITHMS, default=DEFAULTS.algorithm)
    mining_options.add_argument("--exclude", action="append", metavar="PRODUCT",
                                help="leave this product out of rules, recommendations and bundles (repeatable)")
    both = [data_options, mining_options]

    parser = argparse.ArgumentParser(prog="python -m market_basket", description="BasketWise command line.")
    commands = parser.add_subparsers(dest="command", required=True)

    p = commands.add_parser("summary", parents=[data_options], help="dataset summary and best sellers")
    p.add_argument("--top", type=_count(1), default=10)
    p.set_defaults(handler=cmd_summary)

    p = commands.add_parser("validate", parents=[data_options], help="validation report for a CSV file")
    p.add_argument("--removed-output", help="write removed/rejected rows to this CSV")
    p.set_defaults(handler=cmd_validate)

    p = commands.add_parser("rules", parents=both, help="mine and rank association rules")
    p.add_argument("--top", type=_count(1), default=15)
    p.add_argument("--rank-by", choices=RULE_RANK_METRICS, default="lift")
    p.add_argument("--output", help="write all rules to this CSV file")
    p.set_defaults(handler=cmd_rules)

    p = commands.add_parser("recommend", parents=both, help="explainable recommendations for a basket")
    p.add_argument("products", nargs="*", help='products in the basket, e.g. "Milk" "Bread"')
    p.add_argument("--top", type=_count(1), default=5)
    p.add_argument("--rank-by", choices=RECOMMENDATION_RANK_METRICS, default="confidence")
    p.add_argument("--no-fallback", action="store_true", help="do not fill free slots with best sellers")
    p.set_defaults(handler=cmd_recommend)

    p = commands.add_parser("insights", parents=both, help="cross-selling and bundle opportunities")
    p.add_argument("--top", type=_count(1), default=10)
    p.set_defaults(handler=cmd_insights)

    p = commands.add_parser("evaluate", parents=both, help="offline evaluation against best sellers")
    p.add_argument("--splits", type=_count(1), default=5)
    p.add_argument("--test-size", type=_finite_float, default=0.2, help="share of transactions held out (0-1)")
    p.add_argument("--seed", type=_count(0), default=42)
    p.set_defaults(handler=cmd_evaluate)

    p = commands.add_parser("report", parents=both, help="write report.md and a ZIP of CSV exports")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--splits", type=_count(0), default=5, help="evaluation splits to include (0 to skip)")
    p.set_defaults(handler=cmd_report)
    return parser


def _utf8_output() -> None:
    """Product names can contain any character; a redirected Windows console defaults to cp1252."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure") and (stream.encoding or "").lower().replace("-", "") != "utf8":
            with contextlib.suppress(OSError, ValueError):
                stream.reconfigure(encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    _utf8_output()
    args = build_parser().parse_args(argv)
    try:
        args.handler(args, _load(args))
    except (OSError, DataValidationError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0
