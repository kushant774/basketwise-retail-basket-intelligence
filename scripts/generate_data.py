"""Regenerate the synthetic demo dataset.

Run from the project root:
    python scripts/generate_data.py
    python scripts/generate_data.py --transactions 20000 --products 30 --seed 7 --output Data/custom.csv
    python scripts/generate_data.py --transactions 3000 --seed 7 --with-issues --output Data/demo_upload_with_issues.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from market_basket.config import DEMO_DATA_PATH  # noqa: E402
from market_basket.synthetic_data import CATALOG, add_data_quality_issues, generate_transactions  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the synthetic BasketWise demo dataset.")
    parser.add_argument("--transactions", type=int, default=8000, help="number of transactions (default 8000)")
    parser.add_argument("--products", type=int, default=None,
                        help=f"number of products, 8 to {len(CATALOG)} (default: all {len(CATALOG)})")
    parser.add_argument("--seed", type=int, default=42, help="random seed, 0 or more (default 42)")
    parser.add_argument("--output", type=Path, default=DEMO_DATA_PATH, help="CSV file to write")
    parser.add_argument("--with-issues", action="store_true",
                        help="add known data-quality problems (for demonstrating validation)")
    args = parser.parse_args()
    if args.products is not None and not 8 <= args.products <= len(CATALOG):
        parser.error(f"--products must be between 8 and {len(CATALOG)}, got {args.products}")
    if args.transactions < 1:
        parser.error(f"--transactions must be at least 1, got {args.transactions}")
    if args.seed < 0:
        parser.error(f"--seed must be 0 or more, got {args.seed}")
    if args.output.suffix.lower() != ".csv":
        parser.error(f"--output must be a .csv file, got {args.output}")

    try:
        data = generate_transactions(n_transactions=args.transactions, n_products=args.products, seed=args.seed)
        issues = {}
        if args.with_issues:
            data, issues = add_data_quality_issues(data, seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output, index=False, lineterminator="\n")
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    basket_sizes = data.groupby("transaction_id").size()
    print(f"Wrote {args.output} (synthetic data, seed {args.seed})")
    print(f"  rows (transaction lines): {len(data):,}")
    print(f"  transactions:             {data['transaction_id'].nunique():,}")
    print(f"  products:                 {data['product_name'].nunique()}")
    print(f"  categories:               {data['category'].nunique()}")
    print(f"  synthetic customers:      {data['customer_id'].nunique():,}")
    print(f"  average basket size:      {basket_sizes.mean():.2f}")
    print(f"  dates:                    {data['transaction_date'].min()} to {data['transaction_date'].max()}")
    if issues:
        print("  (the counts above are before validation; the app and `python -m market_basket validate`")
        print("   should report exactly the injected problems below)")
    for problem, count in issues.items():
        print(f"  injected {problem + ':':<26}{count:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
