"""Time encoding and frequent-itemset mining as the number of transactions grows.

Larger datasets are produced by the synthetic generator (same 47-product catalogue), so this
measures scaling in the number of transactions, not in the number of products. The default
settings are the app's defaults (minimum support 1%, at most 3 products per itemset); timings
depend on the computer, so compare algorithms within one run rather than across machines.
Run from the project root:
    python scripts/benchmark.py
    python scripts/benchmark.py --min-support 0.002 --max-len 0     # low support, no size limit
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from market_basket.config import ALGORITHMS, MiningConfig  # noqa: E402
from market_basket.data import encode  # noqa: E402
from market_basket.mining import find_frequent_itemsets  # noqa: E402
from market_basket.synthetic_data import generate_transactions  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    defaults = MiningConfig()
    parser.add_argument("--sizes", type=int, nargs="+", default=[8_000, 40_000, 120_000])
    parser.add_argument("--min-support", type=float, default=defaults.min_support)
    parser.add_argument("--max-len", type=int, default=defaults.max_len,
                        help=f"largest itemset size; 0 means no limit (default {defaults.max_len})")
    args = parser.parse_args()
    max_len = None if args.max_len == 0 else args.max_len
    print(f"Settings: minimum support {args.min_support:.2%}, "
          f"{'no size limit' if max_len is None else f'at most {max_len} products per itemset'}")

    rows = []
    for size in args.sizes:
        start = time.perf_counter()
        frame = generate_transactions(n_transactions=size, seed=42)
        baskets = frame.groupby("transaction_id", sort=False)["product_name"].agg(list).tolist()
        generate_seconds = time.perf_counter() - start
        for sparse in (False, True):
            start = time.perf_counter()
            matrix = encode(baskets, sparse=sparse)
            encode_seconds = time.perf_counter() - start
            megabytes = matrix.memory_usage(deep=True).sum() / 1e6
            for algorithm in ALGORITHMS:
                start = time.perf_counter()
                itemsets = find_frequent_itemsets(matrix, MiningConfig(args.min_support, algorithm=algorithm,
                                                                       max_len=max_len))
                rows.append({
                    "transactions": size,
                    "generate s": round(generate_seconds, 2),
                    "matrix": "sparse" if sparse else "dense",
                    "matrix MB": round(megabytes, 2),
                    "encode s": round(encode_seconds, 2),
                    "algorithm": algorithm,
                    "mine s": round(time.perf_counter() - start, 3),
                    "itemsets": len(itemsets),
                })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
