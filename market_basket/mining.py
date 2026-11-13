"""Frequent itemsets and association rules, plus filtering, ranking and plain-English explanations.

The mining itself is done by the mlxtend library (TransactionEncoder, apriori, fpgrowth and
association_rules). This module chooses the algorithm, applies the user's thresholds, removes
excluded products, ranks the rules and explains them.
"""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from .config import MAX_FREQUENT_ITEMSETS, MAX_RULE_CANDIDATES, RULE_RANK_METRICS, MiningConfig
from .data import encode

# Metrics are rounded to this many decimals before sorting, so that values which are
# mathematically equal but differ in the last floating-point digits (for example the lift of
# A -> B and of B -> A) count as ties. Real differences are far larger than 1e-10.
RANK_DECIMALS = 10

RULE_COLUMNS = [
    "antecedents",
    "consequents",
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
]


@dataclass(frozen=True)
class MiningResult:
    frequent_itemsets: pd.DataFrame
    rules: pd.DataFrame
    n_transactions: int
    config: MiningConfig


def _counts(basket: pd.DataFrame) -> tuple[np.ndarray, object]:
    """Per-product transaction counts and a matrix that supports column selection and row sums."""
    if len(basket.columns) and all(isinstance(dtype, pd.SparseDtype) for dtype in basket.dtypes):
        matrix = basket.sparse.to_coo().tocsc()
        return np.asarray(matrix.sum(axis=0)).ravel(), matrix
    matrix = basket.to_numpy(dtype=bool)
    return matrix.sum(axis=0), matrix


def estimate_itemsets(basket: pd.DataFrame, min_support: float, max_len: int | None,
                      stop_above: int | None = None) -> int:
    """Guaranteed upper bound on the number of frequent itemsets, computed cheaply BEFORE mining.

    Only products that reach the minimum support on their own can be in a frequent itemset
    (the Apriori property). A frequent itemset of size k occurs in at least m transactions
    (m = the minimum support as a count), and a transaction with s frequent products contains
    C(s, k) itemsets of size k. So there are at most min(C(n, k), sum of C(s, k) / m) frequent
    k-itemsets, where n is the number of frequent products. Stops early once stop_above is passed.
    """
    n_transactions = len(basket)
    if n_transactions == 0:
        return 0
    counts, matrix = _counts(basket)
    frequent = counts / n_transactions >= min_support * (1 - 1e-9)  # never miss a borderline product
    n_frequent = int(frequent.sum())
    if n_frequent == 0:
        return 0
    sizes = np.asarray(matrix[:, frequent].sum(axis=1)).ravel().astype(np.int64)
    size_counts = np.bincount(sizes)
    min_count = max(1, math.ceil(min_support * n_transactions * (1 - 1e-12)))
    largest = min(len(size_counts) - 1, max_len or n_frequent)
    bound = 0
    for k in range(1, largest + 1):
        occurrences = sum(int(count) * math.comb(size, k)
                          for size, count in enumerate(size_counts) if count and size >= k)
        bound += min(math.comb(n_frequent, k), occurrences // min_count)
        if stop_above is not None and bound > stop_above:
            break
    return bound


def find_frequent_itemsets(basket: pd.DataFrame, config: MiningConfig) -> pd.DataFrame:
    """Itemsets whose support reaches config.min_support, with a 'length' column.

    Raises ValueError, before any mining starts, when the settings could produce more than
    MAX_FREQUENT_ITEMSETS itemsets, so that very low support or "no limit" on dense data cannot
    exhaust memory.
    """
    bound = estimate_itemsets(basket, config.min_support, config.max_len, stop_above=MAX_FREQUENT_ITEMSETS)
    if bound > MAX_FREQUENT_ITEMSETS:
        size = f"at most {config.max_len} products" if config.max_len else "no size limit"
        raise ValueError(
            f"Minimum support {config.min_support:.2%} with {size} could produce more than "
            f"{MAX_FREQUENT_ITEMSETS:,} frequent itemsets (an upper bound checked before mining), which is too "
            "many to analyse safely. Raise the minimum support or lower the maximum number of products in a rule."
        )
    miner = fpgrowth if config.algorithm == "fpgrowth" else apriori
    itemsets = miner(basket, min_support=config.min_support, use_colnames=True, max_len=config.max_len)
    if len(itemsets) > MAX_FREQUENT_ITEMSETS:  # cannot happen if the bound is correct; kept as a safety net
        raise ValueError(
            f"Minimum support {config.min_support:.2%} produced {len(itemsets):,} frequent itemsets, more than "
            f"the limit of {MAX_FREQUENT_ITEMSETS:,}. Raise the minimum support or lower the maximum number of "
            "products in a rule."
        )
    itemsets["length"] = itemsets["itemsets"].map(len)
    return itemsets.sort_values(["length", "support"], ascending=[True, False], ignore_index=True)


def empty_rules() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object" if column in ("antecedents", "consequents") else float)
                         for column in RULE_COLUMNS})


def rank_rules(rules: pd.DataFrame, by: str = "lift") -> pd.DataFrame:
    """Sort rules by one metric, breaking ties by confidence, support and then product names.

    Metrics are compared after rounding to RANK_DECIMALS, so mathematically equal values are ties.
    """
    if by not in RULE_RANK_METRICS:
        raise ValueError(f"Rank metric must be one of {RULE_RANK_METRICS}, got {by!r}.")
    if rules.empty:
        return rules
    keys = list(dict.fromkeys([by, "confidence", "support"]))
    sort_columns = {f"_key_{key}": rules[key].astype(float).round(RANK_DECIMALS) for key in keys}
    sort_columns["_names"] = rules["antecedents"].map(as_text) + " -> " + rules["consequents"].map(as_text)
    ordered = rules.assign(**sort_columns).sort_values(
        list(sort_columns), ascending=[False] * len(keys) + [True], kind="stable"
    )
    return ordered.drop(columns=list(sort_columns)).reset_index(drop=True)


def generate_rules(
    frequent_itemsets: pd.DataFrame, n_transactions: int, config: MiningConfig, rank_by: str = "lift"
) -> pd.DataFrame:
    """Split frequent itemsets into rules A -> B and keep those passing the lift and confidence thresholds.

    Raises ValueError before rule generation if mlxtend would have to evaluate more than
    MAX_RULE_CANDIDATES candidate rules (an itemset of k products gives 2^k - 2 candidates).
    """
    if frequent_itemsets.empty or frequent_itemsets["length"].max() < 2:
        return empty_rules()
    lengths = frequent_itemsets["length"]
    candidates = int((2.0 ** lengths[lengths >= 2] - 2).sum())
    if candidates > MAX_RULE_CANDIDATES:
        raise ValueError(
            f"These settings give {len(frequent_itemsets):,} frequent itemsets and {candidates:,} candidate rules, "
            f"more than the limit of {MAX_RULE_CANDIDATES:,}. Raise the minimum support or lower the maximum "
            "number of products in a rule."
        )
    rules = association_rules(
        frequent_itemsets, num_itemsets=n_transactions, metric="lift", min_threshold=config.min_lift
    )
    rules = rules[rules["confidence"] >= config.min_confidence]
    if config.exclude:
        excluded = frozenset(config.exclude)
        keep = rules["antecedents"].map(excluded.isdisjoint) & rules["consequents"].map(excluded.isdisjoint)
        rules = rules[keep]
    if rules.empty:
        return empty_rules()
    return rank_rules(rules[RULE_COLUMNS], rank_by)


def mine_rules(
    transactions: list[list[str]], config: MiningConfig | None = None, sparse: bool | str = "auto"
) -> MiningResult:
    """Encode the transactions, mine frequent itemsets and generate ranked rules in one call."""
    config = config or MiningConfig()
    if not transactions:
        raise ValueError("No transactions to analyse.")
    basket = encode(transactions, sparse=sparse)
    itemsets = find_frequent_itemsets(basket, config)
    rules = generate_rules(itemsets, len(transactions), config)
    return MiningResult(itemsets, rules, len(transactions), config)


def product_label(name: str) -> str:
    """A product name as shown in lists: quoted if it contains a comma, so lists stay unambiguous."""
    return f'"{name}"' if "," in name else name


def as_text(itemset: Iterable[str]) -> str:
    """Comma-separated, sorted product names ('Bread, Milk'); names containing a comma are quoted."""
    return ", ".join(product_label(name) for name in sorted(itemset))


def join_names(names: Iterable[str]) -> str:
    """Sorted product names in prose: 'Bread', 'Bread and Milk', 'Bread, Eggs and Milk'."""
    labels = [product_label(name) for name in sorted(names)]
    return labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1]


def explain_rule(antecedents, consequents, support: float, confidence: float, lift: float,
                 consequent_support: float | None = None) -> str:
    """Plain-English reading of one rule, worded as an observation rather than a promise."""
    a, b = join_names(antecedents), join_names(consequents)
    text = (f"In the observed transactions, {confidence:.0%} of baskets with {a} also contained {b}. "
            f"That is {lift:.2f} times the overall rate of {b}")
    text += f" ({consequent_support:.0%})." if consequent_support is not None else "."
    text += f" Both appear together in {support:.1%} of all transactions."
    return text


def readable(table: pd.DataFrame) -> pd.DataFrame:
    """Copy of a rules or itemsets table with frozensets shown as comma-separated text."""
    table = table.copy()
    for column in ("antecedents", "consequents", "itemsets"):
        if column in table:
            table[column] = table[column].map(as_text)
    return table
