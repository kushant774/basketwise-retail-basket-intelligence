"""Frequent itemsets and association rules, plus filtering, ranking and plain-English explanations.

The mining itself is done by the mlxtend library (TransactionEncoder, apriori, fpgrowth and
association_rules). This module chooses the algorithm, applies the user's thresholds, removes
excluded products, ranks the rules and explains them.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

from .config import MAX_FREQUENT_ITEMSETS, RULE_RANK_METRICS, MiningConfig
from .data import encode

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


def find_frequent_itemsets(basket: pd.DataFrame, config: MiningConfig) -> pd.DataFrame:
    """Itemsets whose support reaches config.min_support, with a 'length' column."""
    miner = fpgrowth if config.algorithm == "fpgrowth" else apriori
    itemsets = miner(basket, min_support=config.min_support, use_colnames=True, max_len=config.max_len)
    if len(itemsets) > MAX_FREQUENT_ITEMSETS:
        raise ValueError(
            f"Minimum support {config.min_support} produces {len(itemsets):,} frequent itemsets, too many to "
            "analyse. Raise the minimum support or lower the maximum itemset size."
        )
    itemsets["length"] = itemsets["itemsets"].map(len)
    return itemsets.sort_values(["length", "support"], ascending=[True, False], ignore_index=True)


def empty_rules() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object" if column in ("antecedents", "consequents") else float)
                         for column in RULE_COLUMNS})


def rank_rules(rules: pd.DataFrame, by: str = "lift") -> pd.DataFrame:
    """Sort rules by one metric, breaking ties by confidence, support and then product names."""
    if by not in RULE_RANK_METRICS:
        raise ValueError(f"Rank metric must be one of {RULE_RANK_METRICS}, got {by!r}.")
    if rules.empty:
        return rules
    keys = list(dict.fromkeys([by, "confidence", "support"]))
    names = rules["antecedents"].map(as_text) + " -> " + rules["consequents"].map(as_text)
    ordered = rules.assign(_names=names).sort_values(
        keys + ["_names"], ascending=[False] * len(keys) + [True], kind="stable"
    )
    return ordered.drop(columns="_names").reset_index(drop=True)


def generate_rules(
    frequent_itemsets: pd.DataFrame, n_transactions: int, config: MiningConfig, rank_by: str = "lift"
) -> pd.DataFrame:
    """Split frequent itemsets into rules A -> B and keep those passing the lift and confidence thresholds."""
    if frequent_itemsets.empty or frequent_itemsets["length"].max() < 2:
        return empty_rules()
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


def as_text(itemset) -> str:
    return ", ".join(sorted(itemset))


def _join(names) -> str:
    names = sorted(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def explain_rule(antecedents, consequents, support: float, confidence: float, lift: float,
                 consequent_support: float | None = None) -> str:
    """Plain-English reading of one rule, worded as an observation rather than a promise."""
    a, b = _join(antecedents), _join(consequents)
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
