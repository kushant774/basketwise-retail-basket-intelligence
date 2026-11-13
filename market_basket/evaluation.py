"""Offline evaluation of recommendations against a best-seller baseline.

Protocol
--------
1. Shuffle the transactions with a fixed seed and split them into train and test sets.
2. Mine rules and compute best sellers from the TRAIN set only (no test information is used).
3. For every test transaction with at least two products, hide each product in turn. The other
   products form the query basket, and the hidden product is the target.
4. A method scores a hit at k if the hidden product is among its top k recommendations.
5. Repeat for several random splits and report the mean and standard deviation.

Metrics
-------
hit rate@k:   hits / evaluated cases
availability: share of cases where the method returned at least one recommendation
coverage:     share of training products that the method recommended at least once
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import pandas as pd

from .config import MiningConfig
from .data import item_support
from .mining import mine_rules
from .recommend import Recommender

METHODS = (
    "Best-seller baseline",
    "Rules ranked by confidence",
    "Rules ranked by lift",
    "Rules by confidence + best-seller fallback",
)


@dataclass(frozen=True)
class EvaluationResult:
    summary: pd.DataFrame     # one row per method: mean and std over splits
    per_split: pd.DataFrame   # one row per method and split
    settings: dict


def train_test_split(transactions: list[list[str]], test_size: float = 0.2,
                     seed: int = 42) -> tuple[list[list[str]], list[list[str]]]:
    if not 0 < test_size < 1:
        raise ValueError(f"Test size must be between 0 and 1, got {test_size}.")
    order = list(range(len(transactions)))
    random.Random(seed).shuffle(order)
    cut = round(len(order) * (1 - test_size))
    return [transactions[i] for i in order[:cut]], [transactions[i] for i in order[cut:]]


def leave_one_out_cases(test: list[list[str]]):
    """Yield (query basket, hidden product) for every product of every test basket with 2+ products."""
    for basket in test:
        if len(basket) < 2:
            continue
        for hidden in basket:
            yield [item for item in basket if item != hidden], hidden


def build_methods(train: list[list[str]], config: MiningConfig) -> dict[str, Recommender]:
    """Recommenders built from training transactions only."""
    rules = mine_rules(train, config).rules
    best_sellers = item_support(train)
    return {
        "Best-seller baseline": Recommender(rules.iloc[0:0], "confidence", best_sellers),
        "Rules ranked by confidence": Recommender(rules, "confidence"),
        "Rules ranked by lift": Recommender(rules, "lift"),
        "Rules by confidence + best-seller fallback": Recommender(rules, "confidence", best_sellers),
    }


def score(recommender: Recommender, test: list[list[str]], ks: tuple[int, ...] = (3, 5),
          catalog_size: int | None = None) -> dict[str, float]:
    ks = tuple(sorted(ks))
    cases = available = 0
    hits = dict.fromkeys(ks, 0)
    recommended: dict[str, None] = {}
    for query, hidden in leave_one_out_cases(test):
        products = [rec.product for rec in recommender.recommend(query, top_n=ks[-1])]
        cases += 1
        available += bool(products)
        recommended.update(dict.fromkeys(products))
        for k in ks:
            hits[k] += hidden in products[:k]
    if cases == 0:
        raise ValueError("The test set has no transaction with two or more products to evaluate.")
    result = {"cases": cases}
    result.update({f"hit rate@{k}": hits[k] / cases for k in ks})
    result["availability"] = available / cases
    result["coverage"] = len(recommended) / catalog_size if catalog_size else float("nan")
    return result


def evaluate(transactions: list[list[str]], config: MiningConfig | None = None, n_splits: int = 5,
             test_size: float = 0.2, seed: int = 42, ks: tuple[int, ...] = (3, 5)) -> EvaluationResult:
    """Score every method on n_splits random train/test splits (seeds seed, seed+1, ...)."""
    if n_splits < 1:
        raise ValueError(f"Number of splits must be at least 1, got {n_splits}.")
    config = config or MiningConfig()
    rows = []
    for split in range(n_splits):
        train, test = train_test_split(transactions, test_size, seed + split)
        catalog_size = len({item for basket in train for item in basket})
        for method, recommender in build_methods(train, config).items():
            rows.append({"split": split + 1, "method": method,
                         **score(recommender, test, ks, catalog_size)})
    per_split = pd.DataFrame(rows)
    metrics = [column for column in per_split.columns if column not in ("split", "method")]
    by_method = per_split.groupby("method", sort=False)
    summary = by_method[metrics].mean()
    summary["cases"] = summary["cases"].round().astype(int)
    for k in ks:
        summary[f"hit rate@{k} std"] = by_method[f"hit rate@{k}"].std(ddof=1) if n_splits > 1 else 0.0
    baseline = per_split[per_split["method"] == METHODS[0]].set_index("split")
    for k in ks:
        wins = per_split.pivot(index="split", columns="method", values=f"hit rate@{k}").gt(
            baseline[f"hit rate@{k}"], axis=0).sum()
        summary[f"splits beating baseline@{k}"] = wins.astype("Int64")
        summary.loc[METHODS[0], f"splits beating baseline@{k}"] = pd.NA
    settings = {"n_transactions": len(transactions), "n_splits": n_splits, "test_size": test_size,
                "seed": seed, "ks": ks, "config": config}
    return EvaluationResult(summary.reset_index(), per_split, settings)
