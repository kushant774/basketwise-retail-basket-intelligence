"""Explainable product recommendations built from association rules.

For a basket, a rule A -> B applies when every product in A is already in the basket and the
single product B is not. Applicable rules are taken in ranked order (confidence by default),
each product is recommended once through its best rule, and every recommendation carries a
reason. If fewer than top_n products are found, best sellers fill the remaining slots and are
labelled as such.
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import pandas as pd

from .config import RECOMMENDATION_RANK_METRICS
from .mining import as_text


@dataclass(frozen=True)
class Recommendation:
    product: str
    source: str  # "association rule" or "best seller"
    reason: str
    based_on: tuple[str, ...] = ()
    support: float | None = None
    confidence: float | None = None
    lift: float | None = None


def _join(names: Iterable[str]) -> str:
    names = sorted(names)
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


class Recommender:
    def __init__(self, rules: pd.DataFrame, rank_by: str = "confidence", best_sellers: pd.Series | None = None):
        """rules: output of mining.generate_rules. best_sellers: product -> share of transactions
        (for example data.item_support), used as the labelled fallback. Pass None for rules only."""
        if rank_by not in RECOMMENDATION_RANK_METRICS:
            raise ValueError(f"Ranking must be one of {RECOMMENDATION_RANK_METRICS}, got {rank_by!r}.")
        self.rank_by = rank_by
        single = rules[rules["consequents"].map(len) == 1] if len(rules) else rules
        entries = [
            (row.antecedents, next(iter(row.consequents)), row.support, row.confidence, row.lift)
            for row in single.itertuples(index=False)
        ]
        position = {"support": 2, "confidence": 3, "lift": 4}[rank_by]
        entries.sort(key=lambda e: (-e[position], -e[3], -e[4], -e[2], as_text(e[0]), e[1]))
        self._rules = entries
        self.best_sellers = best_sellers if best_sellers is not None else pd.Series(dtype=float)

    def recommend(self, basket: Iterable[str], top_n: int = 5) -> list[Recommendation]:
        if top_n < 1:
            raise ValueError(f"Number of recommendations must be at least 1, got {top_n}.")
        items = frozenset(basket)
        chosen: list[Recommendation] = []
        taken = set(items)
        for antecedent, product, support, confidence, lift in self._rules:
            if product in taken or not antecedent <= items:
                continue
            reason = (f"Customers whose basket contained {_join(antecedent)} also bought {product} in "
                      f"{confidence:.0%} of those transactions ({lift:.1f}x its usual rate) in the observed data.")
            chosen.append(Recommendation(product, "association rule", reason, tuple(sorted(antecedent)),
                                         support, confidence, lift))
            taken.add(product)
            if len(chosen) == top_n:
                return chosen
        why = "the basket is empty" if not items else "no further association rule applies to this basket"
        for product, share in self.best_sellers.items():
            if product in taken:
                continue
            chosen.append(Recommendation(product, "best seller",
                                         f"Best seller, bought in {share:.0%} of all transactions; shown because {why}."))
            taken.add(product)
            if len(chosen) == top_n:
                break
        return chosen


def recommendations_table(recommendations: list[Recommendation]) -> pd.DataFrame:
    columns = ["rank", "recommended product", "source", "based on", "support", "confidence", "lift", "reason"]
    return pd.DataFrame(
        [
            {
                "rank": position,
                "recommended product": rec.product,
                "source": rec.source,
                "based on": ", ".join(rec.based_on),
                "support": rec.support,
                "confidence": rec.confidence,
                "lift": rec.lift,
                "reason": rec.reason,
            }
            for position, rec in enumerate(recommendations, start=1)
        ],
        columns=columns,
    )


def compare_rankings(rules: pd.DataFrame, basket: Iterable[str], best_sellers: pd.Series,
                     top_n: int = 5) -> pd.DataFrame:
    """Side-by-side top-n products for each ranking method (rules ranked by confidence, lift, support)."""
    basket = list(basket)
    columns = {}
    for metric in RECOMMENDATION_RANK_METRICS:
        recs = Recommender(rules, metric, best_sellers).recommend(basket, top_n)
        names = [rec.product + (" (best seller)" if rec.source == "best seller" else "") for rec in recs]
        columns[f"ranked by {metric}"] = names + [""] * (top_n - len(names))
    return pd.DataFrame(columns, index=pd.RangeIndex(1, top_n + 1, name="rank"))
