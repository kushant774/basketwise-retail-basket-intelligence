"""Project defaults and the validated settings used to mine association rules."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEMO_DATA_PATH = PROJECT_ROOT / "Data" / "synthetic_retail_transactions.csv"

ALGORITHMS = ("fpgrowth", "apriori")
RULE_RANK_METRICS = ("lift", "confidence", "support", "leverage", "conviction")
RECOMMENDATION_RANK_METRICS = ("confidence", "lift", "support")

MAX_UPLOAD_MB = 50
MAX_UPLOAD_ROWS = 2_000_000
MAX_FREQUENT_ITEMSETS = 200_000


@dataclass(frozen=True)
class MiningConfig:
    """Thresholds for frequent itemsets and association rules.

    min_support:    an itemset must appear in at least this share of transactions.
    min_confidence: a rule A -> B must hold in at least this share of the transactions containing A.
    min_lift:       a rule must make B at least this many times more likely than its overall rate.
    max_len:        largest itemset size to mine (None means no limit).
    algorithm:      "fpgrowth" (default) or "apriori"; both return the same itemsets.
    exclude:        products whose rules are removed after mining (other supports are unchanged).
    """

    min_support: float = 0.01
    min_confidence: float = 0.2
    min_lift: float = 1.2
    max_len: int | None = 3
    algorithm: str = "fpgrowth"
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not 0 < self.min_support <= 1:
            raise ValueError(f"Minimum support must be above 0 and at most 1, got {self.min_support}.")
        if not 0 <= self.min_confidence <= 1:
            raise ValueError(f"Minimum confidence must be between 0 and 1, got {self.min_confidence}.")
        if self.min_lift < 0:
            raise ValueError(f"Minimum lift cannot be negative, got {self.min_lift}.")
        if self.max_len is not None and self.max_len < 2:
            raise ValueError(f"Maximum itemset size must be at least 2 to form a rule, got {self.max_len}.")
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Algorithm must be one of {ALGORITHMS}, got {self.algorithm!r}.")
        object.__setattr__(self, "exclude", tuple(self.exclude))
