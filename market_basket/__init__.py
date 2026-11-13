"""BasketWise: retail basket analysis, association rules and explainable recommendations.

Modules
-------
synthetic_data  reproducible synthetic transactions for demos and tests
data            CSV reading, validation, cleaning, summaries and one-hot encoding
mining          frequent itemsets and association rules (mlxtend), ranking, explanations
recommend       explainable basket recommendations with a best-seller fallback
evaluation      offline leave-one-out evaluation against a best-seller baseline
insights        product, category, cross-selling, bundle and merchandising summaries
plots           interactive Plotly charts
reporting       Markdown report and ZIP of CSV exports
"""
from .config import DEMO_DATA_PATH, MiningConfig
from .data import DataValidationError, TransactionData, item_support, load_demo_data, load_transactions, summarize
from .evaluation import evaluate
from .mining import MiningResult, mine_rules, rank_rules, readable
from .recommend import Recommender, recommendations_table

__all__ = [
    "DEMO_DATA_PATH",
    "DataValidationError",
    "MiningConfig",
    "MiningResult",
    "Recommender",
    "TransactionData",
    "evaluate",
    "item_support",
    "load_demo_data",
    "load_transactions",
    "mine_rules",
    "rank_rules",
    "readable",
    "recommendations_table",
    "summarize",
]
