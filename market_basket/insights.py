"""Business-facing summaries calculated from the cleaned data and the mined rules.

Every number comes from the data; wording describes observed patterns and potential
opportunities, never guaranteed sales impact.
"""
from __future__ import annotations

import math

import networkx as nx
import pandas as pd

from .data import TransactionData
from .mining import MiningResult, as_text


def single_product_rules(rules: pd.DataFrame) -> pd.DataFrame:
    """Rules with exactly one product on each side."""
    if rules.empty:
        return rules
    return rules[(rules["antecedents"].map(len) == 1) & (rules["consequents"].map(len) == 1)]


def product_popularity(data: TransactionData) -> pd.DataFrame:
    """Every product with the number and share of transactions that contain it, most popular first."""
    items = data.items
    table = items.groupby("product_name").agg(transactions=("transaction_id", "nunique"))
    table["share of transactions"] = table["transactions"] / data.report.valid_transactions
    if "category" in items:
        table["category"] = items.drop_duplicates("product_name").set_index("product_name")["category"]
    if "quantity" in items and items["quantity"].notna().any():
        table["units"] = items.groupby("product_name")["quantity"].sum(min_count=1)
    table = table.rename_axis("product").reset_index()
    return table.sort_values(["transactions", "product"], ascending=[False, True], ignore_index=True)


def category_summary(data: TransactionData) -> pd.DataFrame:
    """Products, transactions and share of transactions for each category (empty if no categories)."""
    items = data.items
    if "category" not in items:
        return pd.DataFrame(columns=["category", "products", "transactions", "share of transactions"])
    table = items.groupby("category").agg(products=("product_name", "nunique"),
                                         transactions=("transaction_id", "nunique"))
    table["share of transactions"] = table["transactions"] / data.report.valid_transactions
    table = table.rename_axis("category").reset_index()
    return table.sort_values(["transactions", "category"], ascending=[False, True], ignore_index=True)


def cross_sell_opportunities(rules: pd.DataFrame, categories: dict[str, str] | None = None,
                             top_n: int = 10) -> pd.DataFrame:
    """Strongest product pairs for "customers who buy X also buy Y" prompts, one row per pair.

    A pair appears as A -> B and B -> A with the same lift; the direction with the higher
    confidence is kept because it is the more reliable suggestion.
    """
    columns = ["when a basket has", "suggest", "support", "confidence", "lift", "categories"]
    pairs = single_product_rules(rules)
    if pairs.empty:
        return pd.DataFrame(columns=columns)
    pairs = pairs.assign(
        a=pairs["antecedents"].map(as_text), b=pairs["consequents"].map(as_text),
        pair=[as_text(x | y) for x, y in zip(pairs["antecedents"], pairs["consequents"])],
    )
    best = pairs.sort_values(["confidence", "a"], ascending=[False, True], kind="stable").drop_duplicates("pair")
    best = best.sort_values(["lift", "confidence", "pair"], ascending=[False, False, True], kind="stable").head(top_n)
    categories = categories or {}
    labels = [_category_label([a, b], categories) for a, b in zip(best["a"], best["b"])]
    return pd.DataFrame({"when a basket has": best["a"].to_numpy(), "suggest": best["b"].to_numpy(),
                         "support": best["support"].to_numpy(), "confidence": best["confidence"].to_numpy(),
                         "lift": best["lift"].to_numpy(), "categories": labels}, columns=columns)


def _category_label(products: list[str], categories: dict[str, str]) -> str:
    names = list(dict.fromkeys(categories.get(product, "Uncategorised") for product in products))
    return names[0] if len(names) == 1 else " + ".join(names)


def bundle_opportunities(result: MiningResult, item_support: pd.Series, categories: dict[str, str] | None = None,
                         top_n: int = 10) -> pd.DataFrame:
    """Product sets (2 or 3 products) that are bought together far more often than chance.

    all-confidence = support(set) / largest single-product support in the set, i.e. at least
    this share of the buyers of ANY product in the set bought the whole set. Unlike lift, it is
    not inflated by set size, so pairs and triples can be compared. Sets are ranked by it.
    """
    columns = ["products", "size", "transactions together", "support", "all-confidence",
               "lift vs independence", "categories", "potential use"]
    sets = result.frequent_itemsets[result.frequent_itemsets["length"].between(2, 3)]
    if sets.empty:
        return pd.DataFrame(columns=columns)
    categories = categories or {}
    rows = []
    for itemset, support, size in zip(sets["itemsets"], sets["support"], sets["length"]):
        products = sorted(itemset)
        singles = [float(item_support[p]) for p in products]
        label = _category_label(products, categories)
        use = (f"Shelve together within {label}" if " + " not in label
               else "Cross-category display or bundle offer")
        rows.append({
            "products": " + ".join(products),
            "size": int(size),
            "transactions together": round(support * result.n_transactions),
            "support": support,
            "all-confidence": support / max(singles),
            "lift vs independence": support / math.prod(singles),
            "categories": label,
            "potential use": use,
        })
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values(["all-confidence", "support", "products"], ascending=[False, False, True],
                             ignore_index=True).head(top_n)


def product_groups(rules: pd.DataFrame, min_lift: float = 2.0) -> list[list[str]]:
    """Clusters of strongly associated products (candidates for merchandising groups).

    Builds a graph with an edge for each single-product rule whose lift reaches min_lift
    (weight = lift) and splits it into communities by greedy modularity maximisation.
    """
    pairs = single_product_rules(rules)
    pairs = pairs[pairs["lift"] >= min_lift] if len(pairs) else pairs
    graph = nx.Graph()
    edges = sorted((as_text(a), as_text(b), float(lift))
                   for a, b, lift in zip(pairs["antecedents"], pairs["consequents"], pairs["lift"]))
    for a, b, lift in edges:
        graph.add_edge(a, b, weight=lift)
    if graph.number_of_edges() == 0:
        return []
    communities = nx.community.greedy_modularity_communities(graph, weight="weight")
    groups = [sorted(group) for group in communities if len(group) > 1]
    return sorted(groups, key=lambda group: (-len(group), group))


def key_insights(data: TransactionData, result: MiningResult, popularity: pd.DataFrame,
                 bundles: pd.DataFrame) -> list[str]:
    """Short, calculated observations for the overview page."""
    notes = []
    top = popularity.iloc[0]
    notes.append(f"{top['product']} is the most frequently purchased product, in "
                 f"{top['share of transactions']:.0%} of transactions.")
    single = (data.items.groupby("transaction_id").size() == 1).mean()
    notes.append(f"{single:.0%} of transactions contain a single product; the rest can reveal combinations.")
    rules = result.rules
    if rules.empty:
        notes.append("No association rules pass the current thresholds. Try a lower minimum support.")
        return notes
    best = rules.iloc[0]
    notes.append(f"Strongest rule by lift: {as_text(best['antecedents'])} -> {as_text(best['consequents'])} "
                 f"(confidence {best['confidence']:.0%}, lift {best['lift']:.1f}).")
    pairs = cross_sell_opportunities(rules, data.categories, top_n=1)
    if len(pairs):
        p = pairs.iloc[0]
        notes.append(f"Top cross-sell candidate: suggest {p['suggest']} when a basket has {p['when a basket has']} "
                     f"({p['confidence']:.0%} of such baskets in the observed data).")
    if len(bundles):
        b = bundles.iloc[0]
        notes.append(f"Most cohesive bundle: {b['products']}, bought together in {b['transactions together']:,} "
                     "transactions.")
    return notes
