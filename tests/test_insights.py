import plotly.graph_objects as go
import pytest

from market_basket import plots
from market_basket.config import MiningConfig
from market_basket.data import item_support
from market_basket.evaluation import evaluate
from market_basket.insights import (
    bundle_opportunities,
    category_summary,
    cross_sell_opportunities,
    key_insights,
    product_groups,
    product_popularity,
)
from market_basket.mining import empty_rules, mine_rules

ALL_RULES = MiningConfig(min_support=0.4, min_confidence=0.0, min_lift=0.0, max_len=None)


def test_product_popularity_and_categories(toy_data):
    popularity = product_popularity(toy_data).set_index("product")
    assert popularity.loc["Bread", "transactions"] == 4
    assert popularity.loc["Eggs", "share of transactions"] == pytest.approx(0.2)
    assert popularity.loc["Beer", "category"] == "Drinks"
    categories = category_summary(toy_data).set_index("category")
    assert categories.loc["Drinks", "products"] == 2
    assert categories.loc["Drinks", "transactions"] == 4  # Beer or Cola in T2, T3, T4, T5


def test_cross_sell_lists_each_pair_once_in_its_stronger_direction(toy, toy_data):
    rules = mine_rules(toy, ALL_RULES).rules
    table = cross_sell_opportunities(rules, toy_data.categories, top_n=50)
    pairs = [frozenset({a, b}) for a, b in zip(table["when a basket has"], table["suggest"])]
    assert len(pairs) == len(set(pairs))
    beer = table[table["when a basket has"].eq("Beer") & table["suggest"].eq("Diapers")]
    assert len(beer) == 1 and beer.iloc[0]["confidence"] == pytest.approx(1.0)  # Beer->Diapers beats 0.75
    assert beer.iloc[0]["categories"] == "Drinks + Baby"


def test_bundle_metrics(toy, toy_data):
    result = mine_rules(toy, ALL_RULES)
    table = bundle_opportunities(result, item_support(toy), toy_data.categories, top_n=50).set_index("products")
    beer_diapers = table.loc["Beer + Diapers"]
    assert beer_diapers["transactions together"] == 3
    assert beer_diapers["all-confidence"] == pytest.approx(0.6 / 0.8)       # support / max single support
    assert beer_diapers["lift vs independence"] == pytest.approx(0.6 / (0.6 * 0.8))
    assert beer_diapers["potential use"] == "Cross-category display or bundle offer"
    assert table["all-confidence"].is_monotonic_decreasing


def test_product_groups_are_disjoint_and_handle_no_rules(demo):
    groups = product_groups(mine_rules(demo.transactions).rules, min_lift=2.0)
    flat = [product for group in groups for product in group]
    assert groups and len(flat) == len(set(flat))
    assert ["Cheese", "Pasta", "Pasta Sauce"] in groups  # a planted mission
    assert product_groups(empty_rules()) == []


def test_key_insights_are_computed_from_the_data(demo):
    result = mine_rules(demo.transactions)
    popularity = product_popularity(demo)
    bundles = bundle_opportunities(result, item_support(demo.transactions), demo.categories)
    notes = key_insights(demo, result, popularity, bundles)
    top = popularity.iloc[0]
    assert notes[0] == f"{top['product']} is the most frequently purchased product, in " \
                       f"{top['share of transactions']:.0%} of transactions."
    empty = key_insights(demo, mine_rules(demo.transactions, MiningConfig(min_support=0.9)), popularity, bundles)
    assert "No association rules" in empty[-1]


def test_every_chart_renders_including_empty_inputs(demo):
    result = mine_rules(demo.transactions)
    popularity = product_popularity(demo)
    evaluation = evaluate(demo.transactions, n_splits=1)
    figures = [
        plots.product_bar(popularity),
        plots.product_bar(popularity, least=True),
        plots.category_bar(category_summary(demo)),
        plots.basket_size_chart(demo),
        plots.weekly_transactions_chart(demo),
        plots.rules_scatter(result.rules),
        plots.rule_network(result.rules, demo.categories),
        plots.rule_network(empty_rules()),
        plots.evaluation_chart(evaluation.summary),
    ]
    assert all(isinstance(figure, go.Figure) for figure in figures)
    colors = plots.category_colors(["b", "a", "c"])
    assert colors == {"a": plots.CATEGORICAL[0], "b": plots.CATEGORICAL[1], "c": plots.CATEGORICAL[2]}
