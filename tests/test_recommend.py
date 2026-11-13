import pandas as pd
import pytest

from market_basket.data import item_support
from market_basket.mining import mine_rules
from market_basket.recommend import Recommender, compare_rankings, recommendations_table


def rules_frame(rows):
    return pd.DataFrame(
        [{"antecedents": frozenset(a), "consequents": frozenset(b), "support": s, "confidence": c, "lift": l}
         for a, b, s, c, l in rows]
    )


RULES = rules_frame(
    [
        ({"Pasta"}, {"Sauce"}, 0.10, 0.50, 2.0),
        ({"Pasta"}, {"Cheese"}, 0.05, 0.30, 3.0),
        ({"Pasta"}, {"Wine"}, 0.12, 0.20, 1.5),
        ({"Pasta", "Sauce"}, {"Basil"}, 0.02, 0.60, 4.0),
        ({"Bread"}, {"Butter"}, 0.10, 0.60, 1.5),
        ({"Pasta"}, {"Sauce", "Cheese"}, 0.04, 0.35, 5.0),  # two-product consequent: never used
    ]
)
BEST_SELLERS = pd.Series({"Milk": 0.4, "Sauce": 0.3, "Eggs": 0.2})


def products(recs):
    return [rec.product for rec in recs]


def test_only_rules_whose_antecedent_is_in_the_basket_apply():
    recs = Recommender(RULES).recommend(["Pasta"])
    assert products(recs) == ["Sauce", "Cheese", "Wine"]  # Basil needs Pasta + Sauce; Butter needs Bread


def test_products_already_in_the_basket_are_never_recommended():
    recs = Recommender(RULES, best_sellers=BEST_SELLERS).recommend(["Pasta", "Sauce", "Milk"], top_n=10)
    names = products(recs)
    assert not {"Pasta", "Sauce", "Milk"} & set(names)
    assert len(names) == len(set(names))  # each product once
    assert names[0] == "Basil" and recs[0].based_on == ("Pasta", "Sauce")


@pytest.mark.parametrize("metric, expected", [("confidence", ["Sauce", "Cheese", "Wine"]),
                                              ("lift", ["Cheese", "Sauce", "Wine"]),
                                              ("support", ["Wine", "Sauce", "Cheese"])])
def test_ranking_metric_changes_the_order(metric, expected):
    assert products(Recommender(RULES, metric).recommend(["Pasta"])) == expected


def test_best_seller_fallback_is_labelled_and_explained():
    recs = Recommender(RULES, best_sellers=BEST_SELLERS).recommend(["Pasta"], top_n=5)
    assert products(recs) == ["Sauce", "Cheese", "Wine", "Milk", "Eggs"]  # Sauce is not repeated
    assert [rec.source for rec in recs] == ["association rule"] * 3 + ["best seller"] * 2
    assert recs[3].confidence is None and "40% of all transactions" in recs[3].reason
    assert "no further association rule applies" in recs[3].reason


def test_empty_and_unknown_baskets():
    with_fallback = Recommender(RULES, best_sellers=BEST_SELLERS)
    empty = with_fallback.recommend([], top_n=2)
    assert products(empty) == ["Milk", "Sauce"] and "basket is empty" in empty[0].reason
    assert products(with_fallback.recommend(["Tea"], top_n=1)) == ["Milk"]
    assert Recommender(RULES).recommend(["Tea"]) == []          # rules only: nothing applies
    assert Recommender(RULES.iloc[0:0]).recommend(["Pasta"]) == []


def test_invalid_arguments():
    with pytest.raises(ValueError):
        Recommender(RULES, "leverage")
    with pytest.raises(ValueError):
        Recommender(RULES).recommend(["Pasta"], top_n=0)


def test_explanations_match_the_supporting_rule(demo):
    rules = mine_rules(demo.transactions).rules
    recs = Recommender(rules, best_sellers=item_support(demo.transactions)).recommend(["Milk", "Bread"], top_n=5)
    assert recs[0].product == "Butter"
    for rec in recs:
        if rec.source != "association rule":
            continue
        match = rules[(rules["antecedents"] == frozenset(rec.based_on)) & (rules["consequents"] == frozenset({rec.product}))]
        assert len(match) == 1
        assert match.iloc[0]["confidence"] == pytest.approx(rec.confidence)
        assert f"{rec.confidence:.0%} of those transactions" in rec.reason
        assert set(rec.based_on) <= {"Milk", "Bread"}


def test_tables_for_display():
    table = recommendations_table(Recommender(RULES, best_sellers=BEST_SELLERS).recommend(["Pasta"], top_n=4))
    assert table["rank"].tolist() == [1, 2, 3, 4]
    assert table.loc[3, "source"] == "best seller" and table.loc[0, "based on"] == "Pasta"
    comparison = compare_rankings(RULES, ["Pasta"], BEST_SELLERS, top_n=4)
    assert list(comparison.columns) == ["ranked by confidence", "ranked by lift", "ranked by support"]
    assert comparison.loc[1, "ranked by lift"] == "Cheese"
    assert comparison.loc[4, "ranked by confidence"] == "Milk (best seller)"


# ---------------------------------------------------------------- release regressions (C2)


def test_excluded_products_are_never_recommended_including_the_best_seller_fallback(demo):
    from market_basket.config import MiningConfig

    config = MiningConfig(exclude=("Milk",))
    rules = mine_rules(demo.transactions, config).rules
    best_sellers = item_support(demo.transactions)
    assert best_sellers.index[0] == "Milk"  # Milk is the top best seller, so it would be the first fallback
    recommender = Recommender(rules, "confidence", best_sellers, config.exclude)
    for basket in (["Pasta"], [], ["Bread"], ["Shampoo", "Conditioner"]):
        assert "Milk" not in products(recommender.recommend(basket, top_n=10))
    for metric in ("confidence", "lift", "support"):
        assert "Milk" not in products(Recommender(rules, metric, best_sellers, ["Milk"]).recommend(["Pasta"], 10))
    comparison = compare_rankings(rules, ["Pasta"], best_sellers, top_n=10, exclude=["Milk"])
    assert not comparison.apply(lambda column: column.str.startswith("Milk")).any().any()


def test_several_exclusions_and_an_empty_candidate_set():
    recommender = Recommender(RULES, best_sellers=BEST_SELLERS, exclude=["Sauce", "Milk"])
    assert products(recommender.recommend(["Pasta"], top_n=10)) == ["Cheese", "Wine", "Eggs"]
    everything = ["Sauce", "Cheese", "Wine", "Basil", "Butter", "Milk", "Eggs"]
    assert Recommender(RULES, best_sellers=BEST_SELLERS, exclude=everything).recommend(["Pasta"], 5) == []
    assert Recommender(RULES, best_sellers=BEST_SELLERS, exclude=everything).recommend([], 5) == []


def test_rules_that_mention_an_excluded_product_are_not_used_even_if_given():
    # a rule whose condition contains an excluded product is skipped too
    recommender = Recommender(RULES, exclude=["Sauce"])
    assert "Basil" not in products(recommender.recommend(["Pasta", "Sauce"], top_n=10))


def test_a_basket_given_as_one_string_is_one_product():
    assert products(Recommender(RULES).recommend("Pasta")) == ["Sauce", "Cheese", "Wine"]
