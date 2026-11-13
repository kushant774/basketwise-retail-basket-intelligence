import math

import pytest

from market_basket import mining
from market_basket.config import MiningConfig
from market_basket.data import encode
from market_basket.mining import (
    explain_rule,
    find_frequent_itemsets,
    mine_rules,
    rank_rules,
    readable,
)

ALL_RULES = MiningConfig(min_support=0.4, min_confidence=0.0, min_lift=0.0, max_len=None)


def rule(rules, antecedent, consequent):
    match = rules[(rules["antecedents"] == frozenset(antecedent)) & (rules["consequents"] == frozenset(consequent))]
    assert len(match) == 1, f"rule {antecedent} -> {consequent} not found"
    return match.iloc[0]


def test_metrics_match_hand_calculation(toy):
    rules = mine_rules(toy, ALL_RULES).rules

    beer_diapers = rule(rules, {"Beer"}, {"Diapers"})
    assert beer_diapers["support"] == pytest.approx(3 / 5)          # 3 of 5 baskets
    assert beer_diapers["confidence"] == pytest.approx(3 / 3)       # every Beer basket has Diapers
    assert beer_diapers["lift"] == pytest.approx(1.0 / (4 / 5))     # 1.25
    assert beer_diapers["leverage"] == pytest.approx(3 / 5 - (3 / 5) * (4 / 5))
    assert math.isinf(beer_diapers["conviction"])

    diapers_beer = rule(rules, {"Diapers"}, {"Beer"})
    assert diapers_beer["confidence"] == pytest.approx(3 / 4)
    assert diapers_beer["lift"] == pytest.approx(beer_diapers["lift"])  # lift is symmetric
    assert diapers_beer["conviction"] == pytest.approx((1 - 3 / 5) / (1 - 3 / 4))

    two_to_one = rule(rules, {"Milk", "Diapers"}, {"Beer"})
    assert two_to_one["support"] == pytest.approx(2 / 5)
    assert two_to_one["confidence"] == pytest.approx((2 / 5) / (3 / 5))
    assert two_to_one["lift"] == pytest.approx(((2 / 5) / (3 / 5)) / (3 / 5))


def test_apriori_and_fpgrowth_find_the_same_itemsets_and_rules(demo):
    fp = mine_rules(demo.transactions, MiningConfig(algorithm="fpgrowth"))
    ap = mine_rules(demo.transactions, MiningConfig(algorithm="apriori"))
    assert set(fp.frequent_itemsets["itemsets"]) == set(ap.frequent_itemsets["itemsets"])
    merged = fp.rules.merge(ap.rules, on=["antecedents", "consequents"], suffixes=("_fp", "_ap"))
    assert len(merged) == len(fp.rules) == len(ap.rules) > 0
    for metric in ("support", "confidence", "lift"):
        assert merged[f"{metric}_fp"].to_numpy() == pytest.approx(merged[f"{metric}_ap"].to_numpy())


def test_every_rule_matches_brute_force_counts(demo):
    """Recount support, confidence and lift of every mined rule with plain Python set operations."""
    baskets = [set(basket) for basket in demo.transactions]
    n = len(baskets)
    rules = mine_rules(demo.transactions).rules
    assert len(rules) > 100
    for row in rules.itertuples(index=False):
        a, b = set(row.antecedents), set(row.consequents)
        n_a = sum(a <= basket for basket in baskets)
        n_b = sum(b <= basket for basket in baskets)
        n_ab = sum((a | b) <= basket for basket in baskets)
        assert row.support == pytest.approx(n_ab / n)
        assert row.confidence == pytest.approx(n_ab / n_a)
        assert row.lift == pytest.approx((n_ab / n_a) / (n_b / n))


def test_thresholds_and_max_len_are_applied(demo):
    config = MiningConfig(min_support=0.02, min_confidence=0.3, min_lift=1.5, max_len=2)
    result = mine_rules(demo.transactions, config)
    assert len(result.rules) > 0
    assert (result.frequent_itemsets["support"] >= 0.02).all()
    assert result.frequent_itemsets["length"].max() <= 2
    assert (result.rules["confidence"] >= 0.3).all() and (result.rules["lift"] >= 1.5).all()


def test_excluded_products_are_removed_from_rules(demo):
    full = mine_rules(demo.transactions).rules
    filtered = mine_rules(demo.transactions, MiningConfig(exclude=("Milk",))).rules
    mentions = full["antecedents"].map(lambda s: "Milk" in s) | full["consequents"].map(lambda s: "Milk" in s)
    assert mentions.any()
    assert len(filtered) == len(full) - int(mentions.sum())
    assert not (filtered["antecedents"].map(lambda s: "Milk" in s) | filtered["consequents"].map(lambda s: "Milk" in s)).any()


def test_no_itemsets_or_rules_returns_empty_tables(toy, demo):
    result = mine_rules(demo.transactions, MiningConfig(min_support=0.9))
    assert result.frequent_itemsets.empty and result.rules.empty
    assert list(result.rules.columns) == mining.RULE_COLUMNS
    singles_only = mine_rules(toy, MiningConfig(min_support=0.75))  # only single products are frequent
    assert singles_only.rules.empty
    assert rank_rules(singles_only.rules).empty


def test_too_many_itemsets_raise_a_friendly_error(demo, monkeypatch):
    monkeypatch.setattr(mining, "MAX_FREQUENT_ITEMSETS", 10)
    with pytest.raises(ValueError, match="Raise the minimum support"):
        find_frequent_itemsets(encode(demo.transactions), MiningConfig())


def test_empty_transaction_list_raises():
    with pytest.raises(ValueError, match="No transactions"):
        mine_rules([])


@pytest.mark.parametrize("kwargs", [{"min_support": 0}, {"min_support": 1.5}, {"min_confidence": -0.1},
                                    {"min_confidence": 1.1}, {"min_lift": -1}, {"max_len": 1},
                                    {"algorithm": "eclat"}])
def test_invalid_config_raises(kwargs):
    with pytest.raises(ValueError):
        MiningConfig(**kwargs)


def test_rank_rules_orders_and_validates(toy):
    rules = mine_rules(toy, ALL_RULES).rules
    for metric in ("confidence", "support", "lift"):
        assert rank_rules(rules, metric)[metric].is_monotonic_decreasing
    assert rank_rules(rules, "confidence").equals(rank_rules(rules.sample(frac=1, random_state=1), "confidence"))
    with pytest.raises(ValueError):
        rank_rules(rules, "popularity")


def test_explanations_and_readable_text(toy):
    text = explain_rule(frozenset({"Bread", "Milk"}), frozenset({"Butter"}), 0.041, 0.409, 3.75, 0.109)
    assert "41% of baskets with Bread and Milk also contained Butter" in text
    assert "3.75 times" in text and "(11%)" in text and "4.1% of all transactions" in text
    table = readable(mine_rules(toy, ALL_RULES).rules)
    assert table["antecedents"].map(type).eq(str).all()
    assert "Diapers, Milk" in set(table["antecedents"])
