import pandas as pd
import pytest

from market_basket import evaluation
from market_basket.config import MiningConfig
from market_basket.evaluation import METHODS, build_methods, evaluate, leave_one_out_cases, score, train_test_split
from market_basket.recommend import Recommender


def test_split_is_reproducible_complete_and_disjoint(demo):
    train, test = train_test_split(demo.transactions, test_size=0.2, seed=7)
    assert (train, test) == train_test_split(demo.transactions, test_size=0.2, seed=7)
    assert len(train) == 6400 and len(test) == 1600
    assert sorted(map(tuple, train + test)) == sorted(map(tuple, demo.transactions))
    with pytest.raises(ValueError):
        train_test_split(demo.transactions, test_size=1.0)


def test_leave_one_out_hides_each_product_once():
    cases = list(leave_one_out_cases([["a"], ["a", "b"], ["a", "b", "c"]]))
    assert len(cases) == 5  # single-product baskets cannot be evaluated
    assert (["b"], "a") in cases and (["a", "b"], "c") in cases
    assert all(hidden not in query for query, hidden in cases)


def test_score_metrics_on_a_hand_checked_case():
    rules = pd.DataFrame([{"antecedents": frozenset({"a"}), "consequents": frozenset({"b"}),
                           "support": 0.5, "confidence": 0.9, "lift": 2.0}])
    # case 1: hide a, query [b] -> no rule applies -> miss, unavailable
    # case 2: hide b, query [a] -> recommends b -> hit
    result = score(Recommender(rules), [["a", "b"]], ks=(1, 3), catalog_size=4)
    assert result == {"cases": 2, "hit rate@1": 0.5, "hit rate@3": 0.5, "availability": 0.5, "coverage": 0.25}


def test_rules_and_best_sellers_come_from_training_data_only(demo, monkeypatch):
    """Leakage guard: the mining and popularity functions only ever see the training split."""
    seen = []
    real_mine, real_support = evaluation.mine_rules, evaluation.item_support
    monkeypatch.setattr(evaluation, "mine_rules", lambda tx, cfg: seen.append(("mine", tx)) or real_mine(tx, cfg))
    monkeypatch.setattr(evaluation, "item_support", lambda tx: seen.append(("support", tx)) or real_support(tx))
    evaluate(demo.transactions, n_splits=2, seed=3)
    for split, seed in enumerate((3, 4)):
        train, _ = train_test_split(demo.transactions, 0.2, seed)
        assert seen[2 * split] == ("mine", train) and seen[2 * split + 1] == ("support", train)


def test_products_seen_only_in_test_data_are_never_recommended():
    train = [["a", "b"], ["a", "b"], ["a", "c"], ["b", "c"]] * 5
    test = [["a", "z"], ["z", "b", "c"]]  # "z" never appears in training
    methods = build_methods(train, MiningConfig(min_support=0.1, min_confidence=0.0, min_lift=0.0))
    for recommender in methods.values():
        for query, _ in leave_one_out_cases(test):
            assert "z" not in [rec.product for rec in recommender.recommend(query, 5)]


def test_evaluate_on_demo_data_is_reproducible_and_well_formed(demo):
    first = evaluate(demo.transactions, n_splits=2, seed=1)
    second = evaluate(demo.transactions, n_splits=2, seed=1)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    summary = first.summary.set_index("method")
    assert list(summary.index) == list(METHODS)
    for column in ("hit rate@3", "hit rate@5", "availability", "coverage"):
        assert summary[column].between(0, 1).all()
    assert (summary["hit rate@5"] >= summary["hit rate@3"]).all()
    assert summary.loc["Best-seller baseline", "availability"] == 1.0
    assert pd.isna(summary.loc["Best-seller baseline", "splits beating baseline@3"])
    assert len(first.per_split) == 2 * len(METHODS)


def test_small_dataset_does_not_crash():
    tiny = [["a", "b"], ["a", "b"], ["b", "c"], ["a", "c"], ["a"], ["b", "c"], ["a", "b", "c"], ["c"]]
    result = evaluate(tiny, MiningConfig(min_support=0.2, min_confidence=0.0, min_lift=0.0), n_splits=2,
                      test_size=0.5, seed=0)
    assert len(result.summary) == len(METHODS)


def test_no_evaluable_cases_raises_a_clear_error():
    with pytest.raises(ValueError, match="two or more products"):
        evaluate([["a"], ["b"], ["a", "b"], ["c"]], MiningConfig(min_support=0.2), n_splits=1, test_size=0.25, seed=0)
