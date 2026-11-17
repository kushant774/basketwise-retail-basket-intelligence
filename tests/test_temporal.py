"""Chronological evaluation: split by time, no leakage, identical cases, metrics and reproducibility."""
import pandas as pd
import pytest

from market_basket import temporal
from market_basket.config import MiningConfig
from market_basket.data import item_support
from market_basket.evaluation import METHODS, build_methods, leave_one_out_cases, methods_from_rules
from market_basket.mining import mine_rules
from market_basket.temporal import (
    BASELINE,
    PRIMARY,
    candidate_settings,
    cutoff_for_share,
    evaluate_chronological,
    score_cases,
    select_setting,
    split_by_time,
    summarise,
)
from market_basket.uci import preprocess
from tests.conftest import uci_style_table

LOOSE = MiningConfig(min_support=0.05, min_confidence=0.0, min_lift=0.0, max_len=2)


def invoices(rows) -> pd.DataFrame:
    """Invoice table from (invoice, time, items) tuples, in the layout uci.preprocess makes."""
    frame = pd.DataFrame(rows, columns=["invoice", "time", "items"])
    frame["time"] = pd.to_datetime(frame["time"])
    frame["items"] = frame["items"].map(tuple)
    frame["customer_id"] = ""
    frame["country"] = "United Kingdom"
    frame["n_products"] = frame["items"].map(len)
    return frame


@pytest.fixture(scope="module")
def realistic():
    return preprocess(uci_style_table(1500, seed=3)).invoices


# ---------------------------------------------------------------- split


def test_cutoff_is_a_midnight_chosen_by_invoice_count():
    table = invoices([(f"{i}", f"2011-01-{1 + i // 2:02d} 1{i % 2}:00", ["a", "b"]) for i in range(20)])
    cutoff = cutoff_for_share(table["time"], 0.2)
    assert cutoff == pd.Timestamp("2011-01-09")          # first invoice of the latest 20% is on 9 January
    train, test = split_by_time(table, cutoff)
    assert len(train) == 16 and len(test) == 4


def test_split_never_leaks_dates_and_keeps_identical_timestamps_together(realistic):
    cutoff = cutoff_for_share(realistic["time"], 0.25)
    train, test = split_by_time(realistic, cutoff)
    assert train["time"].max() < cutoff <= test["time"].min()
    assert set(train["invoice"]).isdisjoint(test["invoice"])
    assert len(train) + len(test) == len(realistic)
    same_time = invoices([("1", "2011-01-01 09:00", ["a", "b"]), ("2", "2011-01-02 10:00", ["a", "b"]),
                          ("3", "2011-01-02 10:00", ["b", "c"]), ("4", "2011-01-03 10:00", ["a", "c"])])
    for cut in ("2011-01-02", "2011-01-03"):
        early, late = split_by_time(same_time, cut)
        assert {"2", "3"} <= set(early["invoice"]) or {"2", "3"} <= set(late["invoice"])


def test_impossible_splits_are_clear_errors(realistic):
    with pytest.raises(ValueError, match="no training invoices"):
        split_by_time(realistic, "2020-01-01")
    with pytest.raises(ValueError, match="no test invoices"):
        split_by_time(realistic, "2030-01-01")
    for share in (0, 1, 1.5, -0.1):
        with pytest.raises(ValueError, match="between 0 and 1"):
            cutoff_for_share(realistic["time"], share)
    one_day = invoices([("1", "2011-01-01 09:00", ["a", "b"]), ("2", "2011-01-01 10:00", ["a", "b"])])
    with pytest.raises(ValueError, match="no test invoices"):
        split_by_time(one_day, cutoff_for_share(one_day["time"], 0.5))


# ---------------------------------------------------------------- leakage


def test_rules_and_best_sellers_are_learned_from_training_invoices_only(realistic, monkeypatch):
    seen = []
    monkeypatch.setattr(temporal, "mine_rules", lambda baskets, cfg: seen.append(("mine", baskets))
                        or mine_rules(baskets, cfg))
    monkeypatch.setattr(temporal, "item_support", lambda baskets: seen.append(("support", baskets))
                        or item_support(baskets))
    candidates = candidate_settings([0.02, 0.05], [0.1], [1.2])
    result = evaluate_chronological(realistic, candidates, test_share=0.2, validation_share=0.25, n_bootstrap=0)
    cutoff = pd.Timestamp(result.split["cutoff"])
    validation_cut = pd.Timestamp(result.validation["cutoff"])
    train, test = split_by_time(realistic, cutoff)
    inner, _ = split_by_time(train, validation_cut)
    inner_baskets, train_baskets = [list(b) for b in inner["items"]], [list(b) for b in train["items"]]
    # selection: one popularity table and one mining run per candidate, all on the early training part only
    assert seen[:3] == [("support", inner_baskets), ("mine", inner_baskets), ("mine", inner_baskets)]
    # final model: the whole training period, never a test invoice
    assert seen[3:] == [("mine", train_baskets), ("support", train_baskets)]
    assert validation_cut < cutoff


def test_baseline_popularity_comes_from_the_training_period():
    early = [("1", "2011-01-01", ["a", "b"]), ("2", "2011-01-02", ["a", "c"]), ("3", "2011-01-03", ["a", "d"])]
    late = [(str(i), "2011-02-01", ["z", "y"]) for i in range(4, 10)] + [("10", "2011-02-02", ["b", "c"])]
    result = evaluate_chronological(invoices(early + late), [LOOSE], cutoff="2011-02-01", n_bootstrap=0)
    assert result.best_sellers.index[0] == "a"            # z and y dominate the future but are unseen in training
    assert "z" not in result.best_sellers.index


def test_products_first_sold_in_the_test_period_are_misses_for_every_method():
    table = invoices([(f"{i}", f"2011-01-{1 + i:02d}", ["a", "b", "c"]) for i in range(10)]
                     + [("20", "2011-02-01", ["a", "new"])])
    result = evaluate_chronological(table, [LOOSE], cutoff="2011-02-01", n_bootstrap=0)
    cases = result.cases.set_index("hidden_product")
    assert not cases.loc["new", "hidden_product_in_training"]
    for method in METHODS:
        assert cases.loc["new", f"rank: {method}"] == 0
    assert len(result.cases) == 2                         # the miss stays in the denominator
    assert result.settings["candidate_universe"] == 3


# ---------------------------------------------------------------- cases and metrics


def test_leave_one_out_and_metrics_on_a_hand_checked_example():
    train = [["a", "b"]] * 6 + [["a", "c"]] * 2 + [["d"]] * 3
    mined = mine_rules(train, MiningConfig(min_support=0.1, min_confidence=0.5, min_lift=1.0, max_len=2))
    methods = methods_from_rules(mined.rules, item_support(train))
    test = invoices([("t1", "2011-02-01", ["a", "b"]), ("t2", "2011-02-02", ["c", "d", "e"])])
    known = frozenset("abcd")
    cases, recommended = score_cases(methods, test, 1, known)
    assert list(zip(cases["invoice"], cases["hidden_product"], strict=True)) == [
        ("t1", "a"), ("t1", "b"), ("t2", "c"), ("t2", "d"), ("t2", "e")]
    by_case = cases.set_index("hidden_product")
    # hide b from {a, b}: the rule a -> b (confidence 0.75) puts b first; best sellers say a (in basket), then b
    assert by_case.loc["b", f"rank: {PRIMARY}"] == 1 and by_case.loc["b", f"rank: {METHODS[1]}"] == 1
    # hide a from {a, b}: b -> a applies (confidence 1.0), so a is found by the rules
    assert by_case.loc["a", f"rank: {METHODS[1]}"] == 1
    # hide c from {c, d, e}: no rule has d or e on the left, so rules alone suggest nothing (an empty case)
    assert by_case.loc["c", f"shown: {METHODS[1]}"] == 0 and by_case.loc["c", f"rank: {METHODS[1]}"] == 0
    # hide d or e: c -> a applies and suggests a, which is a miss; e was never sold in training
    assert (by_case.loc[["d", "e"], f"shown: {METHODS[1]}"] == 1).all()
    assert not by_case.loc["e", "hidden_product_in_training"] and by_case.loc["d", "hidden_product_in_training"]
    summary = summarise(cases, list(methods), 1, len(known), recommended).set_index("method")
    rules_only = summary.loc[METHODS[1]]
    assert rules_only["hit rate@1"] == pytest.approx(2 / 5)
    assert rules_only["precision@1"] == pytest.approx(2 / 5) and rules_only["recall@1"] == pytest.approx(2 / 5)
    assert rules_only["availability"] == pytest.approx(4 / 5)
    assert rules_only["hit rate@1, hidden product seen in training"] == pytest.approx(2 / 4)
    assert rules_only["coverage"] == pytest.approx(len(recommended[METHODS[1]]) / 4)
    assert rules_only["invoice-averaged hit rate@1"] == pytest.approx((1.0 + 0.0) / 2)
    assert summary["cases"].nunique() == 1 and summary.loc[BASELINE, "availability"] == 1.0


def test_precision_divides_by_k_and_recall_equals_hit_rate(realistic):
    result = evaluate_chronological(realistic, [MiningConfig(0.02, 0.1, 1.2, 2)], n_bootstrap=0, top_k=3)
    summary = result.summary
    assert (summary["precision@3"] - summary["hit rate@3"] / 3).abs().max() < 1e-12
    assert (summary["recall@3"] == summary["hit rate@3"]).all()
    assert summary[["hit rate@3", "availability", "coverage"]].apply(lambda c: c.between(0, 1)).all().all()


def test_every_method_is_scored_on_the_same_cases(realistic):
    result = evaluate_chronological(realistic, [MiningConfig(0.02, 0.1, 1.2, 2)], n_bootstrap=0)
    summary = result.summary
    assert summary["cases"].nunique() == 1 and summary["invoices"].nunique() == 1
    assert list(summary["method"]) == list(METHODS)
    expected = sum(1 for basket in split_by_time(realistic, result.split["cutoff"])[1]["items"]
                   for _ in leave_one_out_cases([list(basket)]))
    assert summary["cases"].iloc[0] == len(result.cases) == expected
    assert all(f"rank: {m}" in result.cases and f"shown: {m}" in result.cases for m in METHODS)


def test_restricting_rules_to_a_basket_never_changes_a_recommendation(demo):
    """The evaluation speed-up (Recommender.for_basket) returns exactly what the full recommender returns."""
    methods = build_methods(demo.transactions[:6000], MiningConfig())
    for basket in demo.transactions[6000:6300]:
        for name, recommender in methods.items():
            restricted = recommender.for_basket(basket)
            for query, _ in leave_one_out_cases([basket]):
                assert restricted.recommend(query, 5) == recommender.recommend(query, 5), name


def test_no_evaluable_test_invoice_is_a_clear_error():
    table = invoices([(f"{i}", f"2011-01-{1 + i:02d}", ["a", "b"]) for i in range(10)]
                     + [("20", "2011-02-01", ["a"]), ("21", "2011-02-02", ["b"])])
    with pytest.raises(ValueError, match="two or more products"):
        evaluate_chronological(table, [LOOSE], cutoff="2011-02-01")


def test_invalid_settings_are_refused(realistic):
    with pytest.raises(ValueError, match="at least one value"):
        candidate_settings([], [0.1], [1.2])
    with pytest.raises(ValueError, match="Minimum support"):
        candidate_settings([0], [0.1], [1.2])
    with pytest.raises(ValueError, match="Minimum confidence"):
        candidate_settings([0.01], [1.5], [1.2])
    with pytest.raises(ValueError, match="at least 1"):
        evaluate_chronological(realistic, [LOOSE], top_k=0)
    with pytest.raises(ValueError, match="at least one threshold"):
        evaluate_chronological(realistic, [])


# ---------------------------------------------------------------- selection, intervals and reproducibility


def test_thresholds_are_chosen_on_validation_with_deterministic_ties(realistic):
    train, _ = split_by_time(realistic, cutoff_for_share(realistic["time"], 0.2))
    candidates = candidate_settings([0.02, 0.05], [0.1, 0.9], [1.2])
    chosen, table, details = select_setting(train, candidates, 0.25, 3)
    assert len(table) == 4 and table["chosen"].sum() == 1
    compared = table[table["status"] == "compared"]
    best = compared["validation hit rate@3"].max()
    winners = compared[compared["validation hit rate@3"].round(10) == round(best, 10)]
    top = winners.sort_values(["min_support", "min_confidence"], ascending=False).iloc[0]
    assert (chosen.min_support, chosen.min_confidence) == (top["min_support"], top["min_confidence"])
    assert pd.Timestamp(details["validation_part"]["last_invoice"]) <= train["time"].max()


def test_settings_refused_by_the_safety_limit_are_reported_not_fatal(realistic, monkeypatch):
    train, _ = split_by_time(realistic, cutoff_for_share(realistic["time"], 0.2))
    real = temporal.mine_rules

    def refuse_small_support(baskets, config):
        if config.min_support < 0.03:
            raise ValueError("too many itemsets")
        return real(baskets, config)

    monkeypatch.setattr(temporal, "mine_rules", refuse_small_support)
    chosen, table, _ = select_setting(train, candidate_settings([0.02, 0.05], [0.1], [1.2]), 0.25, 3)
    assert chosen.min_support == 0.05 and table.loc[0, "status"] == "refused: too many itemsets"
    monkeypatch.setattr(temporal, "mine_rules", lambda baskets, config: (_ for _ in ()).throw(ValueError("no")))
    with pytest.raises(ValueError, match="None of the candidate settings"):
        select_setting(train, candidate_settings([0.02], [0.1, 0.2], [1.2]), 0.25, 3)


def test_results_are_reproducible(realistic):
    candidates = candidate_settings([0.02, 0.05], [0.1], [1.2])
    first = evaluate_chronological(realistic, candidates, n_bootstrap=200, seed=7)
    second = evaluate_chronological(realistic, candidates, n_bootstrap=200, seed=7)
    pd.testing.assert_frame_equal(first.cases, second.cases)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.selection, second.selection)
    assert first.intervals == second.intervals and first.split == second.split


def test_bootstrap_intervals_are_paired_and_contain_the_estimate(realistic):
    result = evaluate_chronological(realistic, [MiningConfig(0.02, 0.1, 1.2, 2)], n_bootstrap=300, seed=1)
    summary = result.summary.set_index("method")
    for method in METHODS:
        low, high = result.intervals["hit_rate"][method]
        assert low <= summary.loc[method, "hit rate@3"] <= high
    difference = result.intervals["difference_vs_baseline"][PRIMARY]
    estimate = summary.loc[PRIMARY, "hit rate@3"] - summary.loc[BASELINE, "hit rate@3"]
    assert difference["interval"][0] <= estimate <= difference["interval"][1]
    assert BASELINE not in result.intervals["difference_vs_baseline"]
    assert result.weekly["cases"].sum() == len(result.cases)


# ---------------------------------------------------------------- review findings (selection errors and fallback)


def test_selection_errors_name_the_validation_period():
    one_day = invoices([(f"{i}", f"2011-01-01 0{i}:00", ["a", "b"]) for i in range(1, 6)]
                       + [("9", "2011-01-05 10:00", ["a", "b"])])
    with pytest.raises(ValueError, match="too short to set aside a validation period"):
        evaluate_chronological(one_day, candidate_settings([0.1, 0.2], [0.1], [1.2]), cutoff="2011-01-05")
    single_at_end = invoices([(f"{i}", f"2011-01-{1 + i:02d}", ["a", "b"]) for i in range(8)]
                             + [("8", "2011-01-20", ["a"]), ("9", "2011-01-21", ["b"]),
                                ("10", "2011-02-01", ["a", "b"])])
    with pytest.raises(ValueError, match="validation period .* has no invoice with two or more products"):
        evaluate_chronological(single_at_end, candidate_settings([0.1, 0.2], [0.1], [1.2]), cutoff="2011-02-01",
                               validation_share=0.2)


def test_refused_settings_give_one_clean_message(realistic, monkeypatch):
    train, _ = split_by_time(realistic, cutoff_for_share(realistic["time"], 0.2))
    monkeypatch.setattr(temporal, "mine_rules",
                        lambda baskets, config: (_ for _ in ()).throw(ValueError("Too many itemsets.")))
    with pytest.raises(ValueError) as error:
        select_setting(train, candidate_settings([0.02, 0.05], [0.1], [1.2]), 0.25, 3)
    message = str(error.value)
    assert "early part of the training period: Too many itemsets." in message
    assert "refused:" not in message and ".." not in message


def test_a_setting_refused_on_the_whole_training_period_falls_back_to_the_next_best(realistic, monkeypatch):
    candidates = candidate_settings([0.02, 0.05], [0.1], [1.2])
    train, _ = split_by_time(realistic, cutoff_for_share(realistic["time"], 0.2))
    winner, _, _ = select_setting(train, candidates, 0.2, 3)
    real = temporal.mine_rules

    def refuse_winner_on_all_training(baskets, config):
        if config == winner and len(baskets) == len(train):
            raise ValueError("over the limit")
        return real(baskets, config)

    monkeypatch.setattr(temporal, "mine_rules", refuse_winner_on_all_training)
    result = evaluate_chronological(realistic, candidates, validation_share=0.2, n_bootstrap=0)
    assert result.config != winner
    table = result.selection
    refused = table[table["status"].str.startswith("refused on the whole training period")]
    assert len(refused) == 1 and not refused["chosen"].iloc[0]
    assert table["chosen"].sum() == 1 and table.loc[table["chosen"], "validation rank"].iloc[0] == 2
