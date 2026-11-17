"""Multi-cutoff (rolling-origin) evaluation: fold boundaries, no leakage, pooling, reports and the CLI."""
import gzip
import json
from dataclasses import replace

import pandas as pd
import pytest

from market_basket import real_world, rolling, temporal, uci
from market_basket.cli import build_parser, main
from market_basket.temporal import BASELINE, PRIMARY, candidate_settings, evaluate_chronological
from market_basket.uci import preprocess
from tests.conftest import uci_style_table

GRID = candidate_settings([0.02, 0.05], [0.1], [1.2])
SMALL_GRID = ["--min-support", "0.02", "0.05", "--min-confidence", "0.1", "--min-lift", "1.2", "--bootstrap", "100"]
REPORT_SECTIONS = ["## 1. Dataset", "## 2. Protocol", "## 3. Results per fold", "## 4. Summary over the folds",
                   "## 5. Threshold selection per fold", "## 6. Limitations", "## 7. How to reproduce"]


def quiet(_: str) -> None:
    pass


@pytest.fixture(scope="module")
def realistic():
    return preprocess(uci_style_table(1500, seed=3)).invoices     # 1 January to 30 June 2025


@pytest.fixture(scope="module")
def result(realistic):
    return rolling.evaluate_rolling(realistic, GRID, n_bootstrap=200, progress=quiet)


@pytest.fixture(scope="module")
def written(uci_csv, tmp_path_factory):
    out = tmp_path_factory.mktemp("rolling")
    assert main(["uci-rolling", "--data", str(uci_csv), "--output-dir", str(out), *SMALL_GRID]) == 0
    return out


# ---------------------------------------------------------------- cutoffs and folds


def test_monthly_cutoffs_start_after_the_training_months_and_stop_at_the_last_invoice():
    times = pd.Series(pd.to_datetime(["2025-01-15 10:00", "2025-03-02 00:00", "2025-06-20 17:00"]))
    assert rolling.monthly_cutoffs(times, 3) == [pd.Timestamp(d) for d in ("2025-04-01", "2025-05-01", "2025-06-01")]
    assert rolling.monthly_cutoffs(times, 5) == [pd.Timestamp("2025-06-01")]
    with pytest.raises(ValueError, match="no month is left to test"):
        rolling.monthly_cutoffs(times, 6)
    for months in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="at least 1 month"):
            rolling.monthly_cutoffs(times, months)


def test_folds_test_every_later_invoice_once_and_train_only_on_earlier_ones(realistic, result):
    assert list(result.folds["cutoff"]) == ["2025-04-01", "2025-05-01", "2025-06-01"]
    assert list(result.folds["test_end_exclusive"]) == ["2025-05-01", "2025-06-01", ""]
    tested = set()
    for row, fold in zip(result.folds.to_dict("records"), result.fold_results, strict=True):
        cutoff = pd.Timestamp(row["cutoff"])
        split = fold.split
        assert pd.Timestamp(split["training"]["last_invoice"]) < cutoff <= pd.Timestamp(split["test"]["first_invoice"])
        assert split["training"]["first_invoice"] == realistic["time"].min().isoformat()     # expanding window
        if row["test_end_exclusive"]:
            assert pd.Timestamp(split["test"]["last_invoice"]) < pd.Timestamp(row["test_end_exclusive"])
        invoices = set(fold.cases["invoice"])
        assert tested.isdisjoint(invoices)
        tested |= invoices
    later = realistic[(realistic["time"] >= pd.Timestamp("2025-04-01")) & (realistic["n_products"] >= 2)]
    assert tested == set(later["invoice"])


def test_each_fold_sees_only_invoices_before_its_test_period_ends(realistic, monkeypatch):
    calls = []

    def spy(window, candidates, **options):
        calls.append((window["time"].max(), pd.Timestamp(options["cutoff"])))
        return evaluate_chronological(window, candidates, **options)

    monkeypatch.setattr(rolling, "evaluate_chronological", spy)
    rolling.evaluate_rolling(realistic, GRID, n_bootstrap=0, progress=quiet)
    cutoffs = [cutoff for _, cutoff in calls]
    for (latest, _), next_cutoff in zip(calls[:-1], cutoffs[1:], strict=True):
        assert latest < next_cutoff          # the next month's invoices never reach this fold
    assert calls[-1][0] == realistic["time"].max()


def test_a_fold_matches_the_single_split_protocol_run_on_its_own(realistic, result):
    window = realistic[realistic["time"] < pd.Timestamp("2025-06-01")]
    alone = evaluate_chronological(window, GRID, cutoff="2025-05-01", n_bootstrap=200)
    fold = result.fold_results[1]
    pd.testing.assert_frame_equal(fold.summary, alone.summary)
    assert fold.config == alone.config and fold.intervals == alone.intervals
    row = result.folds.iloc[1]
    summary = alone.summary.set_index("method")
    assert row["hit rate@3: " + PRIMARY] == pytest.approx(summary.loc[PRIMARY, "hit rate@3"])
    assert row["difference"] == pytest.approx(summary.loc[PRIMARY, "hit rate@3"] - summary.loc[BASELINE, "hit rate@3"])
    assert row["min_support"] == alone.config.min_support


def test_pooled_results_weight_every_case_once(result):
    pooled = result.pooled.set_index(["group", "method"])
    every = pooled.loc[(rolling.ALL_FOLDS, PRIMARY)]
    assert every["cases"] == result.folds["cases"].sum() == len(result.cases)
    weighted = (result.folds["hit rate@3: " + PRIMARY] * result.folds["cases"]).sum() / result.folds["cases"].sum()
    assert every["hit rate@3"] == pytest.approx(weighted)
    assert set(result.pooled["method"]) == set(real_world.METHOD_NOTES)
    assert "coverage" not in result.pooled.columns                 # each fold has its own candidate universe


def test_the_early_group_holds_only_folds_that_end_before_the_single_split_cutoff(realistic, result):
    reference = temporal.cutoff_for_share(realistic["time"], 0.2)
    assert result.reference_cutoff == reference
    early = result.folds[result.folds["before the single-split test period"]]
    assert list(early["fold"]) == [f for f, end in zip(result.folds["fold"], result.folds["test_end_exclusive"],
                                                        strict=True) if end and pd.Timestamp(end) <= reference]
    pooled = result.pooled[result.pooled["group"] == rolling.EARLY_FOLDS]
    assert (pooled["cases"] == early["cases"].sum()).all()
    assert pd.Timestamp(result.fold_results[0].split["test"]["last_invoice"]) < reference


def test_verdicts_follow_the_interval_for_the_difference():
    assert rolling._verdict([0.01, 0.05]) == "recommender higher"
    assert rolling._verdict([-0.05, -0.01]) == "baseline higher"
    assert rolling._verdict([-0.01, 0.02]) == "no clear difference"
    assert rolling._verdict([0.0, 0.02]) == "no clear difference"
    assert rolling._verdict(None) == "not computed"


def test_given_cutoffs_are_sorted_and_errors_name_the_fold(realistic):
    given = rolling.evaluate_rolling(realistic, [GRID[0]], cutoffs=["2025-06-01", "2025-04-01", "2025-04-01"],
                                     n_bootstrap=0, progress=quiet)
    assert list(given.folds["cutoff"]) == ["2025-04-01", "2025-06-01"]
    assert given.settings["cutoff_rule"] == "dates given by the user" and given.settings["min_training_months"] is None
    assert given.selection.empty and set(given.folds["verdict"]) == {"not computed"}
    with pytest.raises(ValueError, match=r"Fold 1 \(test period from 2020-01-01\): .*no training invoices"):
        rolling.evaluate_rolling(realistic, [GRID[0]], cutoffs=["2020-01-01"], n_bootstrap=0, progress=quiet)
    with pytest.raises(ValueError, match="at least one threshold"):
        rolling.evaluate_rolling(realistic, [], progress=quiet)


def test_results_are_reproducible(realistic, result):
    again = rolling.evaluate_rolling(realistic, GRID, n_bootstrap=200, progress=quiet)
    pd.testing.assert_frame_equal(result.folds, again.folds)
    pd.testing.assert_frame_equal(result.pooled, again.pooled)
    assert result.intervals == again.intervals


# ---------------------------------------------------------------- reports and command line


def test_the_command_writes_every_output_with_consistent_numbers(written):
    names = {p.name for p in written.iterdir()}
    assert names == {"rolling_evaluation.json", "rolling_evaluation.md", "rolling_folds.csv",
                     "rolling_pooled_summary.csv", "rolling_threshold_selection.csv", "rolling_cases.csv.gz"}
    text = (written / "rolling_evaluation.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    data = json.loads(text)
    assert data["dataset"]["licence"].startswith("Creative Commons Attribution 4.0")
    folds = pd.read_csv(written / "rolling_folds.csv")
    assert len(folds) == len(data["folds"]) == len(data["settings"]["cutoffs"])
    with gzip.open(written / "rolling_cases.csv.gz", "rt", encoding="utf-8") as handle:
        cases = pd.read_csv(handle, dtype={"invoice": str})
    assert "customer_id" not in cases.columns and cases.columns[0] == "fold"
    assert cases.groupby("fold").size().tolist() == folds["cases"].tolist()
    hits = (cases[f"rank: {PRIMARY}"] > 0).groupby(cases["fold"]).mean()
    assert hits.tolist() == pytest.approx(folds[f"hit rate@3: {PRIMARY}"].tolist())
    selection = pd.read_csv(written / "rolling_threshold_selection.csv")
    assert selection.groupby("fold")["chosen"].sum().tolist() == [1] * len(folds)
    assert sum(data["verdict_counts"].values()) == len(folds)


def test_the_report_has_every_section_and_honest_wording(written):
    report = (written / "rolling_evaluation.md").read_text(encoding="utf-8")
    for section in REPORT_SECTIONS:
        assert section in report
    assert "Chen, D. (2015). Online Retail [Dataset]" in report and "CC BY 4.0" in report
    assert "What this does not show:" in report and "must not be added to the single split's numbers" in report
    assert "not corrected for making several comparisons" in report
    lowered = report.lower()
    for claim in ("increases sales", "increased sales", "increases revenue", "production-ready", "production ready",
                  "a/b test showed", "predicts the next purchase", "proves"):
        assert claim not in lowered
    assert "differs from BasketWise's default multi-cutoff protocol" in report   # small grid, not the official file


def test_runs_are_reproducible_apart_from_the_generation_time(uci_csv, written, tmp_path):
    assert main(["uci-rolling", "--data", str(uci_csv), "--output-dir", str(tmp_path), *SMALL_GRID]) == 0
    first = json.loads((written / "rolling_evaluation.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "rolling_evaluation.json").read_text(encoding="utf-8"))
    first.pop("generated_utc"), second.pop("generated_utc")
    assert first == second
    for name in ("rolling_folds.csv", "rolling_pooled_summary.csv", "rolling_threshold_selection.csv",
                 "rolling_cases.csv.gz"):
        assert (written / name).read_bytes() == (tmp_path / name).read_bytes(), name


def test_each_command_keeps_its_own_output_folder():
    parser = build_parser()
    assert parser.parse_args(["uci-rolling"]).output_dir == str(rolling.OUTPUT_DIR)
    assert parser.parse_args(["uci-evaluate"]).output_dir == str(real_world.OUTPUT_DIR)
    assert parser.parse_args(["uci-prepare"]).output_dir == str(real_world.OUTPUT_DIR)
    args = parser.parse_args(["uci-rolling", "--cutoffs", "2011-03-01", "2011-06-01"])
    assert args.cutoffs == ["2011-03-01T00:00:00", "2011-06-01T00:00:00"]


@pytest.mark.parametrize("argv", [
    ["uci-rolling", "--cutoffs", "01/06/2025"],
    ["uci-rolling", "--min-training-months", "0"],
    ["uci-rolling", "--cutoffs", "2025-04-01", "--min-training-months", "2"],
    ["uci-rolling", "--top-k", "0"],
    ["uci-rolling", "--validation-share", "1"],
])
def test_invalid_command_line_settings_are_usage_errors(argv):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 2


def test_too_few_months_is_a_clear_error(uci_csv, tmp_path, capsys):
    assert main(["uci-rolling", "--data", str(uci_csv), "--output-dir", str(tmp_path), "--min-training-months", "12",
                 "--bootstrap", "0"]) == 2
    assert "no month is left to test" in capsys.readouterr().err


@pytest.fixture(scope="module")
def default_run(uci_csv):
    source = real_world.prepare(uci_csv, progress=quiet)
    return source, rolling.evaluate(source, n_bootstrap=50, progress=quiet)


def test_the_fixed_protocol_claim_needs_the_official_file_and_every_default(default_run):
    source, result = default_run
    official = replace(source, manifest={"files": {"spreadsheet": {"sha256": uci.OFFICIAL_SPREADSHEET_SHA256},
                                                   "csv": {"sha256": source.sha256}}})
    defaults = replace(result, settings={**result.settings, "bootstrap_resamples": real_world.DEFAULT_BOOTSTRAP})
    assert not rolling.uses_default_protocol(source, defaults)          # the fixture is not the official file
    assert rolling.uses_default_protocol(official, defaults)
    assert any("decision 25" in note for note in rolling.limitations(official, defaults))
    for changed in ({"seed": 7}, {"bootstrap_resamples": 50}, {"top_k": 5}, {"min_training_months": 4},
                    {"min_training_months": None}, {"validation_share": 0.3}):
        assert not rolling.uses_default_protocol(official, replace(defaults, settings={**defaults.settings,
                                                                                       **changed}))
    assert any("differs from BasketWise's default multi-cutoff protocol" in note
               for note in rolling.limitations(source, defaults))


def unofficial_source(realistic) -> real_world.PreparedSource:
    prepared = preprocess(uci_style_table(1500, seed=3))
    summary = {"require_customer_id": False, "valid_invoices": len(realistic)}
    return real_world.PreparedSource("x.csv", "x.csv", "0" * 64, {}, {}, replace(prepared, summary=summary))


def test_limitations_describe_the_run_that_was_made(realistic):
    source = unofficial_source(realistic)
    short = rolling.evaluate_rolling(realistic, [GRID[0]], cutoffs=["2025-04-01", "2025-06-20"], n_bootstrap=0,
                                     progress=quiet)
    notes = " ".join(rolling.limitations(source, short))
    assert "fixed by the user" in notes and "chose its own thresholds" not in notes
    assert "The last fold's test period is short" in notes and "(11 days)" in notes
    assert "run past the single-split cutoff" in notes
    full = rolling.evaluate_rolling(realistic, GRID, n_bootstrap=0, progress=quiet)
    notes = " ".join(rolling.limitations(source, full))
    assert "chose its own thresholds" in notes and "Every fold chose the same setting" in notes
    assert "short" not in notes and "were refused" not in notes
    assert "In folds 1, 2 and 3, the chosen minimum support (2.00%) is the lowest value in the grid" in notes


def test_settings_refused_by_the_safety_limits_are_named_in_the_limitations(realistic, monkeypatch):
    mine = temporal.mine_rules

    def refuse_low_support(baskets, config):
        if config.min_support == 0.02:
            raise ValueError("Minimum support 2.00% could produce too many itemsets to analyse safely.")
        return mine(baskets, config)

    monkeypatch.setattr(temporal, "mine_rules", refuse_low_support)
    refused = rolling.evaluate_rolling(realistic, GRID, n_bootstrap=0, progress=quiet)
    assert set(refused.folds["min_support"]) == {0.05}
    notes = " ".join(rolling.limitations(unofficial_source(realistic), refused))
    assert "In folds 1, 2 and 3, 3 candidate settings were refused by BasketWise's safety limits" in notes
    assert "(5.00%) is the highest value in the grid" in notes and "lowest value" not in notes
    assert rolling._fold_list([4]) == "fold 4" and rolling._fold_list([2, 1]) == "folds 1 and 2"
