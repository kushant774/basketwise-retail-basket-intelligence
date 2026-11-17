"""Real-world validation workflow: CLI commands, report files, reproducibility and honest wording."""
import gzip
import io
import json
import zipfile

import pandas as pd
import pytest

from market_basket import real_world
from market_basket.cli import main
from market_basket.temporal import BASELINE, PRIMARY

SMALL_GRID = ["--min-support", "0.02", "0.05", "--min-confidence", "0.1", "--min-lift", "1.2", "--bootstrap", "100"]
REPORT_SECTIONS = ["## 1. Dataset source and attribution", "## 2. Data-quality findings",
                   "## 3. Preprocessing decisions", "## 4. Training and test periods", "## 5. Evaluation method",
                   "## 6. Threshold selection", "## 7. Results on the test period", "## 8. Evaluation settings",
                   "## 9. Limitations",
                   "## 10. How to reproduce"]


@pytest.fixture(scope="module")
def evaluated(uci_csv, tmp_path_factory):
    out = tmp_path_factory.mktemp("real_world")
    assert main(["uci-evaluate", "--data", str(uci_csv), "--output-dir", str(out), *SMALL_GRID]) == 0
    return out


def test_prepare_writes_the_quality_report_and_preprocessing_summary(uci_csv, tmp_path, capsys):
    assert main(["uci-prepare", "--data", str(uci_csv), "--output-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "Removed: cancellation invoice" in out and "Valid invoices (transactions)" in out
    quality = json.loads((tmp_path / "data_quality_report.json").read_text(encoding="utf-8"))
    assert quality["cancellations"]["rows"] == 1 and quality["dataset"]["citation"].startswith("Chen, D. (2015)")
    summary = pd.read_csv(tmp_path / "preprocessing_summary.csv")
    assert summary.iloc[0].tolist() == ["Original rows", quality["rows"]]
    with gzip.open(tmp_path / "removed_rows.csv.gz", "rt", encoding="utf-8") as handle:
        removed = pd.read_csv(handle, dtype=str)
    assert set(removed["reason"]) >= {"cancellation invoice (invoice number starts with C)"}
    assert "CustomerID" not in removed.columns


def test_evaluate_writes_every_output_with_consistent_numbers(evaluated):
    names = {p.name for p in evaluated.iterdir()}
    assert names >= {"real_world_evaluation.json", "real_world_evaluation.md", "data_quality_report.json",
                     "preprocessing_summary.csv", "evaluation_summary.csv", "weekly_results.csv",
                     "threshold_selection.csv", "training_rules.csv", "evaluation_cases.csv.gz", "removed_rows.csv.gz"}
    data = json.loads((evaluated / "real_world_evaluation.json").read_text(encoding="utf-8"))
    assert data["dataset"]["licence"].startswith("Creative Commons Attribution 4.0")
    split = data["split"]
    assert split["training"]["last_invoice"] < split["cutoff"] <= split["test"]["first_invoice"]
    results = {row["method"]: row for row in data["results"]}
    assert set(results) == set(real_world.METHOD_NOTES) and len({r["cases"] for r in results.values()}) == 1
    summary = pd.read_csv(evaluated / "evaluation_summary.csv").set_index("method")
    assert summary.loc[PRIMARY, "hit rate@3"] == pytest.approx(results[PRIMARY]["hit rate@3"])
    with gzip.open(evaluated / "evaluation_cases.csv.gz", "rt", encoding="utf-8") as handle:
        cases = pd.read_csv(handle, dtype={"invoice": str})
    assert len(cases) == results[BASELINE]["cases"] and "customer_id" not in cases.columns
    assert (cases[f"rank: {PRIMARY}"] > 0).mean() == pytest.approx(results[PRIMARY]["hit rate@3"])
    assert data["threshold_selection"]["candidates"] and sum(c["chosen"] for c in data["threshold_selection"]
                                                             ["candidates"]) == 1
    text = (evaluated / "real_world_evaluation.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text


def test_markdown_report_has_every_section_and_honest_wording(evaluated):
    report = (evaluated / "real_world_evaluation.md").read_text(encoding="utf-8")
    for section in REPORT_SECTIONS:
        assert section in report
    assert "Chen, D. (2015). Online Retail [Dataset]" in report and "CC BY 4.0" in report
    assert "What this does not show:" in report and "next-purchase or next-order prediction" in report
    assert "must not be combined" in report
    lowered = report.lower()
    for claim in ("increases sales", "increased sales", "increases revenue", "production-ready", "production ready",
                  "a/b test showed", "predicts the next purchase"):
        assert claim not in lowered
    assert "differs from BasketWise's default protocol" in report   # a small test grid on a non-official file


def test_runs_are_reproducible_apart_from_the_generation_time(uci_csv, evaluated, tmp_path):
    assert main(["uci-evaluate", "--data", str(uci_csv), "--output-dir", str(tmp_path), *SMALL_GRID]) == 0
    first = json.loads((evaluated / "real_world_evaluation.json").read_text(encoding="utf-8"))
    second = json.loads((tmp_path / "real_world_evaluation.json").read_text(encoding="utf-8"))
    first.pop("generated_utc"), second.pop("generated_utc")
    assert first == second
    for name in ("evaluation_cases.csv.gz", "evaluation_summary.csv", "removed_rows.csv.gz", "training_rules.csv",
                 "data_quality_report.json", "preprocessing_summary.csv"):
        assert (evaluated / name).read_bytes() == (tmp_path / name).read_bytes(), name


def test_fixed_cutoff_and_single_thresholds_skip_selection(uci_csv, tmp_path, capsys):
    assert main(["uci-evaluate", "--data", str(uci_csv), "--output-dir", str(tmp_path), "--cutoff", "2025-06-01",
                 "--min-support", "0.03", "--min-confidence", "0.2", "--min-lift", "1.0", "--top-k", "5",
                 "--bootstrap", "0"]) == 0
    assert "chosen on the validation period" not in capsys.readouterr().out
    data = json.loads((tmp_path / "real_world_evaluation.json").read_text(encoding="utf-8"))
    assert data["split"]["cutoff"] == "2025-06-01T00:00:00" and data["split"]["rule"] == "fixed date given by the user"
    assert data["threshold_selection"]["candidates"] == [] and data["bootstrap"] is None
    assert "hit rate@5" in data["results"][0]


@pytest.mark.parametrize("argv", [
    ["uci-evaluate", "--test-share", "1.5"],
    ["uci-evaluate", "--test-share", "0"],
    ["uci-evaluate", "--cutoff", "01/06/2025"],
    ["uci-evaluate", "--cutoff", "2025-06-01", "--test-share", "0.3"],
    ["uci-evaluate", "--top-k", "0"],
    ["uci-evaluate", "--min-support", "abc"],
    ["uci-evaluate", "--bootstrap", "-1"],
])
def test_invalid_command_line_settings_are_usage_errors(argv):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 2


@pytest.mark.parametrize("extra, message", [
    (["--min-support", "0"], "Minimum support"),
    (["--min-confidence", "1.5"], "Minimum confidence"),
    (["--min-lift", "-1"], "Minimum lift"),
    (["--cutoff", "2030-01-01"], "no test invoices"),
    (["--cutoff", "2020-01-01"], "no training invoices"),
])
def test_invalid_evaluation_settings_are_clear_errors(uci_csv, tmp_path, capsys, extra, message):
    assert main(["uci-evaluate", "--data", str(uci_csv), "--output-dir", str(tmp_path), "--bootstrap", "0",
                 *extra]) == 2
    assert message in capsys.readouterr().err


def test_missing_dataset_explains_how_to_download_it(tmp_path, capsys):
    assert main(["uci-prepare", "--data", str(tmp_path / "missing.csv")]) == 2
    assert "download_uci_online_retail.py" in capsys.readouterr().err


def test_the_zip_export_holds_every_result_file(uci_csv):
    source = real_world.prepare(uci_csv, progress=lambda _: None)
    result = real_world.evaluate(source, supports=(0.03,), confidences=(0.2,), lifts=(1.2,), n_bootstrap=0,
                                 progress=lambda _: None)
    with zipfile.ZipFile(io.BytesIO(real_world.export_zip(source, result))) as archive:
        names = set(archive.namelist())
        cases = pd.read_csv(archive.open("evaluation_cases.csv"))
    assert {"real_world_evaluation.json", "real_world_evaluation.md", "evaluation_summary.csv",
            "preprocessing_summary.csv", "data_quality_report.json"} <= names
    assert len(cases) == int(result.summary["cases"].iloc[0])


def test_existing_synthetic_benchmark_is_unchanged(capsys):
    """The published synthetic benchmark (README, docs/EVALUATION_REPORT.md) must not move."""
    assert main(["evaluate", "--splits", "5"]) == 0
    lines = [line.strip() for line in capsys.readouterr().out.splitlines()]
    baseline = next(line for line in lines if line.startswith("Best-seller baseline"))
    assert "5760" in baseline and "0.1971" in baseline and "0.3006" in baseline
    fallback = next(line for line in lines if line.startswith("Rules by confidence + best-seller fallback"))
    assert "0.4348" in fallback and "0.5308" in fallback
    assert main(["rules", "--top", "1"]) == 0
    assert "265 frequent itemsets, 262 rules" in capsys.readouterr().out


# ---------------------------------------------------------------- review findings (honest report text)


@pytest.fixture(scope="module")
def default_run(uci_csv):
    source = real_world.prepare(uci_csv, progress=lambda _: None)
    return source, real_world.evaluate(source, progress=lambda _: None)


def test_the_fixed_protocol_claim_needs_the_official_file_and_every_default(default_run):
    from dataclasses import replace

    from market_basket import uci

    source, result = default_run
    assert not real_world.uses_default_protocol(source, result)      # the fixture is not the official file
    official = replace(source, manifest={"files": {"spreadsheet": {"sha256": uci.OFFICIAL_SPREADSHEET_SHA256},
                                                   "csv": {"sha256": source.sha256}}})
    assert real_world.uses_default_protocol(official, result)
    assert any("matches the default protocol" in note for note in real_world.limitations(official, result))
    for changed in ({"seed": 7}, {"bootstrap_resamples": 50}, {"top_k": 5}, {"test_share": 0.3}):
        assert not real_world.uses_default_protocol(official, replace(result, settings={**result.settings, **changed}))
    strict = replace(official, prepared=replace(source.prepared, summary={**source.prepared.summary,
                                                                          "require_customer_id": True}))
    assert not real_world.uses_default_protocol(strict, result)
    other_sheet = replace(official, manifest={"files": {"spreadsheet": {"sha256": "0" * 64},
                                                        "csv": {"sha256": source.sha256}}})
    assert not real_world.uses_default_protocol(other_sheet, result)
    notes = real_world.limitations(source, result)
    assert any("differs from BasketWise's default protocol" in note for note in notes)
    report = real_world.build_markdown(source, result)
    assert "No download manifest was found" in report and "from the downloaded file" not in report


def test_limitations_describe_the_run_that_was_made(uci_csv):
    source = real_world.prepare(uci_csv, require_customer_id=True, progress=lambda _: None)
    result = real_world.evaluate(source, supports=(0.03,), confidences=(0.2,), lifts=(1.2,), n_bootstrap=0,
                                 progress=lambda _: None)
    notes = " ".join(real_world.limitations(source, result))
    assert "fixed by the user" in notes and "chosen from a small grid" not in notes
    assert "Rows without a customer ID were removed in this run" in notes and "They are kept" not in notes
    assert "Longer rules were not tried" in notes   # the fixture's training period is far below the safety limit
