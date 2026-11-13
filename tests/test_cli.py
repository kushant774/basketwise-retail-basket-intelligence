import pytest

from market_basket.cli import main


def test_summary(capsys):
    assert main(["summary", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "Synthetic demo data" in out and "8000" in out and "Milk" in out


def test_validate_an_uploaded_style_file(tmp_path, capsys):
    path = tmp_path / "sales.csv"
    path.write_text("transaction_id,product_name\nT1,Milk\nT1,Bread\n,Eggs\nT2,Milk\nT2,Tea\n", encoding="utf-8")
    removed = tmp_path / "new_folder" / "removed.csv"  # the folder does not exist yet
    assert main(["validate", "--data", str(path), "--removed-output", str(removed)]) == 0
    out = capsys.readouterr().out
    assert "Rejected: missing transaction ID" in out and "warning:" in out
    assert "missing transaction ID" in removed.read_text(encoding="utf-8")


def test_rules_and_csv_output(tmp_path, capsys):
    target = tmp_path / "rules.csv"
    assert main(["rules", "--top", "3", "--rank-by", "confidence", "--output", str(target)]) == 0
    assert "frequent itemsets" in capsys.readouterr().out
    assert target.read_text(encoding="utf-8").startswith("antecedents,consequents")


def test_rules_with_impossible_thresholds(capsys):
    assert main(["rules", "--min-support", "0.9"]) == 0
    assert "No rules pass these thresholds" in capsys.readouterr().out


def test_recommend_with_explanations(capsys):
    assert main(["recommend", "milk", "Bread", "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "Butter" in out and "observed data" in out


def test_unknown_product_and_bad_threshold_are_clean_errors(capsys):
    assert main(["recommend", "Mlk"]) == 2
    assert "did you mean: Milk" in capsys.readouterr().err
    assert main(["rules", "--min-confidence", "2"]) == 2
    assert "Minimum confidence" in capsys.readouterr().err


def test_insights_evaluate_and_report(tmp_path, capsys):
    assert main(["insights", "--top", "3"]) == 0
    assert "Potential bundles" in capsys.readouterr().out
    assert main(["evaluate", "--splits", "1"]) == 0
    assert "Best-seller baseline" in capsys.readouterr().out
    assert main(["report", "--output-dir", str(tmp_path), "--splits", "0"]) == 0
    assert (tmp_path / "report.md").exists() and (tmp_path / "basketwise_export.zip").exists()


# ---------------------------------------------------------------- release regressions


def test_recommend_never_suggests_an_excluded_product(capsys):
    assert main(["recommend", "Pasta", "--exclude", "Milk", "--top", "8"]) == 0
    out = capsys.readouterr().out
    lines = [line for line in out.splitlines() if not line.startswith("Basket:")]
    assert "ignoring: Milk" in out and not any("Milk" in line for line in lines)
    assert main(["recommend", "--exclude", "milk", "--exclude", "Bread", "--top", "3"]) == 0  # empty basket
    out = capsys.readouterr().out.split("\n", 1)[1]
    assert "Milk" not in out and "Bread" not in out and "best seller" in out


def test_insights_leave_out_excluded_products(capsys):
    assert main(["insights", "--exclude", "Shampoo", "--top", "20"]) == 0
    assert "Shampoo" not in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["rules", "--min-lift", "nan"], ["rules", "--min-support", "inf"],
                                  ["summary", "--top", "-3"], ["evaluate", "--splits", "0"],
                                  ["summary", "--min-support", "0.5"]])
def test_invalid_numbers_and_misplaced_options_are_usage_errors(argv):
    with pytest.raises(SystemExit) as stop:
        main(argv)
    assert stop.value.code == 2


def test_removed_output_is_written_even_when_nothing_was_removed(tmp_path, capsys):
    target = tmp_path / "removed.csv"
    assert main(["validate", "--removed-output", str(target)]) == 0
    assert "No rows were removed" in capsys.readouterr().out
    assert target.read_text(encoding="utf-8").startswith("file_row,transaction_id,product_name,reason")


def test_report_on_the_demo_upload_keeps_the_synthetic_disclaimer(tmp_path, capsys):
    from market_basket.config import PROJECT_ROOT

    source = PROJECT_ROOT / "Data" / "demo_upload_with_issues.csv"
    assert main(["report", "--data", str(source), "--output-dir", str(tmp_path), "--splits", "0"]) == 0
    assert "synthetic data, not real customers or sales" in (tmp_path / "report.md").read_text(encoding="utf-8")
    assert main(["summary", "--data", str(source)]) == 0
    assert "(synthetic)" in capsys.readouterr().out


def test_folder_given_as_data_is_a_clear_error(tmp_path, capsys):
    assert main(["summary", "--data", str(tmp_path)]) == 2
    assert "is a folder" in capsys.readouterr().err


def test_too_large_settings_are_refused_with_a_clear_message(capsys):
    assert main(["rules", "--min-support", "0.0002", "--max-len", "0", "--min-lift", "0",
                 "--min-confidence", "0"]) == 2  # would otherwise build about 500,000 rules
    assert "Raise the minimum support" in capsys.readouterr().err
