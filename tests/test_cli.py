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
