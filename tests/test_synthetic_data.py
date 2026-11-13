import pandas as pd
import pytest

from market_basket.config import DEMO_DATA_PATH, MiningConfig
from market_basket.mining import mine_rules
from market_basket.synthetic_data import CATALOG, COLUMNS, generate_transactions, select_products, validate_synthetic


def test_same_seed_gives_identical_data_and_other_seeds_differ():
    first = generate_transactions(n_transactions=1500, seed=11)
    assert first.equals(generate_transactions(n_transactions=1500, seed=11))
    assert not first.equals(generate_transactions(n_transactions=1500, seed=12))


def test_default_dataset_shape_and_labels():
    data = generate_transactions()
    assert list(data.columns) == COLUMNS
    assert data["transaction_id"].nunique() == 8000
    assert data["product_name"].nunique() == len(CATALOG) == 47
    assert data["category"].nunique() == 8
    assert data["transaction_id"].str.startswith("SYN-T").all()
    assert data["customer_id"].str.startswith("SYN-C").all()
    assert validate_synthetic(data) == []


def test_bundled_csv_matches_the_generator():
    """Data/synthetic_retail_transactions.csv must be exactly what scripts/generate_data.py produces."""
    on_disk = pd.read_csv(DEMO_DATA_PATH)
    pd.testing.assert_frame_equal(on_disk, generate_transactions(), check_dtype=False)


def test_transaction_and_product_counts_are_configurable():
    data = generate_transactions(n_transactions=300, n_products=12, seed=3)
    assert data["transaction_id"].nunique() == 300
    assert data["product_name"].nunique() == 12
    assert data["category"].nunique() == 8  # round-robin keeps every category represented
    assert validate_synthetic(data) == []


@pytest.mark.parametrize("kwargs", [{"n_transactions": 0}, {"n_products": 5}, {"n_products": 99}, {"days": 0}])
def test_invalid_settings_raise(kwargs):
    with pytest.raises(ValueError):
        generate_transactions(**kwargs)


def test_select_products_covers_all_categories():
    chosen = select_products(16)
    assert len(chosen) == 16
    assert len({product.category for product in chosen}) == 8


def test_validation_catches_broken_data():
    data = generate_transactions(n_transactions=50, seed=1)
    assert validate_synthetic(data.drop(columns="category")) == ["missing columns: ['category']"]
    broken = data.copy()
    broken.loc[0, "unit_price"] = -5
    broken = pd.concat([broken, broken.iloc[[1]]])
    problems = validate_synthetic(broken)
    assert "unit_price must be positive" in problems
    assert "a product appears twice in the same transaction" in problems


def test_baskets_vary_in_size_and_include_noise():
    data = generate_transactions(n_transactions=2000, seed=5)
    sizes = data.groupby("transaction_id").size()
    assert sizes.min() == 1 and sizes.max() >= 8
    assert 0.05 < (sizes == 1).mean() < 0.3


def test_planted_associations_are_recovered_and_unrelated_products_are_not_linked():
    """Ground truth check: the generator plants Pasta -> Pasta Sauce; Pasta and Shampoo are unrelated."""
    data = generate_transactions(n_transactions=4000, seed=21)
    baskets = data.groupby("transaction_id")["product_name"].agg(list).tolist()
    rules = mine_rules(baskets, MiningConfig(min_support=0.005, min_confidence=0.0, min_lift=0.0, max_len=2)).rules
    pasta = rules[(rules["antecedents"] == frozenset({"Pasta"})) & (rules["consequents"] == frozenset({"Pasta Sauce"}))]
    assert len(pasta) == 1 and pasta.iloc[0]["lift"] > 5
    unrelated = rules[(rules["antecedents"] == frozenset({"Pasta"})) & (rules["consequents"] == frozenset({"Shampoo"}))]
    assert unrelated.empty or unrelated.iloc[0]["lift"] < 2


def test_demo_upload_file_matches_the_generator():
    """Data/demo_upload_with_issues.csv is reproducible from scripts/generate_data.py --with-issues."""
    from market_basket.config import PROJECT_ROOT
    from market_basket.synthetic_data import add_data_quality_issues

    expected, _ = add_data_quality_issues(generate_transactions(n_transactions=3000, seed=7), seed=7)
    on_disk = pd.read_csv(PROJECT_ROOT / "Data" / "demo_upload_with_issues.csv", keep_default_na=False)
    pd.testing.assert_frame_equal(on_disk, expected, check_dtype=False)


def test_validation_reports_exactly_the_injected_problems():
    """Traceability: every problem added on purpose is found and counted by the validator."""
    from market_basket.data import load_transactions
    from market_basket.synthetic_data import add_data_quality_issues

    clean = generate_transactions(n_transactions=2000, seed=5)
    messy, injected = add_data_quality_issues(clean, seed=5)
    report = load_transactions(messy.to_csv(index=False).encode()).report
    assert report.blank_products_removed == injected["blank product names"]
    assert report.missing_transaction_ids == injected["missing transaction IDs"]
    assert report.case_variants_merged == injected["capitalisation variants"]
    assert report.whitespace_fixed == injected["extra spaces"]
    assert report.duplicate_rows_removed == injected["duplicate rows"]
    assert report.ignored_columns == ("store_id",)
    assert report.unique_products == clean["product_name"].nunique()
