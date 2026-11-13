import pandas as pd
import pytest

from market_basket import data as data_module
from market_basket.data import (
    DataValidationError,
    clean_table,
    encode,
    item_support,
    load_transactions,
    read_csv_table,
    summarize,
    validate_products,
)


def load(text: str):
    return load_transactions(text.encode("utf-8"), name="test.csv")


# ---------------------------------------------------------------- long format


def test_long_format_with_aliases_whitespace_and_optional_columns():
    data = load("Invoice No, Item ,Qty,Category\nA1,  Milk ,2,Dairy\nA1,Bread,1,Bakery\nA2,Milk  Tea,1,Drinks\n")
    assert data.report.source_format == "long"
    assert data.transactions == [["Milk", "Bread"], ["Milk Tea"]]
    assert data.report.whitespace_fixed == 2  # "  Milk " and "Milk  Tea"
    assert data.categories == {"Milk": "Dairy", "Bread": "Bakery", "Milk Tea": "Drinks"}
    assert data.items["quantity"].tolist() == [2, 1, 1]
    assert data.report.warnings  # few transactions


def test_missing_ids_blank_products_and_placeholders_are_rejected_and_listed():
    data = load("transaction_id,product_name\nT1,Milk\n,Bread\nT2,\nT2,nan\nT3,Eggs\nT3,NULL\n")
    assert data.transactions == [["Milk"], ["Eggs"]]
    report = data.report
    assert report.missing_transaction_ids == 1
    assert report.blank_products_removed == 3
    assert report.empty_transactions_removed == 1  # T2 had no valid product
    reasons = data.removed.set_index("file_row")["reason"].to_dict()
    assert reasons == {3: "missing transaction ID", 4: "blank product name",
                       5: "blank product name", 7: "blank product name"}
    assert "rejected or removed" in report.warnings[0]


def test_duplicates_are_merged_and_quantities_added():
    data = load("transaction_id,product_name,quantity\nT1,Milk,2\nT1,Milk,1\nT1,Bread,1\nT2,Bread,x\n")
    assert data.report.duplicate_rows_removed == 1
    milk = data.items[(data.items.transaction_id == "T1") & (data.items.product_name == "Milk")]
    assert milk["quantity"].tolist() == [3]
    assert data.items.loc[data.items.transaction_id == "T2", "quantity"].isna().all()
    assert any("invalid quantity" in w for w in data.report.warnings)
    assert "duplicate product" in data.removed["reason"].iloc[0]


def test_capitalisation_variants_merge_to_the_most_common_or_title_case_spelling():
    data = load("transaction_id,product_name\nT1,BREAD\nT2,Bread\nT3,bread\nT4,milk\nT4,MILK\nT5,Milk\nT6,milk\n")
    assert data.catalog == ["Bread", "milk"]  # tie -> Title Case; otherwise the most common spelling wins
    assert data.report.case_variants_merged == 4
    assert data.report.duplicate_rows_removed == 1  # T4 had milk twice after merging


def test_unexpected_columns_are_reported():
    data = load("transaction_id,product_name,store,cashier\nT1,Milk,A,Z\nT2,Bread,A,Y\n")
    assert data.report.ignored_columns == ("store", "cashier")
    assert any("store, cashier" in w for w in data.report.warnings)


# ---------------------------------------------------------------- wide format


def test_wide_format_orders_items_ignores_padding_and_reports_problems():
    text = ("basket_id,item_2,item_1,item_3\n"
            "B1,Butter,Bread,\n"
            "B2,,,\n"
            ",Milk,,\n"
            "B1,Jam,,\n"
            "B3,Tea,TEA,Sugar\n")
    data = load(text)
    report = data.report
    assert report.source_format == "wide"
    assert data.transactions == [["Bread", "Butter", "Jam"], ["Tea", "Sugar"]]
    assert report.rows_received == 5
    assert report.empty_transactions_removed == 1
    assert report.missing_transaction_ids == 1
    assert report.blank_products_removed == 0  # empty item cells are padding, not removed records
    assert report.duplicate_rows_removed == 1  # TEA merged into Tea
    assert any("more than one row" in w for w in report.warnings)


def test_wide_format_without_transaction_ids_uses_row_numbers():
    data = load("item_1,item_2\nMilk,Bread\nTea,Sugar\n")
    assert data.items["transaction_id"].unique().tolist() == ["ROW-2", "ROW-3"]
    assert any("row numbers" in w for w in data.report.warnings)


# ---------------------------------------------------------------- invalid files


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"", "empty"),
        (b"   \n  ", "empty"),
        (b"transaction_id,product_name\n", "no data rows"),
        (b"PK\x03\x04\x00\x00binary", "not a text CSV"),
        (b"colour,size\nred,10\n", "Could not recognise the columns"),
        (b"transaction_id,item_1\nT1,a,b,c\n", "more values than the header"),
        (b"transaction_id,item_1\nT1,a\nT2,b,c,d\n", "more values than the header"),
        (b"transaction_id,product_name\n,Milk\n,Bread\n", "No valid transactions"),
        (b"transaction_id,product_name\nT1,Milk\nT2,Milk\n", "At least two different products"),
    ],
)
def test_invalid_files_raise_clear_errors(payload, message):
    with pytest.raises(DataValidationError, match=message):
        load_transactions(payload)


def test_non_csv_extension_and_missing_file(tmp_path):
    other = tmp_path / "data.xlsx"
    other.write_bytes(b"x")
    with pytest.raises(DataValidationError, match="not a CSV"):
        read_csv_table(other)
    with pytest.raises(FileNotFoundError):
        read_csv_table(tmp_path / "missing.csv")


def test_size_and_row_limits(monkeypatch):
    monkeypatch.setattr(data_module, "MAX_UPLOAD_MB", 0.00001)
    with pytest.raises(DataValidationError, match="limit is"):
        load("transaction_id,product_name\nT1,Milk\nT1,Bread\n")
    monkeypatch.setattr(data_module, "MAX_UPLOAD_MB", 50)
    monkeypatch.setattr(data_module, "MAX_UPLOAD_ROWS", 1)
    with pytest.raises(DataValidationError, match="rows; the limit"):
        load("transaction_id,product_name\nT1,Milk\nT1,Bread\n")


def test_latin1_file_is_read():
    data = load_transactions("transaction_id,product_name\nT1,Café\nT1,Crème\n".encode("latin-1"))
    assert data.catalog == ["Café", "Crème"]


# ---------------------------------------------------------------- demo data, summaries, encoding


def test_demo_data_is_clean_and_labelled(demo):
    assert demo.is_synthetic and demo.source == "Synthetic demo data"
    assert demo.report.valid_transactions == 8000
    assert demo.report.unique_products == 47
    assert demo.removed.empty
    assert len(set(demo.categories.values())) == 8
    stats = summarize(demo)
    assert stats["transactions"] == 8000 and stats["categories"] == 8
    assert stats["first date"] == "2025-01-01" and stats["last date"] == "2025-06-30"


def test_item_support_counts_each_transaction_once(toy):
    support = item_support(toy + [["Milk", "Milk"]])
    assert support["Milk"] == pytest.approx(5 / 6)
    assert list(support.index[:3]) == ["Milk", "Bread", "Diapers"]  # ties broken by name


def test_dense_and_sparse_encoding_match(toy, monkeypatch):
    dense = encode(toy, sparse=False)
    sparse = encode(toy, sparse=True)
    assert dense.shape == (5, 6)
    assert (sparse.sparse.to_dense().astype(bool) == dense).all().all()
    monkeypatch.setattr(data_module, "SPARSE_CELL_THRESHOLD", 10)
    assert isinstance(encode(toy)["Milk"].dtype, pd.SparseDtype)


def test_validate_products():
    catalog = ["Milk", "Bread", "Peanut Butter"]
    assert validate_products([" milk ", "MILK", "peanut  butter", ""], catalog) == ["Milk", "Peanut Butter"]
    with pytest.raises(ValueError, match="did you mean: Milk"):
        validate_products(["Mlk"], catalog)


def test_clean_table_accepts_a_dataframe():
    table = pd.DataFrame({"order_id": ["1", "1", "2"], "product": ["a", "b", "a"]})
    data = clean_table(table)
    assert data.transactions == [["a", "b"], ["a"]]


# ---------------------------------------------------------------- real-world export variations


@pytest.mark.parametrize("separator, name", [(";", "semicolon"), ("\t", "tab"), ("|", "pipe")])
def test_semicolon_tab_and_pipe_separated_files(separator, name):
    text = separator.join(["transaction_id", "product_name"]) + "\n" + "\n".join(
        separator.join(row) for row in [("T1", "Milk"), ("T1", "Bread"), ("T2", "Milk"), ("T2", "Tea")])
    data = load(text + "\n")
    assert data.transactions == [["Milk", "Bread"], ["Milk", "Tea"]]
    assert data.report.summary_table().set_index("Check").loc["Column separator", "Result"] == name


def test_empty_unnamed_spreadsheet_columns_are_dropped_quietly():
    data = load("transaction_id,product_name,,\nT1,Milk,,\nT1,Bread,,\n")
    assert data.report.ignored_columns == ()
    assert data.report.warnings == ("Only 1 transaction. Rules from fewer than a few hundred transactions "
                                    "are unreliable.",)


def test_unusual_product_names_survive_the_whole_pipeline():
    import io
    import zipfile

    from market_basket.config import MiningConfig
    from market_basket.mining import mine_rules
    from market_basket.recommend import Recommender
    from market_basket.reporting import build_report, export_zip

    long_name = "Organic cold-pressed extra virgin olive oil, family pack (5 litres)"
    text = ('transaction_id,product_name\n'
            '001,"Rice, Basmati"\n001,Café Crème\n'
            '002,7-Up\n002,"Rice, Basmati"\n'
            '003,100% Juice\n003,Ölive Oil | 1L\n'
            f'004,"{long_name}"\n004,7-Up\n'
            '005,7-Up\n005,"Rice, Basmati"\n')
    data = load(text)
    assert data.items["transaction_id"].iloc[0] == "001"  # leading zeros are kept
    assert {"Rice, Basmati", "Café Crème", "Ölive Oil | 1L", long_name} <= set(data.catalog)
    result = mine_rules(data.transactions, MiningConfig(min_support=0.2, min_confidence=0.0, min_lift=0.0))
    recommendations = Recommender(result.rules).recommend(["7-Up"])
    assert recommendations[0].product == "Rice, Basmati"  # bought with 7-Up in 2 of 3 baskets
    report = build_report(data, result)
    assert "Ölive Oil / 1L" in report  # pipes are escaped in Markdown tables
    archive = zipfile.ZipFile(io.BytesIO(export_zip(data, result)))
    cleaned = pd.read_csv(archive.open("cleaned_transactions.csv"), dtype=str)
    assert set(cleaned["product_name"]) == set(data.catalog)


def test_large_upload_is_loaded_and_mined():
    import time

    from market_basket.mining import mine_rules
    from market_basket.synthetic_data import generate_transactions

    payload = generate_transactions(n_transactions=30_000, seed=3).to_csv(index=False).encode()
    start = time.perf_counter()
    data = load_transactions(payload, name="large.csv")
    rules = mine_rules(data.transactions).rules
    assert data.report.valid_transactions == 30_000 and len(rules) > 100
    assert time.perf_counter() - start < 60  # typically about 2 seconds on a laptop
