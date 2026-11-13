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


# ---------------------------------------------------------------- release regressions (I3-I7 and data edge cases)


def test_returns_and_cancellations_are_removed_not_counted_as_purchases():
    data = load("transaction_id,product_name,quantity\nT1,Milk,1\nT1,Bread,2\nT2,Eggs,1\nT2,Tea,-1\n"
                "C1,Milk,-1\nC1,Caviar,-5\n")
    assert data.transactions == [["Milk", "Bread"], ["Eggs"]]  # Tea was returned; C1 was a cancellation
    assert "Caviar" not in data.catalog
    report = data.report
    assert report.returns_removed == 3
    assert report.empty_transactions_removed == 1  # C1 had only returned products
    reasons = data.removed.set_index("product_name")["reason"]
    assert reasons["Caviar"] == reasons["Tea"] == "return or cancellation (negative quantity)"
    table = report.summary_table().set_index("Check")["Result"]
    assert table["Removed: returns or cancellations (negative quantity)"] == "3"
    assert any("negative quantity" in warning for warning in report.warnings)


def test_zero_or_unreadable_quantities_are_still_kept_as_purchases():
    data = load("transaction_id,product_name,quantity\nT1,Milk,0\nT1,Bread,x\nT2,Milk,1\nT2,Tea,1\n")
    assert data.transactions == [["Milk", "Bread"], ["Milk", "Tea"]]
    assert data.report.returns_removed == 0


@pytest.mark.parametrize("dates, first, last", [
    (["05/01/2025", "13/01/2025", "28/02/2025"], "2025-01-05", "2025-02-28"),  # Indian day/month/year
    (["2025-01-05", "2025-01-13", "2025-02-28"], "2025-01-05", "2025-02-28"),  # ISO
    (["01/05/2025", "01/13/2025", "02/28/2025"], "2025-01-05", "2025-02-28"),  # US month/day/year
    (["05.01.25", "13.01.25", "28.02.25"], "2025-01-05", "2025-02-28"),        # dots, two-digit year
    (["12/1/2010 8:26", "12/9/2011 12:50", "1/13/2011 10:00"], "2010-12-01", "2011-12-09"),  # UCI style
])
def test_dates_are_parsed_consistently(dates, first, last):
    rows = "".join(f"T{i},Milk,{d}\nT{i},Bread,{d}\n" for i, d in enumerate(dates))
    data = load("transaction_id,product_name,date\n" + rows)
    stats = summarize(data)
    assert (stats["first date"], stats["last date"]) == (first, last)
    assert data.items["transaction_date"].str.fullmatch(r"\d{4}-\d{2}-\d{2}").all()  # stored as ISO


def test_ambiguous_and_unreadable_dates_are_reported():
    data = load("transaction_id,product_name,date\nT1,Milk,05/01/2025\nT1,Bread,05/01/2025\n"
                "T2,Milk,06/02/2025\nT2,Tea,not a date\n")
    assert summarize(data)["first date"] == "2025-01-05"  # read as day/month/year
    warnings = " ".join(data.report.warnings)
    assert "read as day/month/year" in warnings and "1 date value could not be read" in warnings


def test_weekly_chart_uses_the_same_date_parser():
    from market_basket import plots

    rows = "".join(f"T{i},Milk,{d}\nT{i},Bread,{d}\n" for i, d in enumerate(["13/01/2025", "05/01/2025"]))
    figure = plots.weekly_transactions_chart(load("transaction_id,product_name,date\n" + rows))
    weeks = sorted(str(x)[:10] for x in figure.data[0].x)
    assert weeks == ["2024-12-30", "2025-01-13"]  # weeks of 5 Jan and 13 Jan 2025, not 1 May


def test_uci_online_retail_style_camel_case_headers_are_recognised():
    data = load("InvoiceNo,StockCode,Description,Quantity,InvoiceDate,UnitPrice,CustomerID,Country\n"
                "536365,85123A,WHITE HANGING HEART T-LIGHT HOLDER,6,12/1/2010 8:26,2.55,17850,United Kingdom\n"
                "536365,71053,WHITE METAL LANTERN,6,12/1/2010 8:26,3.39,17850,United Kingdom\n"
                "C536379,D,Discount,-1,12/1/2010 9:41,27.5,14527,United Kingdom\n"
                "536380,85123A,WHITE HANGING HEART T-LIGHT HOLDER,2,12/13/2010 9:41,2.55,17809,United Kingdom\n")
    assert data.report.source_format == "long"
    assert data.transactions == [["WHITE HANGING HEART T-LIGHT HOLDER", "WHITE METAL LANTERN"],
                                 ["WHITE HANGING HEART T-LIGHT HOLDER"]]
    assert {"product_id", "quantity", "transaction_date", "unit_price", "customer_id"} <= set(data.items.columns)
    assert data.report.ignored_columns == ("Country",)
    assert data.report.returns_removed == 1
    assert summarize(data)["last date"] == "2010-12-13"


@pytest.mark.parametrize("header", ["TransactionID,ProductName", "transactionId,productName", "Transaction ID,Item"])
def test_other_header_styles(header):
    assert load(f"{header}\nT1,Milk\nT1,Bread\n").transactions == [["Milk", "Bread"]]


def test_unrelated_columns_are_not_merged_into_known_ones():
    data = load("InvoiceNo,invoice_no,Description\nA1,B1,Milk\nA1,B2,Bread\n")
    assert data.transactions == [["Milk", "Bread"]]  # the first matching column is used
    assert data.report.ignored_columns == ("invoice_no",)


def test_windows_1252_files_keep_curly_apostrophes_dashes_and_euro_signs():
    text = "transaction_id,product_name\nT1,Kellogg’s Corn Flakes\nT1,Bread – White\nT2,Gift Card €10\nT2,Milk\n"
    data = load_transactions(text.encode("cp1252"), name="excel.csv")
    assert {"Kellogg’s Corn Flakes", "Bread – White", "Gift Card €10"} <= set(data.catalog)
    assert data.report.encoding == "Windows-1252"
    assert any("read as Windows-1252" in warning for warning in data.report.warnings)
    utf8 = load(text)
    assert utf8.report.encoding == "UTF-8" and not any("UTF-8" in w for w in utf8.report.warnings)


def test_utf16_and_tsv_files_are_read(tmp_path):
    text = "transaction_id\tproduct_name\nT1\tMilk\nT1\tChai Patti\n"
    data = load_transactions(text.encode("utf-16"), name="unicode.txt")
    assert data.transactions == [["Milk", "Chai Patti"]] and data.report.encoding == "UTF-16"
    path = tmp_path / "export.tsv"
    path.write_text(text, encoding="utf-8")
    assert load_transactions(path).report.delimiter == "\t"


def test_binary_data_after_the_first_kilobytes_is_still_refused():
    payload = b"transaction_id,product_name\n" + b"T1,Milk\n" * 1000 + b"T2,Br\x00ead\n"
    with pytest.raises(DataValidationError, match="binary data"):
        load_transactions(payload)


def test_common_placeholders_and_unicode_variants():
    data = load("transaction_id,product_name\nT1,#N/A\nT1,Milk\n#N/A,Bread\nT2,?\nT2,Milk\n"
                "T3,Café Latte\nT3,Café Latte\nT3,Milk\n")
    assert data.report.blank_products_removed == 2 and data.report.missing_transaction_ids == 1
    assert "Café Latte" in data.catalog and len([p for p in data.catalog if "Latte" in p]) == 1


def test_a_blank_category_never_outvotes_a_real_one():
    data = load("transaction_id,product_name,category\nT1,Milk,\nT2,Milk,\nT3,Milk,Dairy\nT3,Bread,\n")
    assert data.categories == {"Milk": "Dairy", "Bread": "Uncategorised"}
    assert not any("more than one category" in warning for warning in data.report.warnings)


def test_generated_files_are_recognised_as_synthetic_wherever_they_come_from(tmp_path):
    from market_basket.config import PROJECT_ROOT
    from market_basket.synthetic_data import generate_transactions

    issues = load_transactions(PROJECT_ROOT / "Data" / "demo_upload_with_issues.csv")
    assert issues.is_synthetic
    copy = tmp_path / "my_copy.csv"
    copy.write_bytes((PROJECT_ROOT / "Data" / "synthetic_retail_transactions.csv").read_bytes())
    assert load_transactions(copy).is_synthetic
    generated = generate_transactions(n_transactions=300, seed=5).to_csv(index=False).encode()
    assert load_transactions(generated, name="upload.csv").is_synthetic


def test_real_looking_data_is_not_labelled_synthetic():
    assert not load("transaction_id,product_name\nSYN-T000001,Milk\nSYN-T000001,Bread\n").is_synthetic
    mixed = ("transaction_id,product_name,customer_id\nSYN-T000001,Milk,SYN-C00001\n"
             "SYN-T000001,Bread,SYN-C00001\nINV-77,Tea,C-12\nINV-77,Milk,C-12\n")
    assert not load(mixed).is_synthetic
    assert not load("transaction_id,product_name,customer_id\nT1,Milk,C1\nT1,Bread,C1\n").is_synthetic
