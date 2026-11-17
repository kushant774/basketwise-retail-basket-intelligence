"""UCI Online Retail: reading, data-quality report, preprocessing and the reproducible download."""
import io
import json
import unicodedata
import zipfile

import pandas as pd
import pytest

from market_basket import uci
from market_basket.uci import DatasetError, data_quality_report, preprocess, preprocessing_table, read_raw
from tests.conftest import UCI_COLUMNS, uci_rows, uci_style_table

T = "2011-01-10 10:00:00"


def reasons(prepared) -> dict[str, int]:
    return prepared.removed["reason"].value_counts().to_dict()


def basket_of(prepared, invoice: str) -> set[str]:
    row = prepared.invoices[prepared.invoices["invoice"] == invoice]
    return set(row["items"].iloc[0]) if len(row) else set()


# ---------------------------------------------------------------- reading


def test_read_raw_keeps_the_eight_columns_as_text(tmp_path):
    path = tmp_path / "retail.csv"
    table = uci_rows(("536365", "85123A", "WHITE HANGING HEART", "6", T, "2.55", "17850", "United Kingdom"))
    table.assign(Extra="x").to_csv(path, index=False)
    raw = read_raw(path)
    assert list(raw.columns) == UCI_COLUMNS
    assert raw.loc[0, "CustomerID"] == "17850" and raw.loc[0, "UnitPrice"] == "2.55"


@pytest.mark.parametrize("content, message", [
    ("", "empty"),
    ("InvoiceNo,StockCode\n1,2\n", "missing column"),
    (",".join(UCI_COLUMNS) + "\n", "no rows"),
])
def test_read_raw_refuses_empty_and_foreign_files(tmp_path, content, message):
    path = tmp_path / "bad.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(DatasetError, match=message):
        read_raw(path)


def test_read_raw_explains_missing_files_and_folders(tmp_path):
    with pytest.raises(DatasetError, match="download_uci_online_retail.py"):
        read_raw(tmp_path / "missing.csv")
    with pytest.raises(DatasetError, match="is a folder"):
        read_raw(tmp_path)


def test_spreadsheet_and_csv_give_the_same_text(tmp_path):
    """The download converts the official .xlsx to CSV; both must read identically (real dates, whole numbers,
    empty customer IDs and non-ASCII text such as the pound sign)."""
    sheet = pd.DataFrame({
        "InvoiceNo": [536365, "C536379"], "StockCode": ["85123A", 22633], "Description": ["MUG £5", None],
        "Quantity": [6, -1], "InvoiceDate": pd.to_datetime(["2010-12-01 08:26", "2010-12-01 09:41"]),
        "UnitPrice": [2.55, 27.5], "CustomerID": [17850.0, None], "Country": ["United Kingdom", "EIRE"],
    })
    xlsx, csv = tmp_path / "Online Retail.xlsx", tmp_path / "online_retail.csv"
    sheet.to_excel(xlsx, index=False)
    uci.spreadsheet_to_csv(xlsx, csv)
    from_csv, from_xlsx = read_raw(csv), read_raw(xlsx)
    pd.testing.assert_frame_equal(from_csv, from_xlsx)
    assert from_csv.loc[0].tolist() == ["536365", "85123A", "MUG £5", "6", "2010-12-01 08:26:00", "2.55", "17850",
                                        "United Kingdom"]
    assert from_csv.loc[1, "CustomerID"] == "" and from_csv.loc[1, "Description"] == ""


# ---------------------------------------------------------------- preprocessing


def test_returns_cancellations_adjustments_and_service_lines_are_never_purchases():
    raw = uci_rows(
        ("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
        ("536365", "71053", "LANTERN", "6", T, "3.39", "17850", "United Kingdom"),
        ("C536379", "85123A", "HEART HOLDER", "-1", T, "2.55", "17850", "United Kingdom"),   # cancellation
        ("C536380", "22633", "HAND WARMER", "2", T, "1.85", "17850", "United Kingdom"),      # C prefix, positive
        ("536381", "22632", "damaged", "-12", T, "0", "", "United Kingdom"),                  # stock adjustment
        ("A563185", "B", "Adjust bad debt", "1", T, "-11062.06", "", "United Kingdom"),       # bad-debt adjustment
        ("536365", "POST", "POSTAGE", "1", T, "18", "17850", "United Kingdom"),               # service line
        ("536365", "gift_0001_20", "Gift Voucher £20.00", "1", T, "16.67", "17850", "United Kingdom"),
        ("536382", "22423", "CAKESTAND", "1", T, "0", "14911", "EIRE"),                        # zero price
        ("536383", "DCGSSBOY", "BOYS PARTY BAG", "1", T, "3.29", "", "United Kingdom"),       # a real product
    )
    prepared = preprocess(raw)
    assert reasons(prepared) == {
        uci.R_CANCELLATION: 2, uci.R_QUANTITY: 1, uci.R_NON_SALE_INVOICE: 1, uci.R_NON_MERCHANDISE: 2,
        uci.R_PRICE: 1,
    }
    assert basket_of(prepared, "536365") == {"85123A", "71053"}
    assert basket_of(prepared, "536383") == {"DCGSSBOY"}
    assert not prepared.invoices["invoice"].str.startswith("C").any()
    assert (prepared.lines["quantity"] > 0).all() and (prepared.lines["unit_price"] > 0).all()


def test_the_first_applicable_reason_is_recorded_with_the_file_row():
    raw = uci_rows(
        ("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
        ("C536379", "POST", "POSTAGE", "-1", "not a date", "0", "", "United Kingdom"),
    )
    removed = preprocess(raw).removed
    assert removed.loc[0, "reason"] == uci.R_CANCELLATION and removed.loc[0, "file_row"] == 3
    assert list(removed.columns) == uci.REMOVED_COLUMNS


def test_missing_customer_ids_are_kept_and_counted_unless_required():
    raw = uci_rows(
        ("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
        ("536366", "71053", "LANTERN", "6", T, "3.39", "", "United Kingdom"),
        ("536366", "22633", "HAND WARMER", "6", T, "1.85", "nan", "United Kingdom"),
    )
    kept = preprocess(raw)
    assert len(kept.invoices) == 2 and kept.summary["invoices_without_customer_id"] == 1
    assert kept.summary["unique_customers"] == 1
    assert "Valid invoices without a customer ID (kept)" in preprocessing_table(kept)["step"].tolist()
    strict = preprocess(raw, require_customer_id=True)
    assert len(strict.invoices) == 1 and reasons(strict) == {uci.R_NO_CUSTOMER: 2}


def test_missing_descriptions_keep_the_row_and_use_the_codes_other_names():
    raw = uci_rows(
        ("536365", "85123A", "", "6", T, "2.55", "17850", "United Kingdom"),
        ("536366", "85123A", "WHITE HANGING HEART", "6", T, "2.55", "17850", "United Kingdom"),
        ("536366", "99999", "?", "1", T, "1.00", "17850", "United Kingdom"),
    )
    prepared = preprocess(raw)
    labels = prepared.labels()
    assert labels["85123A"] == "WHITE HANGING HEART (85123A)"
    assert labels["99999"] == "[no description] (99999)"
    assert prepared.summary["basket_lines_without_description"] == 2 and prepared.removed.empty


def test_exact_duplicates_are_removed_and_repeated_lines_merged():
    raw = uci_rows(
        ("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
        ("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),   # exact duplicate
        ("536365", "85123A", "HEART HOLDER", "4", T, "2.55", "17850", "United Kingdom"),   # second scan
        ("536365", "71053", "LANTERN", "1", T, "3.39", "17850", "United Kingdom"),
    )
    prepared = preprocess(raw)
    assert reasons(prepared) == {uci.R_EXACT_DUPLICATE: 1, uci.R_REPEATED_LINE: 1}
    heart = prepared.lines[prepared.lines["stock_code"] == "85123A"]
    assert len(heart) == 1 and heart["quantity"].iloc[0] == 10   # 6 + 4; the exact copy is not added
    assert prepared.invoices.loc[0, "n_products"] == 2


@pytest.mark.parametrize("date", ["", "2011-02-30 10:00:00", "12/1/2010 8:26", "yesterday"])
def test_invalid_dates_are_removed_never_guessed(date):
    raw = uci_rows(("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
                   ("536366", "71053", "LANTERN", "6", date, "3.39", "17850", "United Kingdom"))
    prepared = preprocess(raw)
    assert reasons(prepared) == {uci.R_BAD_DATE: 1}
    assert data_quality_report(raw)["dates"]["invalid_or_missing_rows"] == 1


@pytest.mark.parametrize("price", ["", "abc", "0", "-2.5"])
def test_invalid_prices_are_removed(price):
    raw = uci_rows(("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
                   ("536366", "71053", "LANTERN", "6", T, price, "17850", "United Kingdom"))
    assert reasons(preprocess(raw)) == {uci.R_PRICE: 1}


def test_unicode_whitespace_and_case_of_codes_are_normalised_without_merging_products():
    nfd = unicodedata.normalize("NFD", "CAFÉ MUG")
    raw = uci_rows(
        ("536365", "85123a", "CAFÉ MUG", "1", T, "2.55", "17850", "United Kingdom"),
        ("536366", " 85123A ", f"  {nfd}  ", "1", T, "2.55", "17850", "United Kingdom"),
        ("536367", "20840", "FRENCH FLORAL CUSHION COVER", "1", T, "3.75", "17850", "United Kingdom"),
        ("536367", "20837", "FRENCH FLORAL CUSHION COVER", "1", T, "3.75", "17850", "United Kingdom"),
    )
    prepared = preprocess(raw)
    labels = prepared.labels()
    assert set(labels) == {"85123A", "20840", "20837"}          # one product per code, case ignored
    assert labels["85123A"] == "CAFÉ MUG (85123A)"               # NFC and single spaces
    assert labels["20840"] != labels["20837"]                    # same description, different products
    assert basket_of(prepared, "536367") == {"20840", "20837"}


def test_empty_tables_and_tables_without_purchases_are_refused():
    with pytest.raises(DatasetError, match="no rows"):
        preprocess(pd.DataFrame(columns=UCI_COLUMNS))
    only_returns = uci_rows(("C536379", "85123A", "HEART", "-1", T, "2.55", "17850", "United Kingdom"))
    with pytest.raises(DatasetError, match="No valid purchase rows"):
        preprocess(only_returns)


def test_preprocessing_counts_add_up_and_are_reproducible():
    raw = pd.concat([uci_style_table(400, seed=5), uci_rows(
        ("C599999", "10001", "MILK", "-1", T, "50", "12001", "United Kingdom"),
        ("599998", "M", "Manual", "1", T, "3", "", "United Kingdom"),
    )], ignore_index=True)
    first, second = preprocess(raw), preprocess(raw)
    pd.testing.assert_frame_equal(first.invoices, second.invoices)
    pd.testing.assert_frame_equal(first.removed, second.removed)
    s, removed = first.summary, first.summary["removed_by_reason"]
    invalid = sum(v for k, v in removed.items() if k not in (uci.R_EXACT_DUPLICATE, uci.R_REPEATED_LINE))
    assert s["original_rows"] == len(raw) == invalid + s["valid_purchase_rows"]
    assert s["valid_purchase_rows"] == removed[uci.R_EXACT_DUPLICATE] + removed[uci.R_REPEATED_LINE] + s["basket_lines"]
    assert s["original_rows"] == len(first.removed) + len(first.lines)
    assert first.invoices["time"].is_monotonic_increasing
    assert s["basket_lines"] == int(first.invoices["n_products"].sum())


def test_quality_report_measures_the_problems():
    raw = uci_rows(
        ("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
        ("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
        ("C536379", "85123a", "HEART HOLDER", "-1", "2011-02-01 09:00:00", "2.55", "17850", "France"),
        ("536381", "22632", "check", "-12", T, "0", "", "United Kingdom"),
        ("536382", "POST", "", "1", "bad", "18", "", "France"),
    )
    report = data_quality_report(raw)
    assert report["rows"] == 5 and report["exact_duplicate_rows"] == 1
    assert report["invoices"]["total"] == 4 and report["invoices"]["cancellation_invoices"] == 1
    assert report["cancellations"] == {"rows": 1, "rows_with_negative_quantity": 1, "rows_with_positive_quantity": 0}
    assert report["quantity"]["negative_rows_outside_cancellations"] == 1
    assert report["unit_price"]["zero_rows"] == 1 and report["dates"]["invalid_or_missing_rows"] == 1
    assert report["missing_values"]["CustomerID"] == 2 and report["customers"]["unique_customer_ids"] == 1
    assert report["descriptions"]["missing_rows"] == 1 and report["descriptions"]["note_like_rows"] == 1
    assert report["products"]["stock_codes"] == 4 and report["products"]["stock_codes_ignoring_case"] == 3
    assert report["products"]["lower_case_twins_sharing_a_description"] == 1
    assert report["countries"]["rows_by_country"] == {"United Kingdom": 3, "France": 2}
    json.dumps(report)  # JSON-serialisable


# ---------------------------------------------------------------- download


def _zip_with_spreadsheet(tmp_path, member: str = uci.XLSX_MEMBER) -> bytes:
    sheet = tmp_path / "sheet.xlsx"
    pd.DataFrame({"InvoiceNo": [536365], "StockCode": ["85123A"], "Description": ["HEART"], "Quantity": [6],
                  "InvoiceDate": pd.to_datetime(["2010-12-01 08:26"]), "UnitPrice": [2.55],
                  "CustomerID": [17850], "Country": ["United Kingdom"]}).to_excel(sheet, index=False)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.write(sheet, member)
    return buffer.getvalue()


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_download_writes_csv_and_manifest_and_reuses_the_zip(tmp_path, monkeypatch):
    payload = _zip_with_spreadsheet(tmp_path)
    calls = []
    monkeypatch.setattr(uci.urllib.request, "urlopen",
                        lambda request, timeout: calls.append(request.full_url) or _Response(payload))
    target = tmp_path / "data"
    manifest = uci.download_dataset(target, progress=lambda _: None)
    assert calls == [uci.DOWNLOAD_URL]
    assert manifest["rows"] == 1 and manifest["licence"].startswith("Creative Commons Attribution 4.0")
    assert manifest["files"]["zip"]["sha256"] == uci.sha256(target / uci.ZIP_NAME)
    assert json.loads((target / uci.MANIFEST_NAME).read_text(encoding="utf-8"))["citation"] == uci.CITATION
    assert read_raw(target / uci.CSV_NAME).loc[0, "InvoiceDate"] == "2010-12-01 08:26:00"
    again = uci.download_dataset(target, progress=lambda _: None)        # the ZIP is reused, not downloaded
    assert calls == [uci.DOWNLOAD_URL] and again["retrieved_utc"] == manifest["retrieved_utc"]
    uci.download_dataset(target, force=True, progress=lambda _: None)
    assert len(calls) == 2


def test_download_retries_a_dropped_connection_and_refuses_an_unexpected_archive(tmp_path, monkeypatch):
    attempts = []

    def flaky(request, timeout):
        attempts.append(1)
        if len(attempts) == 1:
            raise ConnectionResetError("connection reset")
        return _Response(_zip_with_spreadsheet(tmp_path))

    monkeypatch.setattr(uci.urllib.request, "urlopen", flaky)
    uci.download_dataset(tmp_path / "a", progress=lambda _: None)
    assert len(attempts) == 2
    monkeypatch.setattr(uci.urllib.request, "urlopen",
                        lambda request, timeout: _Response(_zip_with_spreadsheet(tmp_path, "other.xlsx")))
    with pytest.raises(DatasetError, match="does not contain"):
        uci.download_dataset(tmp_path / "b", progress=lambda _: None)
    monkeypatch.setattr(uci.urllib.request, "urlopen", lambda request, timeout: _Response(b"not a zip"))
    with pytest.raises(DatasetError, match="not a valid ZIP"):
        uci.download_dataset(tmp_path / "c", progress=lambda _: None)


# ---------------------------------------------------------------- review findings


def test_time_zones_and_a_numeric_voucher_code_are_handled():
    raw = uci_rows(
        ("536365", "85123A", "HEART HOLDER", "6", T, "2.55", "17850", "United Kingdom"),
        ("536366", "71053", "LANTERN", "6", "2011-01-10 10:00:00+01:00", "3.39", "17850", "United Kingdom"),
        ("536367", "71053", "LANTERN", "6", "2011-01-10T10:00:00Z", "3.39", "17850", "United Kingdom"),
        ("536368", "22016", "Dotcomgiftshop Gift Voucher £100.00", "1", T, "83.33", "", "United Kingdom"),
    )
    assert reasons(preprocess(raw)) == {uci.R_BAD_DATE: 2, uci.R_NON_MERCHANDISE: 1}
    assert data_quality_report(raw)["dates"]["invalid_or_missing_rows"] == 2


def test_quality_report_measures_the_time_span_inside_invoices():
    raw = uci_rows(
        ("536365", "85123A", "HEART HOLDER", "6", "2011-01-10 10:00:00", "2.55", "17850", "United Kingdom"),
        ("536365", "71053", "LANTERN", "6", "2011-01-10 10:03:00", "3.39", "17850", "United Kingdom"),
        ("536366", "71053", "LANTERN", "6", T, "3.39", "17850", "United Kingdom"),
    )
    invoices = data_quality_report(raw)["invoices"]
    assert invoices["regular_invoices_with_several_timestamps"] == 1
    assert invoices["longest_time_span_within_an_invoice_minutes"] == 3.0


def test_download_warns_when_the_spreadsheet_is_not_the_published_one(tmp_path, monkeypatch):
    monkeypatch.setattr(uci.urllib.request, "urlopen",
                        lambda request, timeout: _Response(_zip_with_spreadsheet(tmp_path)))
    messages = []
    manifest = uci.download_dataset(tmp_path / "data", progress=messages.append)
    assert manifest["spreadsheet_matches_published_results"] is False
    assert any(message.startswith("WARNING") for message in messages)


def test_a_zip_placed_by_hand_is_used_and_its_retrieval_time_stays_unknown(tmp_path, monkeypatch):
    target = tmp_path / "data"
    target.mkdir()
    (target / uci.ZIP_NAME).write_bytes(_zip_with_spreadsheet(tmp_path))
    monkeypatch.setattr(uci.urllib.request, "urlopen", lambda *a, **k: pytest.fail("must not download"))
    manifest = uci.download_dataset(target, progress=lambda _: None)
    assert manifest["retrieved_utc"] is None and manifest["rows"] == 1
