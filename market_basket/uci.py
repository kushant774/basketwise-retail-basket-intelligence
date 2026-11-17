"""The UCI Online Retail dataset: download, data-quality report and preprocessing into baskets.

Source: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.
https://doi.org/10.24432/C5BW33, licensed under CC BY 4.0. The raw data is NOT stored in this
repository; download_dataset() fetches the official ZIP into Data/external/uci_online_retail/
(ignored by Git), checks it and converts the spreadsheet to a CSV once, so later runs are fast.
"""
from __future__ import annotations

import hashlib
import io
import json
import shutil
import urllib.error
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PROJECT_ROOT
from .data import MISSING_TOKENS

SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/352/online+retail"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
DOI = "10.24432/C5BW33"
CITATION = ("Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5BW33.")
LICENCE = "Creative Commons Attribution 4.0 International (CC BY 4.0)"
LICENCE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode"

DATA_DIR = PROJECT_ROOT / "Data" / "external" / "uci_online_retail"
ZIP_NAME = "online_retail.zip"
XLSX_MEMBER = "Online Retail.xlsx"
CSV_NAME = "online_retail.csv"
MANIFEST_NAME = "download_manifest.json"
DEFAULT_CSV_PATH = DATA_DIR / CSV_NAME

# SHA-256 of "Online Retail.xlsx" inside the official ZIP, downloaded on 2026-11-14. The published
# results were computed from this file; a different checksum means UCI changed the file.
OFFICIAL_SPREADSHEET_SHA256 = "43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d"

RAW_COLUMNS = ("InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID",
               "Country")
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024   # the official ZIP is about 23 MB
MAX_SPREADSHEET_BYTES = 200 * 1024 * 1024
USER_AGENT = "BasketWise-dataset-downloader (python-urllib)"


class DatasetError(ValueError):
    """The dataset file is missing, unreadable or not the expected UCI Online Retail table."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fetch(url: str, target: Path, timeout: float, progress: Callable[[str], None]) -> None:
    """Stream url to target through a temporary file, refusing anything over MAX_DOWNLOAD_BYTES.
    timeout applies to each network read, so a stalled connection fails instead of hanging."""
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    received, reported = 0, 0
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, partial.open("wb") as out:
            for block in iter(lambda: response.read(1 << 18), b""):
                received += len(block)
                if received > MAX_DOWNLOAD_BYTES:
                    raise DatasetError(f"The download is larger than {MAX_DOWNLOAD_BYTES // 2**20} MB, which is "
                                       "not the expected UCI Online Retail ZIP; stopped.")
                out.write(block)
                if received - reported >= 2 * 2**20:
                    reported = received
                    progress(f"  {received / 2**20:5.1f} MB received")
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    partial.replace(target)


def _fetch_with_retries(url: str, target: Path, timeout: float, attempts: int,
                        progress: Callable[[str], None]) -> None:
    """The UCI server does not support resuming, so a stalled download is restarted (up to `attempts` times)."""
    for attempt in range(1, attempts + 1):
        try:
            _fetch(url, target, timeout, progress)
            return
        except (OSError, urllib.error.URLError) as error:
            if attempt == attempts:
                raise
            progress(f"  attempt {attempt} failed ({error}); retrying")


def _extract_spreadsheet(zip_path: Path, target: Path) -> None:
    """Copy the one expected spreadsheet out of the ZIP (never extracts other paths)."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as error:
        raise DatasetError(f"{zip_path.name} is not a valid ZIP file. Delete it and download again.") from error
    with archive:
        names = archive.namelist()
        if XLSX_MEMBER not in names:
            raise DatasetError(f"{zip_path.name} does not contain '{XLSX_MEMBER}' (it contains: {', '.join(names)}). "
                               "The official download may have changed; check " + SOURCE_PAGE)
        if archive.getinfo(XLSX_MEMBER).file_size > MAX_SPREADSHEET_BYTES:
            raise DatasetError(f"'{XLSX_MEMBER}' is unexpectedly large; stopped.")
        with archive.open(XLSX_MEMBER) as source, target.open("wb") as out:
            shutil.copyfileobj(source, out)


def spreadsheet_to_csv(xlsx_path: Path, csv_path: Path) -> pd.DataFrame:
    """Convert the official spreadsheet to a UTF-8 CSV with the same eight columns.

    Every cell is written as text exactly as the spreadsheet stores it: dates as yyyy-mm-dd hh:mm:ss
    (the spreadsheet holds real dates, so no day/month guessing is involved), whole numbers without
    a decimal part, and empty cells as empty fields.
    """
    table = pd.read_excel(xlsx_path, sheet_name=0, dtype=object, engine="openpyxl")
    check_columns(table)
    for column in table.columns:
        table[column] = table[column].map(_cell_text)
    table.to_csv(csv_path, index=False, encoding="utf-8", lineterminator="\n")
    return table


def _cell_text(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)) or value is pd.NaT:
        return ""
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def check_columns(table: pd.DataFrame) -> None:
    missing = [column for column in RAW_COLUMNS if column not in table.columns]
    if missing:
        raise DatasetError(f"This is not the UCI Online Retail table: missing column(s) {', '.join(missing)}. "
                           f"Columns found: {', '.join(map(str, table.columns))}.")
    if table.empty:
        raise DatasetError("The UCI Online Retail table has no rows.")


def download_dataset(data_dir: Path = DATA_DIR, url: str = DOWNLOAD_URL, force: bool = False, timeout: float = 60,
                     attempts: int = 3, progress: Callable[[str], None] = print) -> dict:
    """Download the official ZIP (unless it is already there), extract the spreadsheet, convert it to
    CSV and write a manifest with the source, retrieval time, sizes and SHA-256 checksums.

    A ZIP that is already present is reused instead of downloading it again; force=True downloads
    a fresh copy. Returns the manifest.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path, xlsx_path, csv_path = data_dir / ZIP_NAME, data_dir / XLSX_MEMBER, data_dir / CSV_NAME
    manifest_path = data_dir / MANIFEST_NAME
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}

    if zip_path.is_file() and not force:
        progress(f"Using the ZIP already downloaded: {zip_path}")
        retrieved = previous.get("retrieved_utc")
    else:
        progress(f"Downloading {url}")
        _fetch_with_retries(url, zip_path, timeout, attempts, progress)
        retrieved = datetime.now(UTC).replace(microsecond=0).isoformat()
    progress(f"Extracting '{XLSX_MEMBER}'")
    _extract_spreadsheet(zip_path, xlsx_path)
    progress("Reading the spreadsheet and writing a CSV copy (this takes a minute or two)")
    table = spreadsheet_to_csv(xlsx_path, csv_path)
    manifest = {
        "dataset": "Online Retail",
        "source_page": SOURCE_PAGE,
        "download_url": url,
        "doi": f"https://doi.org/{DOI}",
        "citation": CITATION,
        "licence": LICENCE,
        "licence_url": LICENCE_URL,
        "retrieved_utc": retrieved,
        "files": {
            "zip": {"name": ZIP_NAME, "bytes": zip_path.stat().st_size, "sha256": sha256(zip_path)},
            "spreadsheet": {"name": XLSX_MEMBER, "bytes": xlsx_path.stat().st_size, "sha256": sha256(xlsx_path)},
            "csv": {"name": CSV_NAME, "bytes": csv_path.stat().st_size, "sha256": sha256(csv_path)},
        },
        "rows": len(table),
        "columns": list(map(str, table.columns)),
        "converted_with": {"pandas": version("pandas"), "openpyxl": version("openpyxl")},
    }
    manifest["spreadsheet_matches_published_results"] = (manifest["files"]["spreadsheet"]["sha256"]
                                                         == OFFICIAL_SPREADSHEET_SHA256)
    if not manifest["spreadsheet_matches_published_results"]:
        progress("WARNING: this spreadsheet differs from the one BasketWise's published results were computed "
                 "from (its SHA-256 changed), so new results may differ from docs/real_world/.")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    progress(f"Wrote {csv_path} ({len(table):,} rows) and {manifest_path.name}")
    return manifest


def read_manifest(csv_path: Path) -> dict:
    """The download manifest next to a CSV, or {} if there is none (for example a hand-made file)."""
    path = Path(csv_path).with_name(MANIFEST_NAME)
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def read_raw(source: str | Path | bytes) -> pd.DataFrame:
    """Read the UCI table (the converted CSV, or the original .xlsx) with every column as text.
    Raises DatasetError with a user-facing message for missing, empty, oversized or foreign files."""
    if isinstance(source, (bytes, bytearray)):
        handle, name, is_excel = io.BytesIO(source), "the uploaded file", bytes(source[:2]) == b"PK"
        size = len(source)
    else:
        path = Path(source)
        if path.is_dir():
            raise DatasetError(f"'{path}' is a folder; give the path of the UCI CSV (or .xlsx) file.")
        if not path.is_file():
            raise DatasetError(f"Dataset file not found: {path}. Download it first with "
                               "'python scripts/download_uci_online_retail.py'.")
        handle, name, is_excel, size = path, path.name, path.suffix.lower() == ".xlsx", path.stat().st_size
    if size > MAX_SOURCE_BYTES:
        raise DatasetError(f"{name} is {size / 2**20:.0f} MB; the limit is {MAX_SOURCE_BYTES // 2**20} MB "
                           "(the UCI CSV is about 45 MB).")
    try:
        if is_excel:
            table = pd.read_excel(handle, sheet_name=0, dtype=object, engine="openpyxl")
            check_columns(table)
            table = table.apply(lambda column: column.map(_cell_text))
        else:
            table = pd.read_csv(handle, dtype=str, keep_default_na=False, encoding="utf-8")
    except DatasetError:
        raise
    except pd.errors.EmptyDataError as error:
        raise DatasetError(f"{name} is empty.") from error
    except UnicodeDecodeError as error:
        raise DatasetError(f"{name} is not UTF-8 text; use the CSV written by the download script or the "
                           "original .xlsx file.") from error
    except (zipfile.BadZipFile, KeyError, ValueError) as error:
        raise DatasetError(f"{name} could not be read as the UCI Online Retail table: {error}") from error
    check_columns(table)
    return table[list(RAW_COLUMNS)].astype(str).reset_index(drop=True)


# ---------------------------------------------------------------- data quality and preprocessing

# Stock codes of service and accounting lines (not merchandise), found by inspecting every stock code
# that is not the documented five-digit product number: postage and carriage, fees and commissions,
# manual entries, discounts, samples, a bad-debt adjustment and gift vouchers. Two five-digit codes
# are service lines too: every row of 23444 is "Next Day Carriage" and of 22016 a £100 gift voucher.
# DCGS... and PADS codes are real products and stay. All of these codes occur before the test period.
NON_MERCHANDISE_CODES = {
    "POST": "postage", "DOT": "dotcom postage", "C2": "carriage", "23444": "next day carriage",
    "M": "manual entry", "D": "discount", "S": "samples", "B": "bad debt adjustment",
    "BANK CHARGES": "bank charges", "AMAZONFEE": "Amazon fee", "CRUK": "CRUK commission",
    "22016": "gift voucher",
}
GIFT_VOUCHER_CODE = r"GIFT_\d{4}_\d+"
CANCELLATION_PREFIX = "C"

R_MISSING_INVOICE = "missing invoice number"
R_CANCELLATION = "cancellation invoice (invoice number starts with C)"
R_NON_SALE_INVOICE = "non-sale invoice (invoice number with another letter, e.g. A = bad-debt adjustment)"
R_BAD_DATE = "missing or invalid invoice date"
R_MISSING_CODE = "missing stock code"
R_NON_MERCHANDISE = "non-merchandise line (postage, carriage, fees, discounts, manual entries, samples, vouchers)"
R_QUANTITY = "missing, zero or negative quantity (return or stock adjustment)"
R_PRICE = "missing, zero or negative unit price"
R_NO_CUSTOMER = "missing customer ID (excluded because customer IDs were required)"
R_EXACT_DUPLICATE = "exact duplicate of an earlier row"
R_REPEATED_LINE = "repeated product in the same invoice (merged into one basket line)"
REMOVAL_REASONS = (R_MISSING_INVOICE, R_CANCELLATION, R_NON_SALE_INVOICE, R_BAD_DATE, R_MISSING_CODE,
                   R_NON_MERCHANDISE, R_QUANTITY, R_PRICE, R_NO_CUSTOMER, R_EXACT_DUPLICATE, R_REPEATED_LINE)
REMOVED_COLUMNS = ["file_row", "InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice",
                   "Country", "reason"]
MAX_SOURCE_BYTES = 500 * 1024 * 1024


def _missing(values: pd.Series) -> pd.Series:
    return values.str.strip().str.lower().isin(MISSING_TOKENS)


def _as_text(raw: pd.DataFrame) -> pd.DataFrame:
    """The eight UCI columns as text (empty cells become ''), whatever types the caller passed."""
    check_columns(raw)
    table = raw[list(RAW_COLUMNS)].reset_index(drop=True)
    return table.astype(object).where(table.notna(), "").astype(str)


def normalise_text(values: pd.Series) -> pd.Series:
    """One Unicode form (NFC), single spaces and no leading or trailing spaces."""
    return values.str.normalize("NFC").str.replace(r"\s+", " ", regex=True).str.strip()


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Invoice date-times in ISO form (yyyy-mm-dd hh:mm[:ss], as written by the download script);
    anything else becomes NaT and is reported as an invalid date, never guessed. Values with a time
    zone (Z or +hh:mm) are also invalid: the UCI times are local times without a zone."""
    text = values.str.strip()
    zoned = text.str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True).astype(bool)
    try:
        return pd.to_datetime(text.where(~zoned, ""), format="ISO8601", errors="coerce")
    except (ValueError, TypeError):  # a value pandas cannot coerce as a column: parse one by one
        return pd.Series([pd.to_datetime(v, format="ISO8601", errors="coerce") if not z else pd.NaT
                          for v, z in zip(text, zoned, strict=True)], index=values.index, dtype="datetime64[ns]")


def _numbers(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.str.strip(), errors="coerce")


def is_note_like(descriptions: pd.Series) -> pd.Series:
    """Descriptions with lower-case letters and no run of three capitals. Product names in this
    dataset are written in capitals, so these are mostly stock notes ('check', 'damaged', 'found')
    and service lines ('Manual', 'Next Day Carriage'); a few real products also match."""
    return (descriptions.str.contains(r"[a-z]", regex=True)
            & ~descriptions.str.contains(r"[A-Z]{3}", regex=True)).astype(bool)


def is_non_merchandise(codes: pd.Series) -> pd.Series:
    """Service and accounting stock codes (codes already stripped and in upper case)."""
    return codes.isin(list(NON_MERCHANDISE_CODES)) | codes.str.fullmatch(GIFT_VOUCHER_CODE)


def data_quality_report(raw: pd.DataFrame) -> dict:
    """Facts about the raw table, measured on the data (nothing is assumed or hard-coded)."""
    raw = _as_text(raw)
    invoice = raw["InvoiceNo"].str.strip().str.upper()
    code = raw["StockCode"].str.strip()
    description = normalise_text(raw["Description"])
    quantity, price = _numbers(raw["Quantity"]), _numbers(raw["UnitPrice"])
    times = parse_timestamps(raw["InvoiceDate"])
    customer = raw["CustomerID"].str.strip()
    missing_customer = _missing(customer)
    missing_invoice = _missing(invoice)
    cancellation = invoice.str.startswith(CANCELLATION_PREFIX) & ~missing_invoice
    other_letter = ~invoice.str.fullmatch(r"\d+") & ~cancellation & ~missing_invoice
    missing_description = _missing(description)
    known_code = code[~_missing(code)]
    known_description = description[~missing_description]
    valid_times = times.dropna()
    regular = ~(missing_invoice | cancellation | other_letter) & times.notna()
    first_time = times[regular].groupby(invoice[regular]).min()
    last_time = times[regular].groupby(invoice[regular]).max()
    per_month = first_time.dt.to_period("M").value_counts().sort_index()
    countries = raw["Country"].str.strip()
    note_like = is_note_like(description) & ~missing_description
    non_merchandise = is_non_merchandise(code.str.upper())
    descriptions_per_code = known_description.groupby(code.str.upper()[~missing_description]).nunique()
    codes_per_description = code.str.upper()[~missing_description].groupby(known_description).nunique()
    distinct_codes = set(known_code)
    lower_codes = {c for c in distinct_codes if any(ch.islower() for ch in c)}
    lower_with_twin = {c for c in lower_codes if c.upper() in distinct_codes}
    shared = sum(1 for c in lower_with_twin
                 if set(known_description[code[~missing_description] == c])
                 & set(known_description[code[~missing_description] == c.upper()]))
    return {
        "rows": len(raw),
        "columns": list(RAW_COLUMNS),
        "missing_values": {column: int(_missing(raw[column]).sum()) for column in RAW_COLUMNS},
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "invoices": {
            "total": int(invoice[~missing_invoice].nunique()),
            "cancellation_invoices": int(invoice[cancellation].nunique()),
            "other_non_numeric_invoices": int(invoice[other_letter].nunique()),
            "other_non_numeric_examples": sorted(invoice[other_letter].unique().tolist())[:5],
            "regular_invoices": int(invoice[regular].nunique()),
            "regular_invoices_with_several_timestamps": int((times[regular].groupby(invoice[regular]).nunique()
                                                             > 1).sum()),
            "longest_time_span_within_an_invoice_minutes": float(
                (last_time - first_time).dt.total_seconds().max() / 60 if regular.any() else 0.0),
            "regular_invoices_per_month": {str(period): int(count) for period, count in per_month.items()},
        },
        "cancellations": {
            "rows": int(cancellation.sum()),
            "rows_with_negative_quantity": int((cancellation & (quantity < 0)).sum()),
            "rows_with_positive_quantity": int((cancellation & (quantity > 0)).sum()),
        },
        "quantity": {
            "unreadable_rows": int(quantity.isna().sum()),
            "negative_rows": int((quantity < 0).sum()),
            "negative_rows_outside_cancellations": int(((quantity < 0) & ~cancellation).sum()),
            "zero_rows": int((quantity == 0).sum()),
        },
        "unit_price": {
            "unreadable_rows": int(price.isna().sum()),
            "zero_rows": int((price == 0).sum()),
            "negative_rows": int((price < 0).sum()),
            "zero_price_rows_with_customer_id": int(((price == 0) & ~missing_customer).sum()),
            "minimum": float(price.min()) if price.notna().any() else None,
            "maximum": float(price.max()) if price.notna().any() else None,
        },
        "dates": {
            "invalid_or_missing_rows": int(times.isna().sum()),
            "first": valid_times.min().isoformat() if len(valid_times) else None,
            "last": valid_times.max().isoformat() if len(valid_times) else None,
            "days_with_invoices": int(valid_times.dt.normalize().nunique()),
        },
        "products": {
            "stock_codes": int(known_code.nunique()),
            "stock_codes_ignoring_case": int(known_code.str.upper().nunique()),
            "rows_with_lower_case_stock_code": int(known_code.str.contains(r"[a-z]", regex=True).sum()),
            "lower_case_stock_codes": len(lower_codes),
            "lower_case_stock_codes_with_upper_case_twin": len(lower_with_twin),
            "lower_case_twins_sharing_a_description": shared,
            "descriptions": int(known_description.nunique()),
            "stock_codes_with_several_descriptions": int((descriptions_per_code > 1).sum()),
            "descriptions_shared_by_several_stock_codes": int((codes_per_description > 1).sum()),
            "non_merchandise_rows": int(non_merchandise.sum()),
            "non_merchandise_rows_by_code": {str(k): int(v) for k, v in
                                             code.str.upper()[non_merchandise].value_counts().items()},
        },
        "descriptions": {
            "missing_rows": int(missing_description.sum()),
            "note_like_rows": int(note_like.sum()),
            "note_like_examples": description[note_like].value_counts().head(10).index.tolist(),
            "descriptions_with_non_ascii_characters": int(
                known_description[~known_description.map(str.isascii).astype(bool)].nunique()),
        },
        "customers": {
            "unique_customer_ids": int(customer[~missing_customer].nunique()),
            "rows_without_customer_id": int(missing_customer.sum()),
            "regular_invoices_without_customer_id": int(invoice[regular & missing_customer].nunique()),
        },
        "countries": {
            "count": int(countries[~_missing(countries)].nunique()),
            "rows_by_country": {str(k): int(v) for k, v in countries.value_counts().items()},
            "regular_invoices_by_country": {str(k): int(v) for k, v in
                                            countries[regular].groupby(invoice[regular]).first()
                                            .value_counts().items()},
        },
    }


@dataclass(frozen=True)
class PreparedRetail:
    """The cleaned dataset: one row per invoice-product line, one row per invoice, and the products."""
    lines: pd.DataFrame      # invoice, stock_code, quantity, unit_price, invoice_time, customer_id, country
    invoices: pd.DataFrame   # invoice, time, customer_id, country, n_products, items (tuple of stock codes)
    products: pd.DataFrame   # stock_code, description, label
    removed: pd.DataFrame    # REMOVED_COLUMNS: each removed raw row with its reason
    summary: dict            # preprocessing counts (see preprocessing_table)
    require_customer_id: bool = False

    def labels(self) -> dict[str, str]:
        return dict(zip(self.products["stock_code"], self.products["label"], strict=True))


def preprocess(raw: pd.DataFrame, require_customer_id: bool = False) -> PreparedRetail:
    """Turn the raw UCI rows into purchase baskets, keeping every removed row with its reason.

    Each row gets the FIRST reason that applies, in the order of REMOVAL_REASONS. Cancellation
    invoices and negative quantities are never counted as purchases. Products are identified by
    their stock code (trimmed, upper case); descriptions are only display names. Rows without a
    customer ID are kept (a basket is an invoice), unless require_customer_id is True.
    """
    raw = _as_text(raw)
    invoice = raw["InvoiceNo"].str.strip().str.upper()
    code = raw["StockCode"].str.strip().str.upper()
    description = normalise_text(raw["Description"])
    quantity, price = _numbers(raw["Quantity"]), _numbers(raw["UnitPrice"])
    times = parse_timestamps(raw["InvoiceDate"])
    customer = raw["CustomerID"].str.strip()
    missing_customer = _missing(customer)

    reason = pd.Series(None, index=raw.index, dtype=object)

    def flag(mask: pd.Series, text: str) -> None:
        reason[mask & reason.isna()] = text

    flag(_missing(invoice), R_MISSING_INVOICE)
    flag(invoice.str.startswith(CANCELLATION_PREFIX), R_CANCELLATION)
    flag(~invoice.str.fullmatch(r"\d+"), R_NON_SALE_INVOICE)
    flag(times.isna(), R_BAD_DATE)
    flag(_missing(code), R_MISSING_CODE)
    flag(is_non_merchandise(code), R_NON_MERCHANDISE)
    flag(quantity.isna() | (quantity <= 0), R_QUANTITY)
    flag(price.isna() | (price <= 0), R_PRICE)
    if require_customer_id:
        flag(missing_customer, R_NO_CUSTOMER)
    valid_rows = int(reason.isna().sum())
    # identical rows have identical validity, so an exact duplicate of a valid row is itself valid here
    flag(raw.duplicated(keep="first"), R_EXACT_DUPLICATE)
    remaining = reason.isna()
    keys = pd.DataFrame({"invoice": invoice, "code": code})
    flag(keys[remaining].duplicated(keep="first").reindex(raw.index, fill_value=False), R_REPEATED_LINE)

    kept = reason.isna()
    if not kept.any():
        raise DatasetError("No valid purchase rows remain after cleaning (every row is a cancellation, adjustment, "
                           "non-merchandise line or has an invalid date, quantity or price). Dates must look like "
                           "yyyy-mm-dd hh:mm:ss, as in the CSV written by the download script.")
    # a repeated line adds its quantity to the kept line (a second scan); an exact duplicate row does not
    merged_quantity = quantity.where(kept | (reason == R_REPEATED_LINE)).groupby(
        [invoice, code], sort=False).transform("sum")
    lines = pd.DataFrame({
        "invoice": invoice[kept], "stock_code": code[kept], "quantity": merged_quantity[kept],
        "unit_price": price[kept], "invoice_time": times[kept],
        "customer_id": customer[kept].where(~missing_customer[kept], ""), "country": raw["Country"][kept].str.strip(),
    }).reset_index(drop=True)

    grouped = lines.groupby("invoice", sort=False)
    known_customer = lines["customer_id"].mask(lines["customer_id"] == "")
    invoices = pd.DataFrame({
        "time": grouped["invoice_time"].min(),   # the few invoices spanning two minutes take their first time
        "customer_id": known_customer.groupby(lines["invoice"], sort=False).first().fillna(""),
        "country": grouped["country"].first(),
        "n_products": grouped.size(),
        "items": grouped["stock_code"].agg(tuple),
    }).rename_axis("invoice").reset_index()
    invoices = invoices.sort_values(["time", "invoice"], kind="stable", ignore_index=True)

    products = _product_names(code[kept], description[kept], _missing(description)[kept])
    positions = np.flatnonzero(~kept.to_numpy())
    removed = raw.iloc[positions].assign(file_row=positions + 2, reason=reason.iloc[positions].to_numpy())
    removed = removed[REMOVED_COLUMNS]
    counts = reason[~kept].value_counts()
    summary = {
        "original_rows": len(raw),
        "removed_by_reason": {text: int(counts.get(text, 0)) for text in REMOVAL_REASONS
                              if text != R_NO_CUSTOMER or require_customer_id},
        "valid_purchase_rows": valid_rows,
        "basket_lines": len(lines),
        "valid_invoices": len(invoices),
        "invoices_with_two_or_more_products": int((invoices["n_products"] >= 2).sum()),
        "median_products_per_invoice": float(invoices["n_products"].median()),
        "largest_invoice_products": int(invoices["n_products"].max()),
        "unique_products": int(lines["stock_code"].nunique()),
        "unique_customers": int(lines.loc[lines["customer_id"] != "", "customer_id"].nunique()),
        "invoices_without_customer_id": int((invoices["customer_id"] == "").sum()),
        "basket_lines_without_description": int(_missing(description)[kept].sum()),
        "basket_lines_with_note_like_description": int(is_note_like(description)[kept].sum()),
        "first_invoice": invoices["time"].min().isoformat(),
        "last_invoice": invoices["time"].max().isoformat(),
        "require_customer_id": require_customer_id,
    }
    return PreparedRetail(lines, invoices, products, removed.reset_index(drop=True), summary, require_customer_id)


def _product_names(codes: pd.Series, descriptions: pd.Series, missing: pd.Series) -> pd.DataFrame:
    """One display name per stock code: its most frequent description (ties: alphabetical). Labels add
    the code, so two products with the same description stay distinguishable."""
    known = pd.DataFrame({"stock_code": codes[~missing], "description": descriptions[~missing]})
    counts = known.groupby(["stock_code", "description"]).size().rename("n").reset_index()
    counts = counts.sort_values(["stock_code", "n", "description"], ascending=[True, False, True], kind="stable")
    names = counts.drop_duplicates("stock_code").set_index("stock_code")["description"]
    products = pd.DataFrame({"stock_code": sorted(codes.unique())})
    products["description"] = products["stock_code"].map(names).fillna("")
    products["label"] = [f"{name} ({code})" if name else f"[no description] ({code})"
                         for code, name in zip(products["stock_code"], products["description"], strict=True)]
    return products


def preprocessing_table(prepared: PreparedRetail) -> pd.DataFrame:
    """The preprocessing summary as rows of (step, count), in pipeline order."""
    s = prepared.summary
    rows = [("Original rows", s["original_rows"])]
    rows += [(f"Removed: {text}", count) for text, count in s["removed_by_reason"].items()
             if text not in (R_EXACT_DUPLICATE, R_REPEATED_LINE)]
    rows += [
        ("Valid purchase rows", s["valid_purchase_rows"]),
        (f"Removed: {R_EXACT_DUPLICATE}", s["removed_by_reason"][R_EXACT_DUPLICATE]),
        (f"Merged: {R_REPEATED_LINE}", s["removed_by_reason"][R_REPEATED_LINE]),
        ("Basket lines (invoice-product pairs)", s["basket_lines"]),
        ("Valid invoices (transactions)", s["valid_invoices"]),
        ("Invoices with two or more products", s["invoices_with_two_or_more_products"]),
        ("Unique products (stock codes)", s["unique_products"]),
        ("Unique customers", s["unique_customers"]),
        ("Valid invoices without a customer ID (kept)" if not prepared.require_customer_id
         else "Valid invoices without a customer ID", s["invoices_without_customer_id"]),
    ]
    return pd.DataFrame(rows, columns=["step", "count"])
