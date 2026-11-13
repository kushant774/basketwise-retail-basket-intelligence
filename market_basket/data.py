"""Read, validate, clean, summarise and encode transaction data.

Two CSV layouts are accepted (column names are matched case-insensitively, with common aliases):

* Long format: one row per product in a transaction.
  Required: transaction_id, product_name. Optional: category, quantity, unit_price,
  transaction_date, customer_id.
* Wide format: one row per transaction.
  Columns: transaction_id (optional), item_1, item_2, item_3, ...

Column names are matched after normalising them, so "InvoiceNo", "Invoice No" and "invoice_no"
are the same column (the UCI Online Retail headers InvoiceNo, StockCode, Description, Quantity,
InvoiceDate, UnitPrice and CustomerID are recognised).

Cleaning never drops data silently: every removed or rejected row is kept in
TransactionData.removed with a reason, and the ValidationReport counts each kind of change.
Rows with a negative quantity are returns or cancellations; they are removed (with that
reason) so that a returned product is never counted as a purchase.
"""
from __future__ import annotations

import codecs
import difflib
import io
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from .config import DEMO_DATA_PATH, MAX_UPLOAD_MB, MAX_UPLOAD_ROWS

MISSING_TOKENS = ("", "nan", "none", "null", "n/a", "na", "-", "#n/a", "?")
TRANSACTION_ALIASES = ("transaction_id", "transaction", "transaction_no", "invoice", "invoice_no", "invoice_id",
                       "invoice_number", "order_id", "order", "basket_id", "receipt_id", "bill_no")
PRODUCT_ALIASES = ("product_name", "product", "item_name", "item", "product_description", "description")
OPTIONAL_ALIASES = {
    "product_id": ("product_id", "sku", "item_id", "stock_code"),
    "category": ("category", "product_category", "department"),
    "quantity": ("quantity", "qty", "units"),
    "unit_price": ("unit_price", "price"),
    "transaction_date": ("transaction_date", "date", "order_date", "invoice_date"),
    "customer_id": ("customer_id", "customer"),
}
WIDE_ITEM_COLUMN = re.compile(r"^(?:item|product)_?(\d+)$")
DELIMITER_NAMES = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
SPARSE_CELL_THRESHOLD = 10_000_000
REMOVED_COLUMNS = ["file_row", "transaction_id", "product_name", "reason"]
TEXT_ENCODINGS = (("utf-8-sig", "UTF-8"), ("cp1252", "Windows-1252"), ("latin-1", "Latin-1"))
DATE_PARTS = r"^(\d{1,4})[/.-](\d{1,2})[/.-](\d{1,4})$"
# Files written by scripts/generate_data.py (and the in-app generator) use these ID formats.
GENERATED_TRANSACTION_ID = r"SYN-T\d{6,}"
GENERATED_CUSTOMER_ID = r"SYN-C\d{5,}"


class DataValidationError(ValueError):
    """Raised when a file cannot be turned into usable transactions; the message is user-facing."""


@dataclass(frozen=True)
class ValidationReport:
    source_format: str
    rows_received: int
    valid_rows: int
    valid_transactions: int
    unique_products: int
    missing_transaction_ids: int
    blank_products_removed: int
    duplicate_rows_removed: int
    empty_transactions_removed: int
    whitespace_fixed: int
    case_variants_merged: int
    ignored_columns: tuple[str, ...]
    warnings: tuple[str, ...]
    delimiter: str = ","
    returns_removed: int = 0
    encoding: str = "UTF-8"

    def summary_table(self) -> pd.DataFrame:
        unit = "transactions (rows)" if self.source_format == "wide" else "product lines (rows)"
        rows = [
            ("File layout", "wide: one row per transaction" if self.source_format == "wide"
             else "long: one row per product"),
            (f"Total {unit} received", f"{self.rows_received:,}"),
            ("Valid transactions", f"{self.valid_transactions:,}"),
            ("Valid product lines", f"{self.valid_rows:,}"),
            ("Unique products", f"{self.unique_products:,}"),
            ("Rejected: missing transaction ID", f"{self.missing_transaction_ids:,}"),
            ("Removed: blank product names", f"{self.blank_products_removed:,}"),
            ("Removed: returns or cancellations (negative quantity)", f"{self.returns_removed:,}"),
            ("Removed: duplicate product in a transaction", f"{self.duplicate_rows_removed:,}"),
            ("Removed: transactions with no products", f"{self.empty_transactions_removed:,}"),
            ("Fixed: extra spaces", f"{self.whitespace_fixed:,}"),
            ("Merged: capitalisation variants", f"{self.case_variants_merged:,}"),
            ("Ignored columns", ", ".join(self.ignored_columns) or "none"),
            ("Column separator", DELIMITER_NAMES.get(self.delimiter, self.delimiter)),
            ("Text encoding", self.encoding),
        ]
        return pd.DataFrame(rows, columns=["Check", "Result"])


@dataclass(frozen=True)
class TransactionData:
    items: pd.DataFrame          # cleaned long format, one row per product in a transaction
    removed: pd.DataFrame        # removed or rejected rows with the reason
    report: ValidationReport
    source: str
    is_synthetic: bool = False

    @cached_property
    def transactions(self) -> list[list[str]]:
        return self.items.groupby("transaction_id", sort=False)["product_name"].agg(list).tolist()

    @cached_property
    def catalog(self) -> list[str]:
        return sorted(self.items["product_name"].unique())

    @cached_property
    def categories(self) -> dict[str, str]:
        if "category" not in self.items:
            return {}
        return self.items.drop_duplicates("product_name").set_index("product_name")["category"].to_dict()


# ---------------------------------------------------------------- reading


def _check_size(n_bytes: int) -> None:
    size_mb = n_bytes / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise DataValidationError(
            f"The file is {size_mb:.1f} MB; the limit is {MAX_UPLOAD_MB} MB. Upload a shorter period of data."
        )


def decode_text(raw: bytes) -> tuple[str, str]:
    """Decode a text file and name the encoding used.

    UTF-16 is recognised by its byte-order mark (Excel's "Unicode text" export). Otherwise UTF-8
    (with or without a byte-order mark) is tried first, then Windows-1252 (the usual encoding of
    Excel CSV exports on Windows, which keeps characters such as ’ – €), and Latin-1 last.
    """
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return raw.decode("utf-16"), "UTF-16"
        except UnicodeDecodeError as error:
            raise DataValidationError("The file looks like UTF-16 text but could not be decoded.") from error
    if b"\x00" in raw:
        raise DataValidationError("This is not a text CSV file (it contains binary data). Export it as CSV first.")
    for codec, label in TEXT_ENCODINGS:
        try:
            return raw.decode(codec), label
        except UnicodeDecodeError:
            continue
    raise AssertionError("Latin-1 decodes any bytes")  # pragma: no cover


def read_csv_table(source: str | Path | bytes) -> pd.DataFrame:
    """Read a CSV file path or uploaded bytes into a DataFrame of strings, with user-facing errors."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in (".csv", ".txt", ".tsv"):
            raise DataValidationError(f"'{path.name}' is not a CSV file. Export the data as .csv and try again.")
        _check_size(path.stat().st_size)  # before reading, so a huge file is never loaded
        raw = path.read_bytes()
    else:
        raw = bytes(source)

    if not raw.strip():
        raise DataValidationError("The file is empty.")
    _check_size(len(raw))
    text, encoding = decode_text(raw)
    ragged_message = (
        "Some rows have more values than the header has columns. In wide format, add enough item columns "
        "(item_1, item_2, ...) for the largest transaction."
    )
    delimiter = detect_delimiter(text)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", pd.errors.ParserWarning)
            table = pd.read_csv(io.StringIO(text), sep=delimiter, dtype=str, keep_default_na=False,
                                skip_blank_lines=True, index_col=False)
    except pd.errors.EmptyDataError as error:
        raise DataValidationError("The file has no readable rows.") from error
    except pd.errors.ParserError as error:
        raise DataValidationError(f"{ragged_message} Details: {error}") from error
    if any(issubclass(w.category, pd.errors.ParserWarning) for w in caught):
        raise DataValidationError(ragged_message)
    if table.empty:
        raise DataValidationError("The file has a header row but no data rows.")
    if len(table) > MAX_UPLOAD_ROWS:
        raise DataValidationError(
            f"The file has {len(table):,} rows; the limit is {MAX_UPLOAD_ROWS:,}. Upload a shorter period of data."
        )
    # spreadsheet exports often add trailing columns with no header and no values
    unnamed_empty = [column for column in table.columns
                     if str(column).startswith("Unnamed:") and (table[column].str.strip() == "").all()]
    table = table.drop(columns=unnamed_empty)
    table.attrs["delimiter"] = delimiter
    table.attrs["encoding"] = encoding
    return table


def detect_delimiter(text: str) -> str:
    """Column separator used by the header row: comma, semicolon, tab or pipe (comma if unclear).
    Semicolon and tab files are common exports from spreadsheet and point-of-sale software."""
    header = text.split("\n", 1)[0]
    counts = {candidate: header.count(candidate) for candidate in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _normalise(name: str) -> str:
    """Column name in snake_case: 'InvoiceNo', 'Invoice No' and 'invoice_no' all give 'invoice_no'."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(name).strip())  # camelCase boundary
    text = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", text)              # acronym followed by a word
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _find(columns: dict[str, str], aliases: Iterable[str]) -> str | None:
    for alias in aliases:
        if alias in columns:
            return columns[alias]
    return None


def _is_missing(values: pd.Series) -> pd.Series:
    return values.str.lower().isin(MISSING_TOKENS)


# ---------------------------------------------------------------- cleaning


def clean_table(table: pd.DataFrame, source: str = "uploaded file", is_synthetic: bool = False) -> TransactionData:
    """Detect the layout of a raw table, clean it and return validated transactions."""
    columns: dict[str, str] = {}
    for column in table.columns:
        columns.setdefault(_normalise(column), column)
    transaction_column = _find(columns, TRANSACTION_ALIASES)
    product_column = _find(columns, PRODUCT_ALIASES)
    item_columns = sorted(
        (int(match.group(1)), original)
        for normalised, original in columns.items()
        if (match := WIDE_ITEM_COLUMN.match(normalised))
    )

    if transaction_column and product_column:
        frame, removed, ignored, notes = _from_long(table, columns, transaction_column, product_column)
        layout = "long"
    elif item_columns:
        frame, removed, ignored, notes = _from_wide(table, transaction_column, [c for _, c in item_columns])
        layout = "wide"
    else:
        raise DataValidationError(
            "Could not recognise the columns. Use either (1) long format with columns transaction_id and "
            "product_name, one row per product, or (2) wide format with columns transaction_id, item_1, "
            f"item_2, ..., one row per transaction. Columns found: {', '.join(map(str, table.columns))}."
        )
    encoding = table.attrs.get("encoding", "UTF-8")
    if encoding not in ("UTF-8", "UTF-16"):
        notes.append(f"The file is not UTF-8, so it was read as {encoding} text. Check that product names with "
                     "accents or symbols look right in the preview.")
    return _finalise(frame, removed, layout, len(table), ignored, notes, source, is_synthetic,
                     delimiter=table.attrs.get("delimiter", ","), encoding=encoding)


def _from_long(table: pd.DataFrame, columns: dict[str, str], transaction_column: str,
               product_column: str) -> tuple[pd.DataFrame, tuple[list[pd.DataFrame], dict], list[str], list[str]]:
    frame = pd.DataFrame(
        {
            "file_row": np.arange(2, len(table) + 2),
            "transaction_id": table[transaction_column].astype(str),
            "product_name": table[product_column].astype(str),
        }
    )
    used = {transaction_column, product_column}
    for key, aliases in OPTIONAL_ALIASES.items():
        column = _find(columns, aliases)
        if column and column not in used:
            frame[key] = table[column].astype(str).str.strip()
            used.add(column)
    ignored = [str(column) for column in table.columns if column not in used]
    return frame, ([], {}), ignored, []


def _from_wide(table: pd.DataFrame, transaction_column: str | None,
               item_columns: list[str]) -> tuple[pd.DataFrame, tuple[list[pd.DataFrame], dict], list[str], list[str]]:
    notes = []
    if transaction_column:
        transaction_ids = table[transaction_column].astype(str)
    else:
        transaction_ids = pd.Series([f"ROW-{row}" for row in range(2, len(table) + 2)], index=table.index)
        notes.append("No transaction ID column was found, so file row numbers were used as transaction IDs.")
    items = table[item_columns].astype(str)
    no_products = _is_missing(items.stack().str.strip()).groupby(level=0).all().reindex(table.index, fill_value=True)
    missing_id = _is_missing(transaction_ids.str.strip())
    file_rows = pd.Series(np.arange(2, len(table) + 2), index=table.index)

    removed = [
        pd.DataFrame({"file_row": file_rows[missing_id], "transaction_id": transaction_ids[missing_id],
                      "product_name": "", "reason": "missing transaction ID"}),
        pd.DataFrame({"file_row": file_rows[no_products & ~missing_id],
                      "transaction_id": transaction_ids[no_products & ~missing_id],
                      "product_name": "", "reason": "transaction has no products"}),
    ]
    keep = ~(missing_id | no_products)
    long = items[keep].assign(transaction_id=transaction_ids[keep], file_row=file_rows[keep])
    long = long.melt(id_vars=["file_row", "transaction_id"], value_vars=item_columns,
                     var_name="position", value_name="product_name")
    long["position"] = long["position"].map({column: i for i, column in enumerate(item_columns)})
    long = long.sort_values(["file_row", "position"], kind="stable")
    long = long[~_is_missing(long["product_name"].str.strip())].drop(columns="position")
    repeated_ids = transaction_ids[keep].str.strip().duplicated().sum()
    if repeated_ids:
        notes.append(f"{repeated_ids:,} transaction IDs appeared on more than one row; their products were combined.")
    ignored = [str(c) for c in table.columns if c not in set(item_columns) | {transaction_column}]
    stats = {"missing_ids": int(missing_id.sum()), "empty": int((no_products & ~missing_id).sum())}
    return long.reset_index(drop=True), (removed, stats), ignored, notes


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return f"{count:,} {singular if count == 1 else (plural or singular + 's')}"


def _strip_names(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Trim IDs and names, collapse repeated spaces, and use one Unicode form (NFC) for names, so
    that 'Café' typed in two different ways is one product. Returns the number of rows changed by
    whitespace fixes."""
    raw_ids, raw_names = frame["transaction_id"], frame["product_name"]
    ids = raw_ids.str.strip()
    names = raw_names.str.strip().str.replace(r"\s+", " ", regex=True)
    whitespace_fixed = int(((ids != raw_ids) | (names != raw_names)).sum())
    return frame.assign(transaction_id=ids, product_name=names.str.normalize("NFC")), whitespace_fixed


def _reject_long_rows(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, int]]:
    """Long format: reject rows with a missing transaction ID, a blank product name or a negative
    quantity (a return or cancellation), keeping each in the removed rows with its reason."""
    ids, names = frame["transaction_id"], frame["product_name"]
    missing_id = _is_missing(ids)
    blank = _is_missing(names) & ~missing_id
    returns = pd.Series(False, index=frame.index)
    if "quantity" in frame:
        returns = (pd.to_numeric(frame["quantity"], errors="coerce") < 0) & ~missing_id & ~blank
    removed = [
        frame.loc[missing_id, REMOVED_COLUMNS[:-1]].assign(reason="missing transaction ID"),
        frame.loc[blank, REMOVED_COLUMNS[:-1]].assign(reason="blank product name"),
        frame.loc[returns, REMOVED_COLUMNS[:-1]].assign(reason="return or cancellation (negative quantity)"),
    ]
    kept = frame[~(missing_id | blank | returns)]
    counts = {
        "missing_ids": int(missing_id.sum()),
        "blank": int(blank.sum()),
        "returns": int(returns.sum()),
        "empty": int(ids[~missing_id].nunique() - kept["transaction_id"].nunique()),
    }
    return kept, removed, counts


def _merge_case_variants(frame: pd.DataFrame) -> tuple[pd.DataFrame, int, str | None]:
    """Names that differ only in capitalisation become one product. The most frequent spelling
    wins; ties prefer Title Case ("Bread" over "BREAD"), then alphabetical order."""
    lower = frame["product_name"].str.lower().rename("key")
    counts = frame.groupby([lower, frame["product_name"]]).size().rename("n").reset_index()
    counts["not_title"] = ~counts["product_name"].str.istitle()
    counts = counts.sort_values(["n", "not_title", "product_name"], ascending=[False, True, True], kind="stable")
    spelling = counts.drop_duplicates("key").set_index("key")["product_name"]
    canonical = lower.map(spelling)
    changed = canonical != frame["product_name"]
    merged = int(changed.sum())
    note = None
    if merged:
        examples = frame.loc[changed, "product_name"].unique()[:3]
        note = (f"{merged:,} product names differed only in capitalisation and were merged "
                f"(for example {', '.join(repr(e) for e in examples)}).")
    return frame.assign(product_name=canonical), merged, note


def _clean_numbers(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Quantity and unit price become numbers; missing, zero or unreadable values are left empty."""
    notes = []
    for column in ("quantity", "unit_price"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = int((values.isna() | (values <= 0)).sum())
            frame = frame.assign(**{column: values.where(values > 0)})
            if invalid:
                notes.append(f"{_plural(invalid, 'row')} {'has' if invalid == 1 else 'have'} a missing or invalid "
                             f"{column.replace('_', ' ')}; those values are left empty (they are not needed for "
                             "basket analysis).")
    return frame, notes


def _resolve_categories(frame: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Give every product one category: its most common real category, so a blank category never
    outvotes a real one. Products with no category at all become 'Uncategorised'."""
    if "category" not in frame:
        return frame, None
    known = ~_is_missing(frame["category"])
    pairs = (frame.loc[known, ["product_name", "category"]]
             .groupby(["product_name", "category"], sort=False).size().rename("n").reset_index())
    pairs = pairs.sort_values(["product_name", "n"], ascending=[True, False], kind="stable")
    main = pairs.drop_duplicates("product_name").set_index("product_name")["category"]
    conflicts = int((pairs.groupby("product_name").size() > 1).sum())
    frame = frame.assign(category=frame["product_name"].map(main).fillna("Uncategorised"))
    note = (f"{_plural(conflicts, 'product')} had more than one category; the most common category is used."
            if conflicts else None)
    return frame, note


def _merge_duplicates(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """The same product on several lines of one transaction becomes one line. Each line is treated
    as a separate scan of the product, so quantities are added; basket analysis only uses whether a
    product is in the basket, so rules are not affected."""
    keys = ["transaction_id", "product_name"]
    duplicate = frame.duplicated(keys, keep="first")
    if "quantity" in frame and duplicate.any():
        totals = frame.groupby(keys, sort=False)["quantity"].transform("sum")
        frame = frame.assign(quantity=totals.where(totals > 0))
    removed = frame.loc[duplicate, REMOVED_COLUMNS[:-1]].assign(
        reason="duplicate product in the same transaction (merged)")
    return frame[~duplicate].reset_index(drop=True), removed, int(duplicate.sum())


def _looks_generated(frame: pd.DataFrame) -> bool:
    """True for files written by BasketWise's synthetic generator: EVERY transaction ID and EVERY
    customer ID follows the generator's format (SYN-T000001, SYN-C00001). A real export would have
    to copy both formats exactly to be labelled synthetic."""
    if frame.empty or "customer_id" not in frame:
        return False
    return bool(frame["transaction_id"].str.fullmatch(GENERATED_TRANSACTION_ID).all()
                and frame["customer_id"].astype(str).str.fullmatch(GENERATED_CUSTOMER_ID).all())


def _finalise(frame: pd.DataFrame, removed_wide: tuple[list[pd.DataFrame], dict], layout: str,
              rows_received: int, ignored: list[str], notes: list[str], source: str, is_synthetic: bool,
              delimiter: str = ",", encoding: str = "UTF-8") -> TransactionData:
    """Clean the extracted rows, count every change and build the validated TransactionData."""
    removed_parts, stats = list(removed_wide[0]), dict(removed_wide[1])
    messages = list(notes)
    frame, whitespace_fixed = _strip_names(frame)
    if layout == "long":
        frame, rejected, stats = _reject_long_rows(frame)
        removed_parts.extend(rejected)
    missing_ids, empty_transactions = stats.get("missing_ids", 0), stats.get("empty", 0)
    blank_products, returns = stats.get("blank", 0), stats.get("returns", 0)
    if returns:
        messages.append(f"{_plural(returns, 'row')} had a negative quantity (returns or cancellations) and "
                        f"{'was' if returns == 1 else 'were'} removed, so returned products are not counted as "
                        "purchases.")

    frame, case_merged, case_note = _merge_case_variants(frame)
    frame, number_notes = _clean_numbers(frame)
    frame, category_note = _resolve_categories(frame)
    date_notes: list[str] = []
    if "transaction_date" in frame:
        dates, date_notes = parse_dates(frame["transaction_date"])
        frame = frame.assign(transaction_date=dates.dt.strftime("%Y-%m-%d"))
    messages += [note for note in (case_note, *number_notes, category_note, *date_notes) if note]
    frame, duplicates, n_duplicates = _merge_duplicates(frame)
    removed_parts.append(duplicates)

    if frame.empty:
        raise DataValidationError("No valid transactions remain after cleaning. Check that the transaction ID "
                                  "and product columns contain values.")
    removed = pd.concat([part for part in removed_parts if len(part)] or [pd.DataFrame(columns=REMOVED_COLUMNS)],
                        ignore_index=True)[REMOVED_COLUMNS]
    removed = removed.sort_values("file_row", kind="stable", ignore_index=True)

    n_transactions = frame["transaction_id"].nunique()
    n_products = frame["product_name"].nunique()
    rejected = missing_ids + blank_products + returns + (empty_transactions if layout == "wide" else 0)
    if rejected / rows_received > 0.05:
        messages.insert(0, f"{rejected / rows_received:.0%} of the rows were rejected or removed. "
                           "Download the removed rows to check them before relying on the results.")
    if ignored:
        messages.append(f"Columns not used by the analysis: {', '.join(ignored)}.")
    if n_transactions < 200:
        messages.append(f"Only {_plural(n_transactions, 'transaction')}. Rules from fewer than a few hundred "
                        "transactions are unreliable.")
    if n_products < 2:
        raise DataValidationError("At least two different products are needed to find products bought together.")
    sizes = frame.groupby("transaction_id", sort=False).size()
    if (sizes == 1).mean() > 0.5:
        messages.append(f"{(sizes == 1).mean():.0%} of transactions contain a single product, "
                        "so few product combinations can be found.")

    report = ValidationReport(
        source_format=layout,
        rows_received=rows_received,
        valid_rows=len(frame),
        valid_transactions=n_transactions,
        unique_products=n_products,
        missing_transaction_ids=missing_ids,
        blank_products_removed=blank_products,
        duplicate_rows_removed=n_duplicates,
        empty_transactions_removed=int(empty_transactions),
        whitespace_fixed=whitespace_fixed,
        case_variants_merged=case_merged,
        ignored_columns=tuple(ignored),
        warnings=tuple(messages),
        delimiter=delimiter,
        returns_removed=returns,
        encoding=encoding,
    )
    is_synthetic = is_synthetic or _looks_generated(frame)
    return TransactionData(frame.drop(columns="file_row"), removed, report, source, is_synthetic)


# ---------------------------------------------------------------- dates


def parse_dates(values: pd.Series) -> tuple[pd.Series, list[str]]:
    """Parse transaction dates; returns the dates (NaT where unreadable) and notes for the user.

    Accepted: yyyy-mm-dd (ISO), and day/month/year or month/day/year with '/', '-' or '.'
    separators; anything after a space or 'T' (a time) is ignored. The order of day and month is
    decided once for the whole column: if some date has a first number above 12 the column is
    day/month/year, if some date has a second number above 12 it is month/day/year, and if every
    date could be either, day/month/year is used (the usual format in India and the UK) and a note
    says so. Other text dates are left to pandas. This is the only date parser in BasketWise.
    """
    codes, uniques = pd.factorize(values.astype("string"))
    parsed_unique, notes, bad_unique = _parse_unique_dates(pd.Series(uniques, dtype="string"))
    lookup = np.append(parsed_unique.to_numpy(dtype="datetime64[ns]"), np.datetime64("NaT", "ns"))
    parsed = pd.Series(lookup[codes], index=values.index)  # code -1 (missing) maps to the final NaT
    unreadable = int(np.append(bad_unique.to_numpy(dtype=bool), False)[codes].sum())
    if unreadable:
        notes.append(f"{_plural(unreadable, 'date value')} could not be read; "
                     f"{'it is' if unreadable == 1 else 'they are'} left empty and ignored in date summaries and "
                     "charts.")
    return parsed, notes


def _parse_unique_dates(text: pd.Series) -> tuple[pd.Series, list[str], pd.Series]:
    text = text.str.strip()
    date_part = text.str.split(r"[ T]", n=1, regex=True).str[0]
    present = (date_part.notna() & ~date_part.fillna("").str.lower().isin(MISSING_TOKENS)).astype(bool)
    parts = date_part.where(present).str.extract(DATE_PARTS)
    first, second, third = (pd.to_numeric(parts[i], errors="coerce").astype(float) for i in range(3))
    numeric = parts[0].notna().astype(bool)
    year_first = (parts[0].str.len() == 4).fillna(False).astype(bool)
    day_month = numeric & ~year_first
    notes: list[str] = []
    day_first = True
    if day_month.any():
        first_big, second_big = bool((first[day_month] > 12).any()), bool((second[day_month] > 12).any())
        if second_big and not first_big:
            day_first = False
        elif not first_big and not second_big:
            notes.append("Dates such as 05/01/2025 could mean 5 January or 1 May; they were read as "
                         "day/month/year (the usual format in India and the UK).")
    year = third.where(~year_first, first)
    year = year.where(year >= 100, year + 2000)  # two-digit years: 25 -> 2025
    if day_first:
        day, month = first.where(~year_first, third), second
    else:
        day, month = second.where(~year_first, third), first.where(~year_first, second)

    def digits(values: pd.Series, width: int) -> pd.Series:
        return values.astype("Int64").astype("string").str.zfill(width)

    iso = digits(year, 4) + "-" + digits(month, 2) + "-" + digits(day, 2)
    parsed = pd.to_datetime(iso, format="%Y-%m-%d", errors="coerce")
    other = present & ~numeric
    if other.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed[other] = pd.to_datetime(text[other], errors="coerce", dayfirst=day_first, format="mixed")
    return parsed, notes, present & parsed.isna()


def transaction_dates(items: pd.DataFrame) -> pd.Series:
    """Datetimes of the cleaned transaction_date column (cleaning already normalised it to yyyy-mm-dd)."""
    return pd.to_datetime(items["transaction_date"], format="%Y-%m-%d", errors="coerce")


def load_transactions(source: str | Path | bytes, name: str | None = None,
                      is_synthetic: bool = False) -> TransactionData:
    """Read and clean a CSV from a path or uploaded bytes."""
    label = name or (Path(source).name if isinstance(source, (str, Path)) else "uploaded file")
    return clean_table(read_csv_table(source), source=label, is_synthetic=is_synthetic)


def load_demo_data(path: str | Path = DEMO_DATA_PATH) -> TransactionData:
    """The bundled synthetic dataset (regenerate it with scripts/generate_data.py)."""
    return load_transactions(path, name="Synthetic demo data", is_synthetic=True)


# ---------------------------------------------------------------- summaries and encoding


def summarize(data: TransactionData) -> dict[str, int | float | str]:
    """Headline numbers for a dataset (counts, basket sizes and, if dates exist, the date range)."""
    sizes = data.items.groupby("transaction_id", sort=False).size()
    summary = {
        "transactions": len(sizes),
        "unique products": int(data.items["product_name"].nunique()),
        "categories": int(data.items["category"].nunique()) if "category" in data.items else 0,
        "product lines": len(data.items),
        "average basket size": round(float(sizes.mean()), 2),
        "median basket size": float(sizes.median()),
        "largest basket": int(sizes.max()),
        "single-product share": round(float((sizes == 1).mean()), 4),
    }
    if "transaction_date" in data.items:
        dates = transaction_dates(data.items).dropna()
        if len(dates):
            summary["first date"] = dates.min().date().isoformat()
            summary["last date"] = dates.max().date().isoformat()
    return summary


def item_support(transactions: list[list[str]]) -> pd.Series:
    """Share of transactions that contain each product, most frequent first (ties by name)."""
    counts = pd.Series([item for basket in transactions for item in set(basket)]).value_counts()
    support = (counts / len(transactions)).rename("support").rename_axis("product")
    return support.sort_index().sort_values(ascending=False, kind="stable")


def encode(transactions: list[list[str]], sparse: bool | str = "auto") -> pd.DataFrame:
    """One-hot encode transactions into a transactions x products boolean DataFrame.

    sparse="auto" switches to a sparse matrix when the dense one would exceed
    SPARSE_CELL_THRESHOLD cells (about 10 MB of booleans).
    """
    encoder = TransactionEncoder().fit(transactions)
    if sparse == "auto":
        sparse = len(transactions) * len(encoder.columns_) > SPARSE_CELL_THRESHOLD
    if sparse:
        matrix = encoder.transform(transactions, sparse=True)
        return pd.DataFrame.sparse.from_spmatrix(matrix, columns=encoder.columns_)
    return pd.DataFrame(encoder.transform(transactions), columns=encoder.columns_)


def validate_products(products: Iterable[str], catalog: Iterable[str]) -> list[str]:
    """Return the basket's product names in catalog spelling, or raise ValueError for unknown names."""
    by_lower = {name.lower(): name for name in catalog}
    cleaned, unknown = [], []
    for product in products:
        name = re.sub(r"\s+", " ", str(product).strip())
        if not name:
            continue
        match = by_lower.get(name.lower())
        if match is None:
            unknown.append(name)
        elif match not in cleaned:
            cleaned.append(match)
    if unknown:
        hints = []
        for name in unknown:
            close = difflib.get_close_matches(name.lower(), list(by_lower), n=3, cutoff=0.6)
            suggestion = f" (did you mean: {', '.join(by_lower[c] for c in close)}?)" if close else ""
            hints.append(f"'{name}'{suggestion}")
        raise ValueError("Unknown product(s): " + "; ".join(hints))
    return cleaned
