"""Read, validate, clean, summarise and encode transaction data.

Two CSV layouts are accepted (column names are matched case-insensitively, with common aliases):

* Long format: one row per product in a transaction.
  Required: transaction_id, product_name. Optional: category, quantity, unit_price,
  transaction_date, customer_id.
* Wide format: one row per transaction.
  Columns: transaction_id (optional), item_1, item_2, item_3, ...

Cleaning never drops data silently: every removed or rejected row is kept in
TransactionData.removed with a reason, and the ValidationReport counts each kind of change.
"""
from __future__ import annotations

import difflib
import io
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, replace
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from .config import DEMO_DATA_PATH, MAX_UPLOAD_MB, MAX_UPLOAD_ROWS

MISSING_TOKENS = ("", "nan", "none", "null", "n/a", "na", "-")
TRANSACTION_ALIASES = ("transaction_id", "transaction", "transaction_no", "invoice", "invoice_no", "invoice_id",
                       "order_id", "order", "basket_id", "receipt_id", "bill_no")
PRODUCT_ALIASES = ("product_name", "product", "item_name", "item", "product_description", "description")
OPTIONAL_ALIASES = {
    "product_id": ("product_id", "sku", "item_id"),
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
            ("Removed: duplicate product in a transaction", f"{self.duplicate_rows_removed:,}"),
            ("Removed: transactions with no products", f"{self.empty_transactions_removed:,}"),
            ("Fixed: extra spaces", f"{self.whitespace_fixed:,}"),
            ("Merged: capitalisation variants", f"{self.case_variants_merged:,}"),
            ("Ignored columns", ", ".join(self.ignored_columns) or "none"),
            ("Column separator", DELIMITER_NAMES.get(self.delimiter, self.delimiter)),
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


def read_csv_table(source: str | Path | bytes) -> pd.DataFrame:
    """Read a CSV file path or uploaded bytes into a DataFrame of strings, with user-facing errors."""
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")
        if path.suffix.lower() not in (".csv", ".txt"):
            raise DataValidationError(f"'{path.name}' is not a CSV file. Export the data as .csv and try again.")
        raw = path.read_bytes()
    else:
        raw = bytes(source)

    if not raw.strip():
        raise DataValidationError("The file is empty.")
    size_mb = len(raw) / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        raise DataValidationError(
            f"The file is {size_mb:.1f} MB; the limit is {MAX_UPLOAD_MB} MB. Upload a shorter period of data."
        )
    if b"\x00" in raw[:4096]:
        raise DataValidationError("This is not a text CSV file (it contains binary data). Export it as CSV first.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
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
    return table


def detect_delimiter(text: str) -> str:
    """Column separator used by the header row: comma, semicolon, tab or pipe (comma if unclear).
    Semicolon and tab files are common exports from spreadsheet and point-of-sale software."""
    header = text.split("\n", 1)[0]
    counts = {candidate: header.count(candidate) for candidate in (",", ";", "\t", "|")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


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
    data = _finalise(frame, removed, layout, len(table), ignored, notes, source, is_synthetic)
    delimiter = table.attrs.get("delimiter", ",")
    if delimiter != ",":
        data = replace(data, report=replace(data.report, delimiter=delimiter))
    return data


def _from_long(table, columns, transaction_column, product_column):
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
    return frame, [], ignored, []


def _from_wide(table, transaction_column, item_columns):
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


def _finalise(frame, removed_wide, layout, rows_received, ignored, notes, source, is_synthetic) -> TransactionData:
    removed_parts: list[pd.DataFrame] = []
    warnings = list(notes)
    if layout == "wide":
        wide_parts, stats = removed_wide
        removed_parts.extend(wide_parts)
        missing_ids, empty_transactions = stats["missing_ids"], stats["empty"]
    else:
        missing_ids = empty_transactions = 0

    raw_ids, raw_names = frame["transaction_id"], frame["product_name"]
    ids = raw_ids.str.strip()
    names = raw_names.str.strip().str.replace(r"\s+", " ", regex=True)
    whitespace_fixed = int(((ids != raw_ids) | (names != raw_names)).sum())
    frame = frame.assign(transaction_id=ids, product_name=names)

    blank_products = 0
    if layout == "long":
        missing_id = _is_missing(ids)
        blank = _is_missing(names) & ~missing_id
        missing_ids, blank_products = int(missing_id.sum()), int(blank.sum())
        removed_parts.append(frame.loc[missing_id, REMOVED_COLUMNS[:-1]].assign(reason="missing transaction ID"))
        removed_parts.append(frame.loc[blank, REMOVED_COLUMNS[:-1]].assign(reason="blank product name"))
        valid_ids = set(ids[~missing_id])
        frame = frame[~(missing_id | blank)]
        empty_transactions = len(valid_ids) - frame["transaction_id"].nunique()

    lower = frame["product_name"].str.lower().rename("key")
    counts = frame.groupby([lower, frame["product_name"]]).size().rename("n").reset_index()
    # most frequent spelling wins; ties prefer Title Case ("Bread" over "BREAD"), then alphabetical order
    counts["not_title"] = ~counts["product_name"].str.istitle()
    counts = counts.sort_values(["n", "not_title", "product_name"], ascending=[False, True, True], kind="stable")
    spelling = counts.drop_duplicates("key").set_index("key")["product_name"]
    canonical = lower.map(spelling)
    case_merged = int((canonical != frame["product_name"]).sum())
    if case_merged:
        examples = frame.loc[canonical != frame["product_name"], "product_name"].unique()[:3]
        warnings.append(f"{case_merged:,} product names differed only in capitalisation and were merged "
                        f"(for example {', '.join(repr(e) for e in examples)}).")
    frame = frame.assign(product_name=canonical)

    for column in ("quantity", "unit_price"):
        if column in frame:
            values = pd.to_numeric(frame[column], errors="coerce")
            invalid = int((values.isna() | (values <= 0)).sum())
            frame[column] = values.where(values > 0)
            if invalid:
                warnings.append(f"{invalid:,} rows have a missing or invalid {column.replace('_', ' ')}; "
                                "those values are left empty (they are not needed for basket analysis).")
    if "category" in frame:
        frame["category"] = frame["category"].where(~_is_missing(frame["category"]), "Uncategorised")
        conflicts = frame.groupby("product_name")["category"].nunique()
        if (conflicts > 1).any():
            warnings.append(f"{int((conflicts > 1).sum())} products had more than one category; "
                            "the most common category is used.")
            main = frame.groupby("product_name")["category"].agg(lambda s: s.value_counts().index[0])
            frame["category"] = frame["product_name"].map(main)

    keys = ["transaction_id", "product_name"]
    duplicate = frame.duplicated(keys, keep="first")
    if "quantity" in frame and duplicate.any():
        totals = frame.groupby(keys, sort=False)["quantity"].transform("sum")
        frame["quantity"] = totals.where(totals > 0)
    removed_parts.append(frame.loc[duplicate, REMOVED_COLUMNS[:-1]].assign(
        reason="duplicate product in the same transaction (merged)"))
    frame = frame[~duplicate].reset_index(drop=True)

    if frame.empty:
        raise DataValidationError("No valid transactions remain after cleaning. Check that the transaction ID "
                                  "and product columns contain values.")
    removed = pd.concat([part for part in removed_parts if len(part)] or [pd.DataFrame(columns=REMOVED_COLUMNS)],
                        ignore_index=True)[REMOVED_COLUMNS]
    removed = removed.sort_values("file_row", kind="stable", ignore_index=True)

    n_transactions = frame["transaction_id"].nunique()
    n_products = frame["product_name"].nunique()
    rejected_share = (missing_ids + blank_products + (empty_transactions if layout == "wide" else 0)) / rows_received
    if rejected_share > 0.05:
        warnings.insert(0, f"{rejected_share:.0%} of the rows were rejected or removed. "
                           "Download the removed rows to check them before relying on the results.")
    if ignored:
        warnings.append(f"Columns not used by the analysis: {', '.join(ignored)}.")
    if n_transactions < 200:
        warnings.append(f"Only {n_transactions:,} transaction{'s' if n_transactions != 1 else ''}. "
                        "Rules from fewer than a few hundred "
                        "transactions are unreliable.")
    if n_products < 2:
        raise DataValidationError("At least two different products are needed to find products bought together.")
    sizes = frame.groupby("transaction_id", sort=False).size()
    if (sizes == 1).mean() > 0.5:
        warnings.append(f"{(sizes == 1).mean():.0%} of transactions contain a single product, "
                        "so few product combinations can be found.")

    report = ValidationReport(
        source_format=layout,
        rows_received=rows_received,
        valid_rows=len(frame),
        valid_transactions=n_transactions,
        unique_products=n_products,
        missing_transaction_ids=missing_ids,
        blank_products_removed=blank_products,
        duplicate_rows_removed=int(duplicate.sum()),
        empty_transactions_removed=int(empty_transactions),
        whitespace_fixed=whitespace_fixed,
        case_variants_merged=case_merged,
        ignored_columns=tuple(ignored),
        warnings=tuple(warnings),
    )
    return TransactionData(frame.drop(columns="file_row"), removed, report, source, is_synthetic)


def load_transactions(source: str | Path | bytes, name: str | None = None, is_synthetic: bool = False) -> TransactionData:
    """Read and clean a CSV from a path or uploaded bytes."""
    label = name or (Path(source).name if isinstance(source, (str, Path)) else "uploaded file")
    return clean_table(read_csv_table(source), source=label, is_synthetic=is_synthetic)


def load_demo_data(path: str | Path = DEMO_DATA_PATH) -> TransactionData:
    """The bundled synthetic dataset (regenerate it with scripts/generate_data.py)."""
    return load_transactions(path, name="Synthetic demo data", is_synthetic=True)


# ---------------------------------------------------------------- summaries and encoding


def summarize(data: TransactionData) -> dict[str, float]:
    sizes = data.items.groupby("transaction_id", sort=False).size()
    summary = {
        "transactions": int(len(sizes)),
        "unique products": int(data.items["product_name"].nunique()),
        "categories": int(data.items["category"].nunique()) if "category" in data.items else 0,
        "product lines": int(len(data.items)),
        "average basket size": round(float(sizes.mean()), 2),
        "median basket size": float(sizes.median()),
        "largest basket": int(sizes.max()),
        "single-product share": round(float((sizes == 1).mean()), 4),
    }
    if "transaction_date" in data.items:
        dates = pd.to_datetime(data.items["transaction_date"], errors="coerce").dropna()
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
