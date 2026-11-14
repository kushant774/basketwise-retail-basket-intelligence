import pandas as pd
import pytest

from market_basket.data import clean_table, load_demo_data

# Textbook example (Tan, Steinbach & Kumar, "Introduction to Data Mining"): five baskets.
# Counts: Bread 4, Milk 4, Diapers 4, Beer 3, Cola 2, Eggs 1.
TOY_BASKETS = [
    ["Bread", "Milk"],
    ["Bread", "Diapers", "Beer", "Eggs"],
    ["Milk", "Diapers", "Beer", "Cola"],
    ["Bread", "Milk", "Diapers", "Beer"],
    ["Bread", "Milk", "Diapers", "Cola"],
]


def baskets_to_data(baskets, categories=None):
    """Build TransactionData from basket lists through the real cleaning pipeline."""
    rows = [(f"T{i + 1}", product) for i, basket in enumerate(baskets) for product in basket]
    table = pd.DataFrame(rows, columns=["transaction_id", "product_name"])
    if categories:
        table["category"] = table["product_name"].map(categories)
    return clean_table(table, source="test data")


@pytest.fixture
def toy():
    return [list(basket) for basket in TOY_BASKETS]


@pytest.fixture
def toy_data():
    return baskets_to_data(TOY_BASKETS, {"Bread": "Bakery", "Milk": "Dairy", "Diapers": "Baby",
                                         "Beer": "Drinks", "Cola": "Drinks", "Eggs": "Dairy"})


@pytest.fixture(scope="session")
def demo():
    return load_demo_data()


UCI_COLUMNS = ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate", "UnitPrice", "CustomerID", "Country"]


def uci_rows(*rows) -> pd.DataFrame:
    """A raw UCI-style table (all text) from (InvoiceNo, StockCode, Description, Quantity, InvoiceDate,
    UnitPrice, CustomerID, Country) tuples."""
    return pd.DataFrame([[str(value) for value in row] for row in rows], columns=UCI_COLUMNS)


def uci_style_table(n_transactions: int = 1500, seed: int = 3) -> pd.DataFrame:
    """The synthetic generator's baskets (planted associations, about six months of dates) written in the
    UCI Online Retail layout, so the real-data pipeline can be tested without the real file."""
    from market_basket.synthetic_data import generate_transactions

    frame = generate_transactions(n_transactions=n_transactions, seed=seed)
    number = frame["transaction_id"].str[-6:].astype(int)
    minutes = number % 600
    times = pd.to_datetime(frame["transaction_date"]) + pd.to_timedelta(8 * 60 + minutes, unit="min")
    return pd.DataFrame({
        "InvoiceNo": (500000 + number).astype(str),
        "StockCode": (10000 + frame["product_id"].str[1:].astype(int)).astype(str),
        "Description": frame["product_name"].str.upper(),
        "Quantity": frame["quantity"].astype(str),
        "InvoiceDate": times.dt.strftime("%Y-%m-%d %H:%M:%S"),
        "UnitPrice": frame["unit_price"].map(lambda v: f"{v:g}"),
        "CustomerID": (12000 + frame["customer_id"].str[-5:].astype(int)).astype(str),
        "Country": "United Kingdom",
    })


@pytest.fixture(scope="session")
def uci_csv(tmp_path_factory):
    """A UCI-style CSV with realistic baskets plus a few cancellations, returns and service lines."""
    table = uci_style_table()
    extra = uci_rows(
        ("C599999", "10001", "MILK", "-1", "2025-03-01 10:00:00", "50", "12001", "United Kingdom"),
        ("599998", "10002", "damaged", "-5", "2025-03-01 11:00:00", "0", "", "United Kingdom"),
        ("599997", "POST", "POSTAGE", "1", "2025-03-01 12:00:00", "18", "12002", "France"),
    )
    path = tmp_path_factory.mktemp("uci") / "online_retail.csv"
    pd.concat([table, extra], ignore_index=True).to_csv(path, index=False)
    return path
