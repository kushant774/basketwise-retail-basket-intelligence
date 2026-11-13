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
