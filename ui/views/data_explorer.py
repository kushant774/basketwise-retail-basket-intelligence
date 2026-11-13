"""Data Explorer: upload or generate data, review validation results and preview the cleaned data."""
from __future__ import annotations

import streamlit as st

from market_basket.config import MAX_UPLOAD_MB
from market_basket.data import DataValidationError
from market_basket.synthetic_data import CATALOG
from ui import components as ui
from ui import state

LONG_TEMPLATE = ("transaction_id,product_name,category,quantity\n"
                 "T1001,Milk,Dairy,1\nT1001,Bread,Bakery,2\nT1002,Tea,Beverages,1\nT1002,Biscuits,Bakery,1\n")
WIDE_TEMPLATE = "transaction_id,item_1,item_2,item_3\nT1001,Milk,Bread,Butter\nT1002,Tea,Biscuits,\n"


def _handle_upload() -> None:
    file = st.file_uploader(
        "Transaction CSV", type=["csv"], key="uploader",
        help=f"Long format (one row per product) or wide format (one row per transaction), up to {MAX_UPLOAD_MB} MB.",
    )
    if file is None or file.file_id == st.session_state.last_upload_id:
        return
    st.session_state.last_upload_id = file.file_id
    try:
        loaded = state.use_upload(file.name, file.getvalue())
    except DataValidationError as error:
        st.error(f"This file could not be used: {error} The previous dataset is still active.",
                 icon=":material/error:")
        return
    ui.flash(f"Loaded {file.name}: {loaded.report.valid_transactions:,} valid transactions and "
             f"{loaded.report.unique_products:,} products. Review the validation summary below.")
    st.rerun()


def render() -> None:
    ui.page_header("Data Explorer", "Load transaction data, check its quality and preview the cleaned result.")
    upload_column, generate_column = st.columns(2, gap="large")
    with upload_column.container(border=True):
        st.markdown('<div class="bw-card-title">Upload a CSV file</div>', unsafe_allow_html=True)
        _handle_upload()
        with st.expander("Accepted formats and templates", icon=":material/description:"):
            st.markdown(
                "- **Long format**: one row per product in a transaction. Required columns: `transaction_id`, "
                "`product_name`. Optional: `category`, `quantity`, `unit_price`, `transaction_date`, `customer_id`.\n"
                "- **Wide format**: one row per transaction. Columns `transaction_id`, `item_1`, `item_2`, ...\n"
                "- Common alternative names such as `invoice_no`, `order_id`, `item` or `product` are recognised.\n"
                "- Blank products, missing transaction IDs and duplicate rows are removed and listed for download."
            )
            left, right = st.columns(2)
            left.download_button("Long-format template", LONG_TEMPLATE, "basketwise_long_template.csv", "text/csv",
                                 icon=":material/download:")
            right.download_button("Wide-format template", WIDE_TEMPLATE, "basketwise_wide_template.csv", "text/csv",
                                  icon=":material/download:")
    with generate_column.container(border=True):
        st.markdown('<div class="bw-card-title">Or generate synthetic sample data</div>', unsafe_allow_html=True)
        st.caption("Creates realistic-looking but entirely synthetic transactions with a fixed random seed.")
        a, b, c = st.columns(3)
        n_transactions = a.number_input("Transactions", 200, 50_000, 8_000, step=500)
        n_products = b.number_input("Products", 8, len(CATALOG), len(CATALOG))
        seed = c.number_input("Random seed", 0, 99_999, 42)
        left, right = st.columns(2)
        if left.button("Generate sample data", type="primary", icon=":material/auto_awesome:"):
            loaded = state.use_generated(int(n_transactions), int(n_products), int(seed))
            ui.flash(f"Generated {loaded.report.valid_transactions:,} synthetic transactions.")
            st.rerun()
        if right.button("Use demo data", icon=":material/restart_alt:"):
            state.use_demo()
            ui.flash("Switched back to the bundled synthetic demo data.")
            st.rerun()

    data = state.data()
    report = data.report
    st.divider()
    ui.data_badge(data)
    st.subheader("Validation summary")
    removed = len(data.removed)
    ui.metric_row([
        ("Rows received", f"{report.rows_received:,}", "Rows in the file (excluding the header)."),
        ("Valid transactions", f"{report.valid_transactions:,}", None),
        ("Unique products", f"{report.unique_products:,}", None),
        ("Rows removed or rejected", f"{removed:,}", "Download them below to see each reason."),
    ])
    for warning in report.warnings:
        st.warning(warning, icon=":material/warning:")
    if not report.warnings:
        st.success("No data-quality problems were found.", icon=":material/check_circle:")

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.dataframe(report.summary_table(), hide_index=True, width="stretch")
    with right:
        st.markdown("**Removed and rejected rows**")
        if removed:
            st.dataframe(data.removed.head(500), hide_index=True, width="stretch",
                         column_config=ui.column_config(data.removed),
                         height=min(420, 35 * (min(removed, 500) + 1) + 3))
            st.download_button("Download removed rows (CSV)", data.removed.to_csv(index=False), "removed_rows.csv",
                               "text/csv", icon=":material/download:")
        else:
            st.caption("Nothing was removed.")

    st.subheader("Cleaned data preview")
    st.caption(f"First 500 of {len(data.items):,} product lines. Names are trimmed and capitalisation variants merged.")
    st.dataframe(data.items.head(500), hide_index=True, width="stretch", column_config=ui.column_config(data.items),
                 height=min(320, 35 * (min(len(data.items), 500) + 1) + 3))
    st.download_button("Download cleaned data (CSV)", data.items.to_csv(index=False), "cleaned_transactions.csv",
                       "text/csv", icon=":material/download:")

    st.subheader("Transaction summary")
    baskets = data.items.groupby("transaction_id", sort=False)["product_name"].agg(list)
    summary = baskets.to_frame("products").assign(**{"number of products": baskets.map(len)})
    summary["products"] = summary["products"].map(", ".join)
    st.dataframe(summary.head(500).reset_index(), hide_index=True, width="stretch",
                 column_config=ui.column_config(summary.reset_index()),
                 height=min(320, 35 * (min(len(summary), 500) + 1) + 3),
                 column_order=["transaction_id", "number of products", "products"])
