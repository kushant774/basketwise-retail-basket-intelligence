"""Small reusable pieces of the interface: page headers, metric cards, notes and table formats."""
from __future__ import annotations

import streamlit as st

from market_basket.data import TransactionData

CSS = """
<style>
  .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1280px;}
  h1 {font-size: 1.85rem !important; font-weight: 700 !important; letter-spacing: -0.01em;}
  h2, h3 {letter-spacing: -0.005em;}
  [data-testid="stMetricValue"] {font-size: 1.6rem;}
  [data-testid="stMetricLabel"] p {font-size: 0.85rem; color: #5b5f66;}
  .bw-subtitle {color: #5b5f66; font-size: 1rem; margin-top: -0.6rem; margin-bottom: 0.8rem;}
  .bw-card-title {font-weight: 600; font-size: 1.05rem; margin-bottom: 0.15rem;}
  .bw-muted {color: #5b5f66; font-size: 0.9rem;}
  .bw-chip {display: inline-block; padding: 0.15rem 0.6rem; margin: 0 0.35rem 0.35rem 0; border-radius: 999px;
            background: #eef4fd; color: #184f95; font-size: 0.85rem; border: 1px solid #cde2fb;}
</style>
"""

PERCENT = st.column_config.NumberColumn(format="percent")
DECIMAL = st.column_config.NumberColumn(format="%.2f")
INTEGER = st.column_config.NumberColumn(format="localized")

RULE_COLUMNS = {
    "antecedents": st.column_config.TextColumn("If the basket has", width="medium"),
    "consequents": st.column_config.TextColumn("It often also has", width="medium"),
    "antecedent support": st.column_config.NumberColumn("Antecedent support", format="percent"),
    "consequent support": st.column_config.NumberColumn("Consequent support", format="percent"),
    "support": st.column_config.NumberColumn("Support", format="percent",
                                             help="Share of all transactions containing every product in the rule."),
    "confidence": st.column_config.NumberColumn("Confidence", format="percent",
                                                help="Share of baskets with the first products that also had the last."),
    "lift": st.column_config.NumberColumn("Lift", format="%.2f",
                                          help="How many times more often than chance. Above 1 = positive association."),
    "leverage": st.column_config.NumberColumn("Leverage", format="%.4f"),
    "conviction": st.column_config.NumberColumn("Conviction", format="%.2f"),
}

METRIC_HELP = {
    "support": "Support: the share of all transactions that contain the products. "
               "support(A) = transactions containing A / all transactions.",
    "confidence": "Confidence: of the baskets with A, the share that also contained B. "
                  "confidence(A → B) = support(A and B) / support(A).",
    "lift": "Lift: how many times more often B appears with A than it appears overall. "
            "lift(A → B) = confidence(A → B) / support(B). Above 1 means a positive association in the "
            "observed data, not proof that A causes B.",
}


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str) -> None:
    st.title(title)
    st.markdown(f'<div class="bw-subtitle">{subtitle}</div>', unsafe_allow_html=True)


def data_badge(data: TransactionData) -> None:
    if data.is_synthetic:
        st.badge(data.source, icon=":material/science:", color="blue")
    else:
        st.badge(f"Uploaded file: {data.source}", icon=":material/upload_file:", color="green")


def metric_row(metrics: list[tuple[str, str, str | None]]) -> None:
    for column, (label, value, help_text) in zip(st.columns(len(metrics)), metrics):
        column.metric(label, value, help=help_text, border=True)


def chips(names: list[str]) -> None:
    st.markdown("".join(f'<span class="bw-chip">{name}</span>' for name in names), unsafe_allow_html=True)


def explain_metrics() -> None:
    with st.expander("What do support, confidence and lift mean?", icon=":material/help:"):
        for text in METRIC_HELP.values():
            st.markdown(f"- {text}")
        st.markdown("- Example: if 40% of baskets with milk and bread also contain butter, the confidence of "
                    "*milk, bread → butter* is 40%. If butter is in 10% of all baskets, the lift is 4.")


def pct(value: float, digits: int = 0) -> str:
    return f"{value:.{digits}%}"


def chart(figure) -> None:
    """Show a Plotly figure with BasketWise styling (Streamlit's own chart theme is not applied)."""
    st.plotly_chart(figure, width="stretch", theme=None, config={"displaylogo": False})


def flash(message: str, kind: str = "success") -> None:
    """Queue a message to show after the next rerun (for example after switching datasets)."""
    st.session_state["flash"] = (kind, message)


def show_flash() -> None:
    kind, message = st.session_state.pop("flash", (None, None))
    if message:
        {"success": st.success, "info": st.info, "warning": st.warning}[kind](message)


LABELS = {
    "hit rate@3": "Hit Rate@3", "hit rate@5": "Hit Rate@5",
    "hit rate@3 std": "Hit Rate@3 std", "hit rate@5 std": "Hit Rate@5 std",
    "splits beating baseline@3": "Splits beating baseline (@3)",
    "splits beating baseline@5": "Splits beating baseline (@5)",
    "transaction_id": "Transaction ID", "product_name": "Product", "product_id": "Product ID",
    "customer_id": "Customer ID", "transaction_date": "Date", "unit_price": "Unit price", "file_row": "File row",
}
PERCENT_COLUMNS = {"support", "confidence", "share of transactions", "all-confidence", "hit rate@3", "hit rate@5",
                   "availability", "coverage", "antecedent support", "consequent support"}
DECIMAL_COLUMNS = {"lift", "lift vs independence", "conviction", "unit_price"}
SPREAD_COLUMNS = {"hit rate@3 std", "hit rate@5 std"}
COUNT_COLUMNS = {"transactions", "transactions together", "products", "units", "cases", "size", "rank", "split",
                 "number of products", "quantity", "file_row"}


def column_config(frame) -> dict:
    """Consistent headings and number formats for every business-facing table."""
    config = {}
    for column in frame.columns:
        label = LABELS.get(column, str(column)[:1].upper() + str(column)[1:])
        if column in PERCENT_COLUMNS:
            config[column] = st.column_config.NumberColumn(label, format="percent")
        elif column in DECIMAL_COLUMNS:
            config[column] = st.column_config.NumberColumn(label, format="%.2f")
        elif column in SPREAD_COLUMNS:
            config[column] = st.column_config.NumberColumn(label, format="%.4f")
        elif column in COUNT_COLUMNS:
            config[column] = st.column_config.NumberColumn(label, format="localized")
        else:
            config[column] = st.column_config.Column(label)
    return config
