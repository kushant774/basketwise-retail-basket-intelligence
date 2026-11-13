"""Session state and cached computations shared by every page.

Each browser session has its own data source, mining settings, basket and evaluation result
(st.session_state). These are plain session-state keys, not widget keys: Streamlit deletes a
widget's value when its page is not shown, so widgets copy their values into these keys.

Expensive steps are cached with st.cache_data and keyed by the data source, so moving between
pages or changing an unrelated widget does not repeat the mining. The cache lives in the memory
of the server process: it is shared by every session of the running app (an identical upload is
recognised by its content hash) and is cleared when the app restarts. Nothing is written to disk.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import pandas as pd
import streamlit as st

from market_basket.config import MiningConfig
from market_basket.data import TransactionData, item_support, load_demo_data, load_transactions
from market_basket.evaluation import EvaluationResult, evaluate
from market_basket.mining import MiningResult, mine_rules
from market_basket.synthetic_data import generate_transactions

DEMO_SOURCE = {"key": "demo", "payload": None, "name": None, "synthetic": True}


def init() -> None:
    st.session_state.setdefault("source", dict(DEMO_SOURCE))
    st.session_state.setdefault("config", MiningConfig())
    st.session_state.setdefault("basket", [])
    st.session_state.setdefault("evaluation", None)
    st.session_state.setdefault("last_upload_id", None)


@st.cache_data(show_spinner="Loading and validating the data...", max_entries=8)
def _load(key: str, payload: bytes | None, name: str | None, synthetic: bool) -> TransactionData:
    if payload is None:
        return load_demo_data()
    return load_transactions(payload, name=name, is_synthetic=synthetic)


def data() -> TransactionData:
    source = st.session_state.source
    return _load(source["key"], source["payload"], source["name"], source["synthetic"])


def source_key() -> str:
    return st.session_state.source["key"]


def _switch(source: dict) -> None:
    st.session_state.source = source
    st.session_state.basket = []
    st.session_state.evaluation = None
    st.session_state.config = replace(st.session_state.config, exclude=())


def use_upload(name: str, payload: bytes) -> TransactionData:
    """Validate an uploaded file and make it the active dataset. Raises DataValidationError if invalid,
    in which case the previous dataset stays active."""
    key = "upload-" + hashlib.sha256(payload).hexdigest()[:16]
    loaded = _load(key, payload, name, False)
    _switch({"key": key, "payload": payload, "name": name, "synthetic": False})
    return loaded


def use_generated(n_transactions: int, n_products: int, seed: int) -> TransactionData:
    frame = generate_transactions(n_transactions=n_transactions, n_products=n_products, seed=seed)
    payload = frame.to_csv(index=False).encode("utf-8")
    name = f"Generated synthetic data ({n_transactions:,} transactions, {n_products} products, seed {seed})"
    key = f"generated-{n_transactions}-{n_products}-{seed}"
    loaded = _load(key, payload, name, True)
    _switch({"key": key, "payload": payload, "name": name, "synthetic": True})
    return loaded


def use_demo() -> bool:
    """Make the bundled demo data active. Returns False (and changes nothing) if it already is."""
    if source_key() == DEMO_SOURCE["key"]:
        return False
    _switch(dict(DEMO_SOURCE))
    return True


def basket() -> list[str]:
    """The current basket, limited to products in the active dataset."""
    catalog = set(data().catalog)
    return [product for product in st.session_state.basket if product in catalog]


def set_basket(products: list[str]) -> None:
    st.session_state.basket = list(products)


def config() -> MiningConfig:
    return st.session_state.config


def set_config(new_config: MiningConfig) -> None:
    st.session_state.config = new_config


@st.cache_data(show_spinner="Finding frequent itemsets and association rules...", max_entries=32)
def _mine(key: str, mining_config: MiningConfig, _data: TransactionData) -> MiningResult:
    return mine_rules(_data.transactions, mining_config)


def mining() -> tuple[MiningResult | None, str | None]:
    """Mining result for the active data and settings, or (None, user-facing error message)."""
    try:
        return _mine(source_key(), config(), data()), None
    except ValueError as error:
        return None, str(error)


@st.cache_data(show_spinner=False, max_entries=8)
def _best_sellers(key: str, _data: TransactionData) -> pd.Series:
    return item_support(_data.transactions)


def best_sellers() -> pd.Series:
    return _best_sellers(source_key(), data())


@st.cache_data(show_spinner=False, max_entries=8)
def _evaluate(key: str, mining_config: MiningConfig, n_splits: int, test_size: float, seed: int,
              _data: TransactionData) -> EvaluationResult:
    return evaluate(_data.transactions, mining_config, n_splits=n_splits, test_size=test_size, seed=seed)


def run_evaluation(n_splits: int, test_size: float, seed: int) -> EvaluationResult:
    result = _evaluate(source_key(), config(), n_splits, test_size, seed, data())
    st.session_state.evaluation = {"source": source_key(), "config": config(), "result": result}
    return result


def latest_evaluation() -> tuple[EvaluationResult | None, bool]:
    """Most recent evaluation and whether it is out of date (data or settings changed since)."""
    stored = st.session_state.evaluation
    if not stored:
        return None, False
    stale = stored["source"] != source_key() or stored["config"] != config()
    return stored["result"], stale
