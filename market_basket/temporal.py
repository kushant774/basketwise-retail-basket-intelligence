"""Chronological (time-based) evaluation: learn from earlier invoices, test on later ones.

Protocol
--------
1. Every invoice has one timestamp. A cutoff (midnight) splits the invoices: those before it form
   the TRAINING period and those at or after it form the TEST period. An invoice is never split,
   and invoices with identical timestamps always fall on the same side.
2. By default the cutoff is the start of the day that contains the first invoice of the latest
   `test_share` of invoices (counted by invoice, never by score). A fixed date can be given instead.
3. If several threshold settings are given, they are compared on a VALIDATION period: the latest
   `validation_share` of the training invoices, split off in the same way, with rules and best
   sellers learned from the earlier training invoices only. The setting with the highest Hit Rate@K
   of the association-rule recommender is chosen. The test period is not used for this.
4. With the chosen setting, rules and best sellers are learned from the whole training period.
5. Leave-one-out on the test period: in every test invoice with two or more products, each product is
   hidden once and the other products are the query. Every method is scored on exactly these cases,
   including cases whose hidden product never appeared in training (no method can find those).

This answers "do associations learned from the past help complete future baskets?". It does not
measure next-purchase prediction, extra sales or any causal effect.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import MiningConfig
from .data import item_support
from .evaluation import METHODS, leave_one_out_cases, methods_from_rules
from .mining import mine_rules, rank_rules
from .recommend import Recommender

BASELINE = METHODS[0]          # best sellers from the training period
PRIMARY = METHODS[3]           # the app's default: rules ranked by confidence, best sellers fill free slots
RULES_ONLY = METHODS[1]

Progress = Callable[[str], None]


def _quiet(_: str) -> None:
    pass


@dataclass(frozen=True)
class TemporalResult:
    settings: dict             # every setting used, as plain values
    split: dict                # cutoff and the date range and size of each period
    validation: dict           # the validation period and the baseline score there ({} if no selection)
    selection: pd.DataFrame    # one row per candidate setting (empty if only one setting was given)
    config: MiningConfig       # the setting used on the test period
    n_itemsets: int
    rules: pd.DataFrame        # rules learned from the training period, ranked by confidence
    best_sellers: pd.Series    # training-period share of invoices per product
    summary: pd.DataFrame      # one row per method
    intervals: dict            # bootstrap confidence intervals (resampling test invoices)
    weekly: pd.DataFrame       # hit rate per test week
    cases: pd.DataFrame = field(repr=False)  # one row per leave-one-out case


def candidate_settings(supports: Iterable[float], confidences: Iterable[float], lifts: Iterable[float],
                       max_len: int | None = 2, algorithm: str = "fpgrowth",
                       exclude: tuple[str, ...] = ()) -> list[MiningConfig]:
    """Every combination of the given thresholds (duplicates removed), each validated by MiningConfig."""
    values = {"minimum support": sorted(set(supports)), "minimum confidence": sorted(set(confidences)),
              "minimum lift": sorted(set(lifts))}
    for name, given in values.items():
        if not given:
            raise ValueError(f"Give at least one value for the {name}.")
    return [MiningConfig(support, confidence, lift, max_len, algorithm, exclude)
            for support in values["minimum support"] for confidence in values["minimum confidence"]
            for lift in values["minimum lift"]]


def _ordered(invoices: pd.DataFrame) -> pd.DataFrame:
    if invoices["time"].isna().any():
        raise ValueError("Every invoice needs a valid date and time for a chronological split.")
    return invoices.sort_values(["time", "invoice"], kind="stable", ignore_index=True)


def cutoff_for_share(times: pd.Series, share: float) -> pd.Timestamp:
    """Midnight at the start of the day that contains the first invoice of the latest `share` of invoices."""
    if isinstance(share, bool) or not 0 < share < 1:
        raise ValueError(f"The share of invoices held out must be between 0 and 1 (exclusive), got {share}.")
    ordered = np.sort(pd.to_datetime(times).to_numpy())
    if len(ordered) < 2:
        raise ValueError("At least two invoices are needed to split by time.")
    first_later = pd.Timestamp(ordered[min(len(ordered) - 1, math.floor(len(ordered) * (1 - share)))])
    cutoff = first_later.normalize()
    if cutoff <= pd.Timestamp(ordered[0]):  # the whole first day would be held out: start one day later
        cutoff = pd.Timestamp(ordered[0]).normalize() + pd.Timedelta(days=1)
    return cutoff


def split_by_time(invoices: pd.DataFrame, cutoff: str | pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(invoices before the cutoff, invoices at or after it), each in time order."""
    invoices = _ordered(invoices)
    cutoff = pd.Timestamp(cutoff)
    before = invoices["time"] < cutoff
    train, test = invoices[before].reset_index(drop=True), invoices[~before].reset_index(drop=True)
    if train.empty or test.empty:
        first, last = invoices["time"].min(), invoices["time"].max()
        raise ValueError(f"The cutoff {cutoff:%Y-%m-%d} leaves no {'training' if train.empty else 'test'} invoices; "
                         f"the invoices run from {first:%Y-%m-%d} to {last:%Y-%m-%d}.")
    return train, test


def _baskets(invoices: pd.DataFrame) -> list[list[str]]:
    return [list(items) for items in invoices["items"]]


def _period(invoices: pd.DataFrame) -> dict:
    return {"first_invoice": invoices["time"].min().isoformat(), "last_invoice": invoices["time"].max().isoformat(),
            "invoices": len(invoices), "invoices_with_two_or_more_products": int((invoices["n_products"] >= 2).sum())}


def score_cases(methods: dict[str, Recommender], invoices: pd.DataFrame, top_k: int,
                training_products: frozenset[str], exclude: Iterable[str] = (),
                progress: Progress = _quiet) -> tuple[pd.DataFrame, dict[str, set[str]]]:
    """Leave-one-out cases for every invoice with 2+ products, scored by every method on the same cases.

    Returns one row per case (the rank of the hidden product in each method's top_k, 0 if absent,
    and how many products each method suggested) and the set of products each method recommended.
    """
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"The number of recommendations (K) must be a whole number of at least 1, got {top_k!r}.")
    names = list(methods)
    exclude = tuple(exclude)
    recommended: dict[str, set[str]] = {name: set() for name in names}
    records = []
    evaluable = invoices[invoices["n_products"] >= 2]
    total = len(evaluable)
    for position, row in enumerate(evaluable.itertuples(index=False), start=1):
        restricted = [methods[name].for_basket(row.items) for name in names]
        for query, hidden in leave_one_out_cases([list(row.items)], exclude):
            ranks, shown = [], []
            for name, recommender in zip(names, restricted, strict=True):
                products = [rec.product for rec in recommender.recommend(query, top_n=top_k)]
                recommended[name].update(products)
                ranks.append(products.index(hidden) + 1 if hidden in products else 0)
                shown.append(len(products))
            records.append((row.invoice, row.time, row.customer_id, hidden, hidden in training_products,
                            *ranks, *shown))
        if position % 500 == 0 or position == total:
            progress(f"Scored {position:,} of {total:,} invoices")
    columns = (["invoice", "time", "customer_id", "hidden_product", "hidden_product_in_training"]
               + [f"rank: {name}" for name in names] + [f"shown: {name}" for name in names])
    cases = pd.DataFrame.from_records(records, columns=columns)
    if cases.empty:
        raise ValueError("There is no invoice with two or more products to evaluate in this period "
                         "(excluded products are not evaluated).")
    return cases, recommended


def summarise(cases: pd.DataFrame, names: list[str], top_k: int, universe_size: int,
              recommended: dict[str, set[str]]) -> pd.DataFrame:
    """Metrics per method; every method is scored on exactly the same cases."""
    known = cases["hidden_product_in_training"].astype(bool)
    customers = cases.loc[cases["customer_id"] != "", "customer_id"].nunique()
    rows = []
    for name in names:
        hit = cases[f"rank: {name}"] > 0
        rows.append({
            "method": name,
            "cases": len(cases),
            "invoices": cases["invoice"].nunique(),
            "customers": customers,
            f"hit rate@{top_k}": hit.mean(),
            f"precision@{top_k}": (hit / top_k).mean(),   # each case has one relevant product, so at most 1/K
            f"recall@{top_k}": hit.mean(),                # the one hidden product is found or not
            "availability": (cases[f"shown: {name}"] > 0).mean(),
            "coverage": len(recommended[name]) / universe_size if universe_size else float("nan"),
            f"invoice-averaged hit rate@{top_k}": hit.groupby(cases["invoice"], sort=False).mean().mean(),
            f"hit rate@{top_k}, hidden product seen in training": hit[known].mean() if known.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def bootstrap_intervals(cases: pd.DataFrame, names: list[str], reference: str, n_boot: int = 1000,
                        seed: int = 42, level: float = 0.95) -> dict:
    """Percentile intervals for Hit Rate@K and for each method's difference from `reference`.

    Cases from one invoice are not independent, so whole test invoices are resampled with
    replacement (the same resampled invoices for every method, so differences are paired)."""
    if n_boot <= 0:
        return {}
    codes, invoices = pd.factorize(cases["invoice"])
    n_cases = np.bincount(codes).astype(float)
    hits = {name: np.bincount(codes, weights=(cases[f"rank: {name}"] > 0).to_numpy(dtype=float),
                              minlength=len(invoices)) for name in names}
    rng = np.random.default_rng(seed)
    samples = {name: np.empty(n_boot) for name in names}
    for start in range(0, n_boot, 100):  # in chunks, so memory stays small for any number of invoices
        size = min(100, n_boot - start)
        index = rng.integers(0, len(invoices), size=(size, len(invoices)))
        denominator = n_cases[index].sum(axis=1)
        for name in names:
            samples[name][start:start + size] = hits[name][index].sum(axis=1) / denominator
    low, high = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    result = {"resamples": n_boot, "level": level, "seed": seed, "unit": "test invoice",
              "hit_rate": {name: [float(np.percentile(samples[name], low)), float(np.percentile(samples[name], high))]
                           for name in names},
              "difference_vs_baseline": {}}
    for name in names:
        if name != reference:
            difference = samples[name] - samples[reference]
            result["difference_vs_baseline"][name] = {
                "interval": [float(np.percentile(difference, low)), float(np.percentile(difference, high))],
                "share_of_resamples_above_baseline": float((difference > 0).mean()),
            }
    return result


def weekly_results(cases: pd.DataFrame, names: list[str], top_k: int) -> pd.DataFrame:
    """Hit Rate@K per calendar week (Monday to Sunday) of the test period."""
    week = cases["time"].dt.to_period("W-SUN").dt.start_time.rename("week starting")
    grouped = cases.groupby(week)
    table = pd.DataFrame({"invoices": grouped["invoice"].nunique(), "cases": grouped.size()})
    for name in names:
        table[f"hit rate@{top_k}: {name}"] = (cases[f"rank: {name}"] > 0).groupby(week).mean()
    return table.reset_index()


def select_setting(train: pd.DataFrame, candidates: list[MiningConfig], validation_share: float, top_k: int,
                   progress: Progress = _quiet) -> tuple[MiningConfig, pd.DataFrame, dict]:
    """Choose the thresholds on a validation period cut from the END of the training period.

    Rules and best sellers come from the earlier training invoices only, and the test period is
    never seen. Highest Hit Rate@K of the association-rule recommender wins; ties go to the higher
    minimum support, then confidence, then lift (fewer, stronger rules). The returned table ranks
    every compared setting ("validation rank", 1 = chosen).
    """
    cutoff = cutoff_for_share(train["time"], validation_share)
    try:
        inner, validation = split_by_time(train, cutoff)
    except ValueError as error:
        raise ValueError(f"The training period ({train['time'].min():%Y-%m-%d} to {train['time'].max():%Y-%m-%d}) "
                         "is too short to set aside a validation period for choosing thresholds. Give a single "
                         "value for each threshold, or use a longer training period.") from error
    if not (validation["n_products"] >= 2).any():
        raise ValueError("The validation period (the end of the training period) has no invoice with two or more "
                         "products, so thresholds cannot be compared. Give a single value for each threshold or a "
                         "larger validation share.")
    baskets = _baskets(inner)
    known = frozenset(item for basket in baskets for item in basket)
    best_sellers = item_support(baskets)
    k = f"hit rate@{top_k}"
    rows, baseline = [], None
    for number, config in enumerate(candidates, start=1):
        progress(f"Validation {number} of {len(candidates)}: support {config.min_support:.2%}, "
                 f"confidence {config.min_confidence:.0%}, lift {config.min_lift:g}")
        row = {"min_support": config.min_support, "min_confidence": config.min_confidence,
               "min_lift": config.min_lift, "max_len": config.max_len}
        try:
            mined = mine_rules(baskets, config)
        except ValueError as error:
            rows.append({**row, "status": f"refused: {str(error).rstrip('.')}"})
            continue
        methods = methods_from_rules(mined.rules, best_sellers, config.exclude)
        chosen = {PRIMARY: methods[PRIMARY]} if baseline is not None else \
            {PRIMARY: methods[PRIMARY], BASELINE: methods[BASELINE]}
        cases, recommended = score_cases(chosen, validation, top_k, known, config.exclude)
        scores = summarise(cases, list(chosen), top_k, len(known - set(config.exclude)), recommended)
        scores = scores.set_index("method")
        if baseline is None:
            baseline = {"cases": int(scores.loc[BASELINE, "cases"]), k: float(scores.loc[BASELINE, k])}
        rows.append({**row, "status": "compared", "itemsets": len(mined.frequent_itemsets), "rules": len(mined.rules),
                     f"validation {k}": float(scores.loc[PRIMARY, k]),
                     "validation availability": float(scores.loc[PRIMARY, "availability"]),
                     "validation coverage": float(scores.loc[PRIMARY, "coverage"])})
    table = pd.DataFrame(rows)
    compared = table[table["status"] == "compared"] if "status" in table else table.iloc[0:0]
    if compared.empty:
        reasons = sorted(set(status.removeprefix("refused: ") for status in table["status"]))
        raise ValueError("None of the candidate settings could be mined on the early part of the training period: "
                         + "; ".join(reasons) + ".")
    order = compared.assign(_score=compared[f"validation {k}"].round(10)).sort_values(
        ["_score", "min_support", "min_confidence", "min_lift"], ascending=False, kind="stable")
    table["validation rank"] = pd.Series(range(1, len(order) + 1), index=order.index)
    best = order.index[0]
    table["chosen"] = table.index == best
    chosen = candidates[best]
    details = {"cutoff": cutoff.isoformat(), "training_part": _period(inner), "validation_part": _period(validation),
               "cases": baseline["cases"], f"baseline {k}": baseline[k],
               "rule": f"latest {validation_share:.0%} of training invoices, from the start of that day"}
    return chosen, table, details


def evaluate_chronological(invoices: pd.DataFrame, candidates: list[MiningConfig], cutoff: str | None = None,
                           test_share: float = 0.2, validation_share: float = 0.2, top_k: int = 3,
                           n_bootstrap: int = 1000, seed: int = 42,
                           progress: Progress = _quiet) -> TemporalResult:
    """Run the whole protocol described in the module docstring.

    invoices: one row per invoice with columns invoice, time, customer_id, n_products and items
    (a tuple of product identifiers), as made by uci.preprocess.
    """
    if not candidates:
        raise ValueError("Give at least one threshold setting.")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError(f"The number of recommendations (K) must be a whole number of at least 1, got {top_k!r}.")
    invoices = _ordered(invoices)
    if cutoff is None:
        cut = cutoff_for_share(invoices["time"], test_share)
        rule = f"start of the day containing the first invoice of the latest {test_share:.0%} of invoices"
    else:
        cut = pd.Timestamp(cutoff)
        rule = "fixed date given by the user"
    train, test = split_by_time(invoices, cut)
    if len(candidates) > 1:
        config, selection, validation = select_setting(train, candidates, validation_share, top_k, progress)
    else:
        config, selection, validation = candidates[0], pd.DataFrame(), {}

    progress("Mining rules on the whole training period")
    baskets = _baskets(train)
    if selection.empty:
        mined = mine_rules(baskets, config)
    else:
        # The safety limits depend on the data, so a setting accepted on the early part of the training
        # period can be refused on all of it. Then the next-best validated setting is used, and the table says so.
        ranked = selection.dropna(subset=["validation rank"]).sort_values("validation rank").index
        for position, index in enumerate(ranked):
            try:
                mined = mine_rules(baskets, candidates[index])
                config = candidates[index]
                break
            except ValueError as error:
                selection.loc[index, "status"] = f"refused on the whole training period: {str(error).rstrip('.')}"
                if position == len(ranked) - 1:
                    raise ValueError("Every validated setting was refused on the whole training period: "
                                     f"{str(error).rstrip('.')}.") from error
        selection["chosen"] = [candidates[i] == config for i in selection.index]
    best_sellers = item_support(baskets)
    methods = methods_from_rules(mined.rules, best_sellers, config.exclude)
    known = frozenset(item for basket in baskets for item in basket)
    universe = known - set(config.exclude)

    progress("Scoring the test period")
    cases, recommended = score_cases(methods, test, top_k, known, config.exclude, progress)
    names = list(methods)
    summary = summarise(cases, names, top_k, len(universe), recommended)
    intervals = bootstrap_intervals(cases, names, BASELINE, n_bootstrap, seed)
    split = {"cutoff": cut.isoformat(), "rule": rule, "training": _period(train), "test": _period(test),
             "test_share_of_invoices": len(test) / len(invoices)}
    settings = {"top_k": top_k, "test_share": None if cutoff is not None else test_share,
                "fixed_cutoff": None if cutoff is None else cut.isoformat(),
                "validation_share": validation_share if len(candidates) > 1 else None,
                "candidate_settings": len(candidates), "bootstrap_resamples": n_bootstrap, "seed": seed,
                "candidate_universe": len(universe), "excluded_products": list(config.exclude)}
    return TemporalResult(settings, split, validation, selection, config, len(mined.frequent_itemsets),
                          rank_rules(mined.rules, "confidence"), best_sellers, summary, intervals,
                          weekly_results(cases, names, top_k), cases)
