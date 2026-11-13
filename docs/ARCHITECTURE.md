# BasketWise — Architecture

## Overview

BasketWise is a Python package (`market_basket`) with three front ends: a Streamlit app (`app.py` and `ui/`), a command line (`python -m market_basket`) and a notebook. All three call the same tested functions, so the numbers on screen, in the CLI and in reports always agree.

```text
                 ┌──────────────────────────────┐
  CSV upload ───►│ data.py                      │──► ValidationReport, removed rows
  generator  ───►│  read → detect layout →      │
 (synthetic_     │  validate → clean            │
   data.py)      └──────────────┬───────────────┘
                                │ TransactionData (cleaned items + baskets)
                                ▼
                 ┌──────────────────────────────┐
                 │ mining.py                    │  mlxtend: TransactionEncoder, apriori,
                 │  encode → frequent itemsets  │           fpgrowth, association_rules
                 │  → rules → thresholds →      │
                 │  exclusions → ranking        │──► MiningResult (itemsets, rules, config)
                 └──────────────┬───────────────┘
        ┌───────────────┬───────┴────────┬──────────────────┐
        ▼               ▼                ▼                  ▼
  recommend.py     insights.py      evaluation.py      reporting.py
  Recommender      popularity,      split, leave-one-   Markdown report,
  + fallback,      categories,      out hit rate,       ZIP of CSV files
  reasons          cross-sell,      baseline vs rules
                   bundles, groups
        └───────────────┴────────┬───────┴──────────────────┘
                                 ▼
                 plots.py (Plotly) · ui/views/* (8 pages) · cli.py · notebook
```

## Modules

| Module | Responsibility | Key functions |
|---|---|---|
| `config.py` | Paths, limits and the validated `MiningConfig` (support, confidence, lift, maximum length, algorithm, exclusions) | `MiningConfig` |
| `synthetic_data.py` | A reproducible generator with a product catalogue, shopping missions, customers, dates and validation | `generate_transactions`, `validate_synthetic` |
| `data.py` | CSV reading with limits and separator detection; long/wide detection; cleaning; the validation report; summaries; one-hot encoding | `load_transactions`, `clean_table`, `summarize`, `item_support`, `encode`, `validate_products` |
| `mining.py` | Calls mlxtend, applies thresholds and exclusions, ranks rules, explains them in words | `mine_rules`, `rank_rules`, `explain_rule` |
| `recommend.py` | Matches rules to the basket, ranks them, gives one suggestion per product with a reason, and falls back to best sellers | `Recommender`, `compare_rankings` |
| `evaluation.py` | Seeded splits, leave-one-out cases, metrics, repeated runs | `evaluate`, `train_test_split`, `score` |
| `insights.py` | Product popularity, categories, cross-sell pairs, bundles (all-confidence), merchandising groups (NetworkX communities), key insights | `bundle_opportunities`, `product_groups`, `key_insights` |
| `plots.py` | Plotly charts with consistent category colours | `product_bar`, `rules_scatter`, `rule_network`, `evaluation_chart` |
| `reporting.py` | Markdown report and ZIP export | `build_report`, `export_zip` |
| `cli.py` | Commands: summary, validate, rules, recommend, insights, evaluate, report | `main` |

## App structure

- `app.py` sets up the page, registers the navigation (`ui/navigation.py`), shows the sidebar status and runs the selected page.
- Each file in `ui/views/` has a `render()` function for one page.
- `ui/state.py` holds the per-session state: the data source, mining settings, basket and last evaluation. It also caches expensive steps with `st.cache_data`, keyed by the data source:
  - loading and validation (keyed by a hash of the uploaded file)
  - mining (keyed by source and settings)
  - best sellers and evaluation

  Moving between pages or changing an unrelated widget therefore never re-mines.
- When a user uploads a new file, it is validated first. The active dataset only changes if validation succeeds, so a bad upload never breaks the current analysis.
- `ui/components.py` provides shared page headers, metric cards, table formats, chart display and one-off messages that survive a rerun.

## Data flow for one recommendation

1. `data.load_transactions` turns the CSV into cleaned baskets.
2. `mining.mine_rules` one-hot encodes them, runs FP-Growth (via mlxtend), generates rules (via mlxtend) and applies the thresholds.
3. `Recommender` keeps rules with a single product on the right-hand side and sorts them by the chosen metric (with deterministic tie-breaks).
4. For the basket, it walks the sorted rules. It takes the first rule for each product whose left-hand side is fully in the basket, skipping products already in the basket, until it has *n* suggestions. It then fills any remaining slots with best sellers.

## Design principles

- **One source of truth.** The app, CLI, notebook and reports all call the same package functions.
- **Fail clearly.** User-facing errors are `DataValidationError` or `ValueError` messages written for non-technical users.
- **Deterministic.** Seeds are fixed, ties are broken by product name, and sets are never iterated in generators, so output is identical in every run (this is tested).
- **No silent data loss.** Every removed row is kept with its reason.
- **Honest wording.** Results are described as observations in the data, never as guaranteed outcomes.

## Performance

`scripts/benchmark.py` timings were measured on one Windows laptop with the 47-product catalogue at the default 1 % support:

| Transactions | Dense matrix | Apriori | FP-Growth |
|---|---|---|---|
| 8,000 | 0.38 MB | 0.03 s | 0.05 s |
| 40,000 | 1.88 MB | 0.12 s | 0.21 s |
| 120,000 | 5.64 MB | 0.38 s | 0.59 s |

At 0.2 % support with 40,000 transactions (1,741 itemsets), FP-Growth took 0.36 s against Apriori's 0.90 s on the dense matrix, and 0.30 s against 2.10 s on the sparse one.

The encoder switches to a sparse matrix above 10 million cells. A full 5-split evaluation of the demo data takes about 3 seconds.
