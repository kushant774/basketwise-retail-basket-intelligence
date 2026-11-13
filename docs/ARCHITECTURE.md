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
| `config.py` | Paths, limits, display names and the validated `MiningConfig` (support, confidence, lift, maximum length, algorithm, ignored products; non-finite numbers are refused) | `MiningConfig` |
| `synthetic_data.py` | A reproducible generator with a product catalogue, shopping missions, customers, dates and validation | `generate_transactions`, `validate_synthetic` |
| `data.py` | CSV reading with size limits, encoding and separator detection; long/wide detection; header normalisation (including CamelCase); cleaning (returns, duplicates, capitalisation, Unicode, categories, dates); recognition of generated data; the validation report; summaries; one-hot encoding | `load_transactions`, `clean_table`, `parse_dates`, `summarize`, `item_support`, `encode`, `validate_products` |
| `mining.py` | Checks an upper bound on the itemset count before mining, calls mlxtend, checks the candidate-rule count, applies thresholds and ignored products, ranks rules (with a 10-decimal tolerance), explains them in words | `estimate_itemsets`, `mine_rules`, `rank_rules`, `explain_rule` |
| `recommend.py` | Matches rules to the basket, ranks them, gives one suggestion per product with a reason, and falls back to best sellers; ignored products are never suggested | `Recommender`, `compare_rankings` |
| `evaluation.py` | Seeded splits, leave-one-out cases, metrics, repeated runs; ignored products treated the same way by every method | `evaluate`, `train_test_split`, `score` |
| `insights.py` | Product popularity, categories, cross-sell pairs, bundles (lift filter, all-confidence ranking), merchandising groups (NetworkX communities), key insights | `bundle_opportunities`, `product_groups`, `key_insights` |
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

  Moving between pages or changing an unrelated widget therefore never re-mines. The cache lives in the memory of the server process: it is shared by every browser session of the running app (an identical upload is recognised by its content hash) and is cleared when the app restarts. Nothing is written to disk.
- **Widget values versus session state.** Streamlit deletes a widget's value when its page is not shown. Anything that must survive a page change (the basket, the settings, the evaluation) is therefore kept in plain session-state keys, and widgets copy their values into them through callbacks. This is why the basket is still there after visiting another page.
- **Applying settings.** The settings form takes its default values from the current settings. After **Apply settings**, the page reruns at once, so the form is rebuilt from the new settings before the next change and the sidebar (drawn before the page) shows the new rule count. Without this rerun, a second Apply in a row was ignored.
- When a user uploads a new file, it is validated first. The active dataset only changes if validation succeeds, so a bad upload never breaks the current analysis, and the error stays visible until the file is removed.
- `ui/components.py` provides shared page headers, metric cards, table formats, chart display, escaping of product names shown in Markdown or HTML, and one-off messages that survive a rerun.

## Data flow for one recommendation

1. `data.load_transactions` turns the CSV into cleaned baskets.
2. `mining.mine_rules` one-hot encodes them, runs FP-Growth (via mlxtend), generates rules (via mlxtend) and applies the thresholds.
3. `Recommender` keeps rules with a single product on the right-hand side and sorts them by the chosen metric (with deterministic tie-breaks).
4. For the basket, it walks the sorted rules. It takes the first rule for each product whose left-hand side is fully in the basket, skipping products already in the basket, until it has *n* suggestions. It then fills any remaining slots with best sellers.

## Design principles

- **One source of truth.** The app, CLI, notebook and reports all call the same package functions.
- **Fail clearly.** User-facing errors are `DataValidationError` or `ValueError` messages written for non-technical users.
- **Deterministic.** Seeds are fixed, metrics are compared at 10 decimal places, ties are broken by product name, and sets are never iterated in generators, so output is identical in every run (this is tested).
- **No silent data loss.** Every removed row is kept with its reason.
- **Safe limits.** Settings that could produce more than 200,000 frequent itemsets are refused before mining starts (using a guaranteed upper bound), and more than 500,000 candidate rules before rule generation. Each refusal explains what to change.
- **Honest wording.** Results are described as observations in the data, never as guaranteed outcomes.

## Performance

`scripts/benchmark.py` timings, measured on one Windows laptop on 14 November 2026 with the 47-product catalogue and the app's default settings (minimum support 1 %, at most 3 products per itemset). Timings vary between computers.

| Transactions | Dense matrix | Apriori (dense) | FP-Growth (dense) | Apriori (sparse) | FP-Growth (sparse) |
|---|---|---|---|---|---|
| 8,000 | 0.38 MB | 0.03 s | 0.06 s | 0.07 s | 0.08 s |
| 40,000 | 1.88 MB | 0.12 s | 0.28 s | 0.30 s | 0.25 s |
| 120,000 | 5.64 MB | 0.40 s | 0.88 s | 0.91 s | 0.87 s |

At 0.2 % support with no size limit and 40,000 transactions (1,741 itemsets), FP-Growth took 0.37 s against Apriori's 0.88 s on the dense matrix, and 0.31 s against 2.13 s on the sparse one.

The encoder switches to a sparse matrix above 10 million cells. A full 5-split evaluation of the demo data takes a few seconds.
