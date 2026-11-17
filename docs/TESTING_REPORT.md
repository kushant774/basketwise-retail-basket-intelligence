# BasketWise — Testing report

Last run on 18 November 2026 on Windows 11 with Python 3.13.14 and the versions pinned in `requirements.txt`, and by GitHub Actions on Ubuntu (see [Continuous integration](#multi-cutoff-check-and-continuous-integration-november-2026)).

## Automated tests

Run with `python -m pytest`. Result: **270 passed**, with 0 failures and 0 skips (about 85 s).
- The multi-cutoff check added 23 tests to the 247 of the real-world validation.
- Before that, the real-world validation added 72 to the 175 of the previous release, and the release audit added 61 to the earlier 114.
- No earlier test was removed, skipped or loosened.

On 18 November 2026, before any change for the multi-cutoff check, the 247 earlier tests were rerun on the feature branch: 247 passed.

| File | Tests | What is covered |
|---|---|---|
| `test_synthetic_data.py` | 14 | Same seed gives identical data, different seeds differ; both bundled CSV files equal the generator output; configurable sizes; invalid settings; validation catches broken data; basket-size variety; planted associations are recovered and unrelated products are not linked; every injected data problem is reported with its exact count |
| `test_data.py` | 51 | Long and wide formats; aliases; whitespace; missing IDs, blanks and placeholders (including `#N/A` and `?`), with removal reasons; duplicates with summed quantities; capitalisation merging; ignored columns; invalid files; non-CSV and missing files; size and row limits; latin-1 files; separators; empty spreadsheet columns; unusual product names through the whole pipeline; a 30,000-transaction upload; **returns and cancellations removed (negative quantities)**; **dates: day/month/year, ISO, month/day/year, dotted with two-digit years, UCI style with times; ambiguous and unreadable dates reported**; **the weekly chart uses the same parser**; **UCI Online Retail and other CamelCase headers**; **Windows-1252 text (’ – €) and UTF-16 and `.tsv` files**; **binary data beyond the first kilobytes**; **Unicode normalisation**; **blank categories never outvote real ones**; **generated files recognised as synthetic wherever they are loaded, and real-looking data never** |
| `test_mining.py` | 31 | Metrics against hand calculations; Apriori = FP-Growth; a plain-Python recount of every rule; thresholds; exclusions; empty results; the itemset guard; invalid settings; deterministic ranking; explanations; **ties in lift caused by floating-point noise (A → B vs B → A) broken by confidence, while real differences are kept**; **the walkthrough's Pasta rule comes first**; **the itemset estimate is an upper bound**; **dense data is refused before mining starts**; **no setting the app offers is refused on the demo data**; **the candidate-rule limit**; **NaN, infinite and text thresholds refused**; **product names containing commas stay unambiguous** |
| `test_recommend.py` | 14 | Only applicable rules are used; basket products are never recommended or duplicated; ranking options; labelled fallback; empty and unknown baskets; invalid arguments; explanations; display tables; **ignored products never recommended, including as best sellers, for every ranking and in the comparison table**; **several exclusions and an empty candidate set**; **a basket given as one string** |
| `test_evaluation.py` | 10 | Reproducible, complete and disjoint splits; leave-one-out cases; a hand-checked metric case; leakage guards; a reproducible summary; small datasets; a clear error when nothing can be evaluated; **ignored products are never recommended or hidden by any method, and every method is scored on the same cases** |
| `test_insights.py` | 8 | Popularity and categories; cross-sell pairs listed once; bundle metrics; merchandising groups; insights; every chart renders; **bundles and insights leave out ignored products**; **a pair bought together less often than chance (lift 0.83) is never offered as a bundle, and the minimum lift applies** |
| `test_reporting.py` | 5 | Report sections and honest wording; empty results; ZIP contents; Markdown escaping |
| `test_cli.py` | 18 | Every CLI command and clean error messages; **`recommend --exclude` and `insights --exclude` never show the ignored product**; **NaN or infinite numbers, negative counts and mining options on `summary` are usage errors**; **`--removed-output` always writes a file**; **the demo upload keeps its synthetic disclaimer**; **a folder given as data**; **too-large settings refused with a message** |
| `test_app.py` | 26 | Streamlit interface through `streamlit.testing`: each page on demo, generated and tiny data, uploads, rule search, recommendations, evaluation, downloads. **New cross-page tests run several pages in one session with the sidebar, as in the real app:** three Applies in a row each take effect and the sidebar matches the page; a second Apply keeps the first one's settings; the basket survives visits to other pages; ignored products never appear in suggestions, bundles or the Overview insights; a refused upload's error stays visible; "Use demo data" on demo data resets nothing; the evaluation shows the settings it used; a search with no match says so |

**One existing assertion was changed.** `test_rank_rules_orders_and_validates` checked that ranked metric values are strictly in decreasing order. Rule ranking now treats values equal to 10 decimal places as ties (see `docs/ALGORITHM_EXPLANATION.md`), so the check compares the values at that tolerance. The old exact check would have required the floating-point noise this release removes.

### Do the new tests catch the old bugs?

The 48 new regression tests were also run against the previous release (commit `a901aaa`). **43 failed there, as they should.** The other 5 cover behaviour that already worked (for example ISO dates and spaced headers). Two examples of failures on the old code:
- After applying 5 % support, the old sidebar still showed "Support ≥ 1.0% … 262 association rules".
- After visiting another page, the old basket came back empty.

## Release audit: problems found and fixed

| ID | Problem | Fix | Regression tests |
|---|---|---|---|
| C1 | A second "Apply settings" in a row was ignored, and the sidebar showed the previous settings | The page reruns at once after Apply, rebuilding the form from the new settings | `test_app.py` (three Applies in a row; browser check) |
| C2 | Ignored products still appeared as best-seller suggestions, in bundles, in the Overview and in the evaluation baseline | Exclusions apply to the recommender (rules and fallback), bundles, the CLI and every evaluation method | `test_recommend.py`, `test_insights.py`, `test_evaluation.py`, `test_cli.py`, `test_app.py` |
| I1 | The basket was lost when moving to another page | The basket lives in plain session state; the widget copies its value there | `test_app.py`; browser check |
| I2 | Bundles were described as "more often than chance" but were not checked against chance | Only sets with lift above 1 and at least the minimum lift are offered | `test_insights.py` |
| I3 | Returns and cancellations counted as purchases | Negative-quantity rows are removed with their own reason | `test_data.py` |
| I4 | 05/01/2025 was read as 1 May | One shared date parser detects the day/month order per file and warns when it is ambiguous; the chart uses it too | `test_data.py` |
| I5 | `InvoiceNo` and `Description` were refused | CamelCase headers are normalised before alias matching | `test_data.py` |
| I6 | Windows-1252 files were silently garbled | UTF-8, UTF-16, Windows-1252 and then Latin-1 are tried in turn, and the encoding is reported | `test_data.py` |
| I7 | The demo upload was not labelled synthetic | Generated files are recognised by their ID format | `test_data.py`, `test_cli.py`, `test_app.py` |
| I8 | Several documentation statements were inaccurate | Corrected (three methods plus a baseline, average case counts, planted structure, selection effect, benchmark settings, caching scope) | Checked against the code and reruns |
| I9 | Float noise decided the order of equivalent rules | Metrics are compared at 10 decimal places | `test_mining.py` |
| I11 | The itemset limit was checked only after mining | A guaranteed upper bound is checked before mining, and candidate rules before rule generation | `test_mining.py`, `test_cli.py` |
| I12 | Interface tests ran one page at a time | Cross-page tests added | `test_app.py` |

Smaller fixes in the same release:
- empty-state messages on Smart Recommendations, and an empty state for rule searches with no match
- product names escaped in Markdown and HTML
- readable algorithm names and consistent arrows
- a spinner while generating data
- the evaluation settings shown with the result
- an accurate stale-evaluation caption
- CLI output in UTF-8 on Windows
- generator arguments checked before running
- three unused constants removed
- `_finalise` split into smaller functions
- `requires-python` declared, indirect dependencies pinned, CSV line endings fixed with `.gitattributes`

## Real-world validation (November 2026)

The real-world validation was developed on the branch `feature/uci-real-world-validation`. These tests were added; none needs the real dataset or network access, because they use hand-made UCI-style rows and the synthetic generator's baskets written in the UCI layout.

| File | Tests | What is covered |
|---|---|---|
| `test_uci.py` | 29 | **Reading:**<br>- CSV and `.xlsx` give identical text (real dates, whole numbers, empty customer IDs, £).<br>- Empty, header-only, foreign, missing and folder paths give clear errors.<br><br>**Preprocessing:**<br>- Cancellations (also with a positive quantity), returns, stock adjustments, bad-debt invoices, postage and other service codes, numeric voucher codes and zero or negative prices are never purchases.<br>- The first reason wins, with the CSV line number.<br>- Missing customer IDs are kept and counted, or dropped with the option.<br>- Missing descriptions keep the row and take the code's other name.<br>- Exact duplicates are removed and repeated lines merged (quantities added).<br>- Invalid, non-ISO and time-zoned dates are removed.<br>- Invalid prices are removed.<br>- Unicode, whitespace and lower-case codes are normalised without merging products that share a description.<br>- Empty tables and tables with no purchases are refused.<br>- The accounting identities hold.<br>- Repeated runs are reproducible.<br><br>**Quality report:** every measured count, and the time span inside invoices.<br><br>**Download (with a local fake server):**<br>- The CSV, the manifest and the checksums are written.<br>- The ZIP is reused, and `--force` downloads it again.<br>- A dropped connection is retried.<br>- An unexpected archive or a broken ZIP is refused.<br>- A changed spreadsheet gives a warning.<br>- A ZIP placed by hand is used. |
| `test_temporal.py` | 19 | **Split:**<br>- The cutoff is a midnight chosen by invoice count.<br>- No date leaks across the cutoff, and identical timestamps stay together.<br>- Impossible splits give clear errors.<br><br>**Leakage:**<br>- Mining and popularity receive exactly the early-training invoices during selection, and the training invoices afterwards; this is recorded with a monkeypatch.<br>- Best sellers come from the training period.<br>- Products first sold in the test period are misses for every method.<br><br>**Cases and metrics:**<br>- A hand-checked leave-one-out example, including empty-recommendation cases.<br>- Precision@K = Hit Rate@K / K, and Recall@K = Hit Rate@K.<br>- Every method is scored on the same cases.<br>- `Recommender.for_basket` gives exactly the same recommendations as the full recommender on 300 demo baskets.<br><br>**Selection and errors:**<br>- No evaluable test invoice gives a clear error, and invalid settings are refused.<br>- Selection is deterministic, including ties.<br>- Settings refused by the safety limit are reported.<br>- Validation-period errors name the validation period.<br>- A setting refused on the whole training period falls back to the next best.<br><br>**Reproducibility and intervals:**<br>- Results are reproducible.<br>- The bootstrap intervals are paired and contain the estimate. |
| `test_real_world.py` | 22 | **CLI:**<br>- `uci-prepare` and `uci-evaluate` write every file with consistent numbers, and no customer IDs in exports.<br>- Two runs are identical apart from the generation time.<br>- A fixed cutoff and single thresholds skip selection.<br>- Invalid options are usage errors, invalid settings give clear errors, and a missing dataset explains the download.<br><br>**Report and exports:**<br>- The ZIP export holds every file.<br>- The report has all ten sections, the attribution and the honest wording.<br>- The "fixed protocol" claim is made only for the official file with every default (including seed, bootstrap and the customer-ID option).<br>- The limitations match the run.<br><br>**Synthetic benchmark:** still 0.1971 / 0.4348 (Hit Rate@3), and 265 itemsets / 262 rules. |
| `test_app.py` (2 new) | 2 | **Evaluation Lab:**<br>- It has two tabs, and the random-split benchmark still runs.<br>- A missing dataset shows the download command.<br><br>**Real-world tab:**<br>- Nothing runs before a click.<br>- It loads, shows the quality summary, runs, shows the comparison and the limitations, and offers three downloads.<br>- A fixed start date without a date gives a clear message. |

**Independent review.** Before the real test period was scored, a separate review agent audited the new code. It did not see the author's conclusions. It was told not to compute any metric on the real test period.
- **What it confirmed:**
  - no leakage, the same cases for every method, and correct metrics and bootstrap;
  - `for_basket` is exactly equivalent to the full recommender;
  - the preprocessing accounting holds on the real file;
  - the synthetic benchmark is unchanged.
- **What it found:** eight smaller defects, all fixed with regression tests above.
  - The "fixed protocol" sentence ignored the seed, the bootstrap count, the customer-ID option and the file.
  - Two limitation sentences were wrong for some runs.
  - Validation errors named the test period.
  - A chosen setting could be refused on the full training period.
  - A few report sentences were hard-coded.
  - Time-zoned dates crashed the app tab.
  - One £100 gift voucher had a numeric code and was kept as merchandise.

**Runs on the real data** (official file, SHA-256 of the spreadsheet `43465a06…`):
- `scripts/download_uci_online_retail.py` downloaded 541,909 rows; its retry logic recovered from one dropped connection.
- `uci-evaluate` with the defaults scored the real test period for the first time after the protocol, the threshold grid and K were fixed.
- After the review fixes, only report wording and formatting changed. That changed no number, and the outputs were regenerated.
- Two further runs gave byte-identical CSV and gzip files, and the same JSON and Markdown apart from the generation time.
- Each run took 77–82 s with a peak of 674–681 MB of memory.

**Browser check.** A development script (not part of the repository) drove headless Chrome through its DevTools protocol against `streamlit run app.py`.
- The Evaluation Lab loaded with both tabs, and the random-split benchmark still ran.
- The real-world tab loaded the dataset, ran the default evaluation in 72 s and showed no error text.
- The three downloads (JSON, Markdown, ZIP) arrived.
- The downloaded JSON and Markdown equal the CLI outputs apart from the generation time.
- `docs/images/real_world_validation.png` was captured in this run.

**Static and dependency checks.**
- `ruff` (run from a separate environment, line length 120): no pyflakes errors, and no new findings. The same 18 style notices exist on `main`.
- `pip check`: no broken requirements.
- `pip-audit -r requirements.txt`: no known vulnerabilities.

**Fresh clone.** The feature branch was cloned into an empty folder and checked step by step:
1. A new Python 3.13 environment was created, and `pip install -r requirements.txt` ran in it; `pip check` was clean.
2. `python -m pytest`: 247 passed.
3. `evaluate --splits 5` and `rules` reproduced the synthetic benchmark (Hit Rate@3 0.1971 / 0.4348; 265 itemsets, 262 rules).
4. `scripts/download_uci_online_retail.py` downloaded the dataset again from UCI. The ZIP, the spreadsheet and the converted CSV had the same SHA-256 as the first download.
5. `uci-prepare` and `uci-evaluate` reproduced `docs/real_world/`. Every CSV and the scored cases are identical, and the JSON and Markdown differ only in the generation and download times.

## Multi-cutoff check and continuous integration (November 2026)

These tests were added on the same branch. Like the other real-world tests, they need no download.

| File | Tests | What is covered |
|---|---|---|
| `test_rolling.py` | 22 | **Folds:**<br>- Monthly cutoffs start after the training months and stop at the last invoice; too few months is a clear error.<br>- Every later invoice is tested exactly once, every fold trains on all earlier invoices, and no fold receives an invoice from after its test month (recorded with a monkeypatch).<br>- A fold gives exactly the same result as the single-split protocol run on its own window.<br><br>**Summaries:**<br>- Pooled results weight every case once, and coverage is not pooled across different candidate universes.<br>- The early group holds only folds that end before the single-split cutoff.<br>- Verdicts follow the interval; given cutoffs are sorted, and errors name the fold.<br><br>**Command and report:**<br>- `uci-rolling` writes every file with consistent numbers and no customer IDs.<br>- Two runs are identical apart from the generation time.<br>- The report has all seven sections, the attribution and honest wording.<br>- Each command keeps its own output folder.<br>- Invalid options are usage errors.<br>- The "fixed protocol" claim needs the official file and every default.<br>- The limitations match the run: refused settings, grid edges, a short last fold, overlap with the single split. |
| `test_app.py` (1 new) | 1 | The real-world tab shows the published multi-cutoff results without running anything, even before the dataset is downloaded, and hides the panel when the file is missing. |

**Protocol order.**
1. The protocol and its summaries were added to `docs/PRODUCT_DECISIONS.md` (decision 25) before the code was written.
2. The code was tested on synthetic UCI-style data only.
3. `uci-rolling` was then run on the real file.
4. While that first run was in progress, two report sentences were reworded, before any of its results had been seen. After it finished, two limitation notes were added: settings refused by the safety limit (folds 1 to 3), and the chosen support at the edge of the grid (folds 4 to 10). No setting or number changed.
5. The run was repeated with the final code, and its output is what `docs/real_world/rolling/` holds.

**Runs on the real data** (official file, spreadsheet SHA-256 `43465a06…`):
- The two runs took 8 min 38 s and 8 min 18 s.
- Their `rolling_folds.csv`, `rolling_pooled_summary.csv`, `rolling_threshold_selection.csv` and `rolling_cases.csv.gz` are byte-identical.
- Their JSON files differ only in the generation time and the limitation notes added in between.

**Continuous integration.** `.github/workflows/tests.yml` runs on every pull request and every push to `main`. It installs `requirements.txt` on Ubuntu with Python 3.13, runs `pip check`, and then runs `python -m pytest`.

## Static checks

`ruff` (run from a separate environment, line length 120):
- **No pyflakes errors**: no unused imports and no undefined names.
- **Multi-cutoff check (ruff 0.17.5):** the new files have no findings. Each changed file has exactly as many findings as on the branch before the change.
- The remaining notices are style preferences of ruff's newer default rules, such as `dict()` calls and implicit string concatenation. They were deliberately left alone, along with long lines in older files and the complexity of `generate_transactions`. Restructuring the generator could change its random stream and therefore the published datasets.

## Browser verification

A script drove headless Chrome through its DevTools protocol against `streamlit run app.py`. The script was a development aid and is not part of the repository. Every check passed:
- All eight pages load without error text.
- Smart Recommendations: the "Milk + Bread" example shows Butter first.
- Evaluation Lab: "Run evaluation" produces the chart and its interpretation.
- After navigating with the sidebar, **Download report (Markdown)** and **Download all results (ZIP)** both produced files. The report included the evaluation, and the ZIP held 10 files.
- An invalid upload through the real file input was rejected, and the demo data stayed active.
- Uploading `Data/demo_upload_with_issues.csv` loaded 2,997 transactions, showed the capitalisation warning, and labelled the data "Synthetic data: demo_upload_with_issues.csv".
- Recommendations worked on the uploaded data. Generated data became active, and Product Analytics, Association Rules and Bundle Opportunities worked on it.
- **C1 in a real browser:** support was changed with the keyboard and applied twice in a row (2.0 %, then 1.5 %). Both took effect, and the sidebar showed the same rule count as the page.
- **I1 in a real browser:** the Milk + Bread basket was still there, with Butter first, after visiting Bundle Opportunities through the sidebar.

The screenshots in `docs/images/` were captured in this run.

## Clean-environment check

These steps were run on a fresh copy of exactly the files Git would publish (65 files; no virtual environment, caches or outputs):

1. `py -3.13 -m venv .venv` and `pip install -r requirements.txt`. `pip check` reported no broken requirements.
2. `python -m pytest`: 175 passed.
3. Both datasets were regenerated with the documented commands. Their Git blob hashes equal the committed files (`f0f1b3d…` and `d04867a…`), so they are byte-identical. `.gitattributes` now keeps CSV files at LF line endings on Windows checkouts too.
4. Every CLI command exited with 0: `summary`, `validate`, `rules`, `recommend` (also with `--exclude`), `insights` (also with `--exclude`), `evaluate`, `report` and `scripts/benchmark.py`. `--min-lift nan` and `generate_data.py --products 5` gave clear usage errors with exit code 2.
5. `evaluate --splits 5` reproduced the published numbers exactly, and the walkthrough notebook executed without errors.
6. `streamlit run app.py` started, answered its health check and served the page.

## Not tested

- Operating systems other than Windows, and Python versions other than 3.13 (the dependencies need 3.12 or newer).
- Uploads close to the 50 MB limit in a real browser. The size and row limits are unit-tested with lowered limits, and a 30,000-transaction upload is tested.
- Real retail exports other than the UCI Online Retail dataset. The full UCI dataset is now run (see above), but it is one retailer's data, and no other real export was available.
- Accessibility (keyboard-only use, screen readers), mobile layouts, and several users at the same time.
- Long-running use: cached results stay in server memory for the lifetime of the process.
