# BasketWise — Testing report

Last run on 2026-11-14 on Windows 11 with Python 3.13.14 and the versions pinned in `requirements.txt`.

## Automated tests

Run with `python -m pytest`. Result: **114 passed**, with 0 failures and 0 skips (about 30–40 s).

| File | Tests | What is covered |
|---|---|---|
| `test_synthetic_data.py` | 14 | Same seed gives identical data, different seeds differ; both bundled CSV files equal the generator output; configurable sizes; invalid settings; validation catches broken data; basket-size variety; planted associations are recovered and unrelated products are not linked; **every injected data problem is reported with its exact count** |
| `test_data.py` | 30 | Long and wide formats; aliases; whitespace; missing IDs, blanks and placeholders, with removal reasons; duplicates with summed quantities; capitalisation merging; ignored columns; nine kinds of invalid file; non-CSV and missing files; size and row limits; latin-1 files; **semicolon, tab and pipe separators**; **empty spreadsheet columns**; **unusual product names (commas, accents, pipes, long names, leading zeros) through mining, recommendations, report and ZIP**; **a 30,000-transaction upload**; summaries; encoding; product-name validation |
| `test_mining.py` | 17 | Support, confidence, lift, leverage and conviction against hand calculations; Apriori = FP-Growth; a plain-Python recount of every rule; thresholds; exclusions; empty results; the too-many-itemsets guard; invalid settings; deterministic ranking; explanations |
| `test_recommend.py` | 10 | Only applicable rules are used; basket products are never recommended, and none are duplicated; confidence, lift and support ordering; labelled fallback; empty and unknown baskets; invalid arguments; explanations match the supporting rule; display tables |
| `test_evaluation.py` | 8 | Reproducible, complete and disjoint splits; leave-one-out cases; a hand-checked metric case; **leakage guards** (mining and best sellers receive only training data; products seen only in the test set are never recommended); a reproducible summary; small datasets; a clear error when nothing can be evaluated |
| `test_insights.py` | 6 | Popularity and categories; cross-sell pairs listed once; bundle metrics; disjoint merchandising groups; insights computed from the data; every chart renders, including with empty input |
| `test_reporting.py` | 5 | Report sections and honest wording; the report and the ZIP when no rules are found; ZIP contents consistent with the rules; Markdown escaping |
| `test_cli.py` | 7 | Every CLI command, clean error messages, and writing to a new folder |
| `test_app.py` | 17 | Streamlit interface through `streamlit.testing`:<br>• the app starts on Overview<br>• valid, invalid and **demo-with-problems** uploads<br>• generating sample data<br>• rule search and the empty state<br>• a basket recommendation and an empty basket<br>• no applicable rule without the fallback<br>• running the evaluation<br>• bundles and settings downloads<br>• every page on a tiny dataset |

## Browser verification

A script drove headless Chrome through its DevTools protocol against `streamlit run app.py`. The script was a development aid and is not part of the repository. Every check passed:

- All eight pages load without error text.
- Smart Recommendations: the "Milk + Bread" example shows Butter first.
- Evaluation Lab: "Run evaluation" produces the chart and its interpretation.
- After navigating with the sidebar (same session), **Download report (Markdown)** and **Download all results (ZIP)** both produced files. The report included the evaluation section, and the ZIP held 10 files.
- An invalid upload through the real file input was rejected, and the demo data stayed active.
- Uploading `Data/demo_upload_with_issues.csv` loaded 2,997 transactions, showed the capitalisation warning, and switched the badge to "Uploaded".
- The Recommendations page worked on the uploaded data.
- "Generate sample data" made the generated data active, and Product Analytics, Association Rules and Bundle Opportunities all worked on it.

The screenshots in `docs/images/` were captured in this run.

## Clean-environment check

These steps were run on a fresh copy of exactly the files Git would publish (64 files; no virtual environment, caches or outputs), following the README:

1. `py -3.13 -m venv .venv` and `pip install -r requirements.txt`. `pip check` reported no broken requirements.
2. Both `Data/synthetic_retail_transactions.csv` and `Data/demo_upload_with_issues.csv` were regenerated **byte-identical** (same SHA-256).
3. `python -m pytest` ran 114 tests: 114 passed.
4. Every CLI command exited with 0: `summary`; `validate` on the demo upload and on a semicolon-separated file; `rules`; `recommend`; `insights`; `evaluate`; `report`; `scripts/benchmark.py`.
5. The walkthrough notebook executed without errors.
6. `streamlit run app.py` started and answered its health check.

## Problems found and fixed

| Round | Problem | How it was found | Fix |
|---|---|---|---|
| Build | Product Analytics crashed on datasets with 5 or fewer products | Interface test on a tiny dataset | Small catalogues show all products without a slider |
| Build | Product names on charts were cut off; the legend overlapped the title | Screenshot review | Axis auto-margins; legend moved below the chart |
| Build | `validate --removed-output` crashed when the folder did not exist | Clean-environment run | The folder is created; file-system errors are reported as messages |
| Build | pandas silently misread rows longer than the header | Edge-case test | Detected and rejected with a clear message |
| Quality | **Semicolon- and tab-separated exports were rejected** as "unrecognised columns" | Reliability probe with real-world export formats | The separator (comma, semicolon, tab, pipe) is detected from the header and shown in the validation summary |
| Quality | Spreadsheet exports with empty trailing columns produced a confusing "Unnamed: 2" warning | Reliability probe | Empty, unnamed columns are dropped quietly |
| Quality | Warning text said "Only 1 transactions" | Reliability probe | Correct singular and plural wording |
| Quality | Very long product names squeezed the charts | Reliability probe | Axis labels are shortened; the hover shows the full name |
| Quality | Inconsistent table headings ("hit rate@3 std", "transactions together") and "Hit rate" vs "Hit Rate" | UI review | One shared label and format map for all business tables; consistent terminology |

## Not tested

- Operating systems other than Windows, and Python versions other than 3.13 (the dependencies need 3.12 or newer).
- Uploads close to the 50 MB limit in a real browser. The size and row limits are unit-tested with lowered limits, and a 30,000-transaction upload is tested.
- Real retail exports (no real data was available).
- Accessibility (keyboard-only use, screen readers), mobile layouts, and several users at the same time.
- Long-running use: cached results stay in memory for the lifetime of the server process.
