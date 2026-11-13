# BasketWise — Demo walkthrough (about 7 minutes)

This walkthrough follows one complete user journey: upload data, check its quality, find patterns, recommend products, plan bundles, evaluate, and export. All numbers below were produced by the app with `Data/demo_upload_with_issues.csv` and the default settings.

The file is synthetic. It was made with:

```
python scripts/generate_data.py --transactions 3000 --seed 7 --with-issues --output Data/demo_upload_with_issues.csv
```

It contains deliberately injected data problems: blank product names, missing transaction IDs, capitalisation variants, extra spaces, duplicate rows and an unused column.

## Before you start

```powershell
.venv\Scripts\streamlit run app.py
```

Open <http://localhost:8501>. Use the sidebar to move between pages; reloading the browser starts a new session and clears an upload.

## 1. Overview — the problem (30 s)

**Show:** the Overview page, with its metric cards, top-selling products and key insights.

**Say:** "Retailers have transaction data but rarely know which products are bought together. BasketWise turns a sales CSV into explained recommendations and bundle ideas, without needing Python. The badge shows that this is synthetic demo data."

## 2. Data Explorer — upload and trust the data (90 s)

**Do:** open **Data Explorer**, then upload `Data/demo_upload_with_issues.csv`.

**You will see:**
- 11,070 rows received
- 2,997 valid transactions
- 47 products
- 275 rows removed or rejected:
  - 55 missing transaction IDs
  - 110 blank product names
  - 110 duplicate products merged
- 219 capitalisation variants merged (for example `MILK` merged into `Milk`)
- 219 names with extra spaces fixed
- the unused `store_id` column reported
- the badge changes to "Synthetic data: demo_upload_with_issues.csv", because BasketWise recognises files made by its own generator

**Say:** "Real exports are messy, so the first job is trust. Nothing is dropped silently: every removed row can be downloaded with its reason, and a warning appears if more than 5% of rows are lost. Semicolon-, tab- and pipe-separated exports, Excel's Windows-1252 and UTF-16 text, headers such as `InvoiceNo`, returns with negative quantities and day/month/year dates are all handled."

## 3. Product Analytics — what sells (30 s)

**Do:** open **Product Analytics**, then click through the tabs (top products, least purchased, categories, basket sizes).

**Say:** "This is the descriptive view a manager expects: best sellers, slow sellers and category reach."

## 4. Association Rules — what sells together (75 s)

**Do:** open **Association Rules**, keep the defaults (support 1%, confidence 20%, lift 1.2), and expand "What do support, confidence and lift mean?". Search for `pasta` and select a rule.

**You will see:** 242 frequent itemsets and 233 rules. The first rule found by the search is explained below the table: in 64% of baskets with Cheese and Pasta, Pasta Sauce was also bought, which is a lift of 13.8.

**Say:** "Apriori and FP-Growth come from the mlxtend library. BasketWise adds thresholds, ranking and a plain-English reading of every rule. Lift above 1 means a positive association in this data, not proof of cause."

**Optional:** add Milk to **Ignore products** and click **Apply settings**. The sidebar updates at once, and Milk disappears from rules, suggestions (including best sellers) and bundles, while the sales charts still include it. Remove it again before continuing.

## 5. Smart Recommendations — act on it (75 s)

**Do:** open **Smart Recommendations** and click the **Milk + Bread** quick example. Then open **Compare ranking methods**.

**You will see:**
1. Butter: 44.8% of Milk-and-Bread baskets also had Butter (lift 4.1).
2. Eggs: 36.7%.
3. Tea: 26.8% (from the rule Milk → Tea).

Each suggestion shows its reason and supporting rule. Best sellers fill any remaining slots and are labelled as such. The basket stays as it is when you visit another page and come back.

**Say:** "Every suggestion is explainable. Products already in the basket are never suggested. If no rule applies, best sellers fill the gap, clearly labelled."

## 6. Bundle Opportunities — plan promotions (45 s)

**Do:** open **Bundle Opportunities**.

**You will see:**
- The top bundle is Pasta + Pasta Sauce (99 transactions together, all-confidence 43%).
- Cross-sell pairs such as Conditioner → Shampoo (confidence 69%).
- Merchandising groups and a relationship map.

**Say:** "A set is only offered as a bundle if its products are bought together more often than chance. The candidates are then ranked by all-confidence, so pairs and triples are compared fairly. These are potential opportunities to test, not guaranteed wins."

## 7. Evaluation Lab — is it better than best sellers? (75 s)

**Do:** open **Evaluation Lab** with 5 splits, a 20% test share and seed 42. Click **Run evaluation**, then open **How the evaluation works**.

**You will see (three recommendation methods and the best-seller baseline; 2,090 hidden-product cases per split on average):**

| Method | Hit Rate@3 | Hit Rate@5 |
|---|---|---|
| Best-seller baseline | 21.1% | 30.0% |
| Rules ranked by confidence | 40.6% | 48.3% |
| Rules ranked by lift | 37.1% | 46.1% |
| Confidence + best-seller fallback | **41.8%** | **51.2%** |

Every rule-based method beat the baseline in 5 of 5 splits.

**Say:** "Rules and best sellers are learned from the training split only. In each test basket we hide one product and check whether it appears in the top suggestions. Rules clearly beat best sellers here, but the data is synthetic with planted patterns, the default settings were tuned on this kind of data, and offline hit rate is not extra sales. An A/B test would be needed to measure business impact."

## 8. Settings & Data — share the results (30 s)

**Do:** use the sidebar to open **Settings & Data**, then download the Markdown report and the ZIP.

**Say:** "The report includes the data-quality summary, rules, opportunities and the latest evaluation. The ZIP has every table as CSV."

## Questions to expect after the demo

See `docs/INTERVIEW_PREPARATION.md`. The most common ones are: why confidence ranking is the default, why lift can underperform, how leakage is prevented, and how you would run an A/B test.
