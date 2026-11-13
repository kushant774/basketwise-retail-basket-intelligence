# BasketWise — Product requirements

Status labels: **Implemented** means built and covered by automated tests or verified in a browser. **Assumption** means a reasonable belief that has not been validated with real users.

## Problem statement

Small and medium-sized retailers already collect transaction data through their billing systems. They rarely use it to answer practical questions:
- Which products sell together?
- What should we suggest to a customer who has milk and bread in the basket?
- Which products should share a shelf or a promotion?

Market basket analysis can answer these questions, but existing tools assume knowledge of Python, data cleaning and association-rule metrics.

## Target users

| User | Main need |
|---|---|
| Small shop owner | Simple, trustworthy suggestions for what to place together or promote |
| Supermarket manager | Cross-selling and layout ideas backed by numbers |
| Retail analyst | Configurable rule mining, exports and an honest evaluation |
| Merchandising and inventory teams | Product groups and bundle candidates |

## Personas

The personas are illustrative assumptions, not based on user research.

**Priya, owner of a neighbourhood grocery.**
- Exports a CSV from her billing software each month.
- Does not know what "support" means.
- Wants to know which items to place near the counter, and which pairs to offer as a combo.

**Arjun, a retail analyst at a regional supermarket chain.**
- Comfortable with spreadsheets.
- Wants to tune thresholds, inspect individual rules, export results, and check whether rule-based suggestions are better than "show the best sellers".

## User pain points

These are assumptions:
- Data exports are messy: blank rows, inconsistent capitalisation, duplicate lines.
- Rule tables full of decimals are hard to interpret.
- People cannot tell whether a pattern is real or just reflects popular products.
- They have no way of knowing whether recommendations would beat a simple best-seller list.

## Goals

1. Turn a raw transaction CSV into validated data, with a clear report of everything that was changed or removed.
2. Show purchasing patterns in plain language, with every metric explained.
3. Provide explainable product recommendations for any basket.
4. Surface potential bundles, cross-sell pairs and merchandising groups.
5. Evaluate recommendations against a simple baseline, honestly and reproducibly.
6. Export results for use outside the app.

## Non-goals

- Generative AI, chatbots or LLM features.
- Proving that recommendations increase sales. That requires a live experiment.
- Real-time integration with point-of-sale systems, or cloud deployment.
- Personalised recommendations based on customer history.
- Implementing Apriori or FP-Growth from scratch; the mlxtend library is used.

## Core features and user stories

| # | User story | Acceptance criteria | Status |
|---|---|---|---|
| 1 | As a shop owner, I want to upload my sales CSV so I can analyse my own data. | Long and wide formats are accepted, with alias column names (including CamelCase headers such as `InvoiceNo`), comma, semicolon, tab or pipe separators, and UTF-8, UTF-16 or Windows-1252 text. Invalid files show a clear error that stays visible, and the previous dataset is kept. | Implemented (`tests/test_data.py`, `tests/test_app.py`, browser check) |
| 2 | As an analyst, I want to know what cleaning did to my data. | The validation summary shows rows received, valid transactions, products, removed blanks, returns, duplicates, rejected rows, the encoding and warnings. Removed rows can be downloaded with reasons. A warning appears when more than 5 % of rows are removed, or when dates are ambiguous or unreadable. Returns are never counted as purchases. | Implemented |
| 3 | As a user without data, I want sample data so I can try the product. | Bundled synthetic data, plus an in-app generator with a fixed seed, a configurable transaction count and a configurable product count. Synthetic data is clearly labelled. | Implemented |
| 4 | As a manager, I want to see best sellers, slow sellers and categories. | The Product Analytics page shows top and least purchased products, category reach, basket sizes, weekly volume and a frequency table. | Implemented |
| 5 | As an analyst, I want to tune and explore association rules. | Algorithm, support, confidence, lift, maximum size and ignored products are configurable, and every Apply takes effect at once (including several in a row). The rule table can be searched and sorted, each rule has a plain-English explanation, rules export to CSV, and empty states appear when no rules pass or no rule matches the search. | Implemented (`tests/test_app.py`, browser check) |
| 6 | As a shop owner, I want suggestions for a basket, with reasons. | Products already in the basket, and ignored products, are never suggested. Each suggestion shows its reason and supporting rule. Ranking can use confidence, lift or support. Best sellers fill gaps and are labelled. Empty and unknown baskets are handled. The basket is kept when moving between pages. | Implemented |
| 7 | As a merchandiser, I want bundle and grouping ideas. | Bundles are only sets bought together more often than chance (lift above 1 and the minimum lift), ranked by all-confidence, with their categories and a suggested use. Cross-sell pairs and merchandising groups are included. Ignored products never appear. | Implemented |
| 8 | As an analyst, I want to know whether recommendations beat best sellers. | Three recommendation methods are compared with the best-seller baseline on Hit Rate@3, Hit Rate@5, availability and coverage over repeated seeded splits. There is no train/test leakage, the settings used are shown with the result, and the methodology and limitations (including the selection effect of tuning on the demo data) are shown. | Implemented (`tests/test_evaluation.py`) |
| 9 | As a manager, I want to share results. | A Markdown report and a ZIP of CSV files (rules, bundles, cleaned data, removed rows, evaluation) can be downloaded. | Implemented (`tests/test_reporting.py`) |

## Success metrics

These are proposed and have not been measured; there are no real users yet.

| Metric | Why it matters |
|---|---|
| Share of uploads that validate without errors | Shows whether the input formats match real exports |
| Time from upload to the first useful insight | The core promise of simplicity |
| Hit Rate@3 compared with the best-seller baseline on the retailer's own data | Offline evidence before running an experiment |
| Change in average basket value in an A/B test of recommendations | The only real measure of business impact |
| Repeat usage per retailer per month | Shows whether the insights are worth returning for |

## Risks

| Risk | Mitigation in the current product |
|---|---|
| Users read associations as causation | Every explanation says "in the observed data"; the docs and app state that associations are not causation and that A/B testing is needed |
| Synthetic demo results are mistaken for real-world performance | Synthetic data is labelled everywhere; the evaluation page explains that its associations were planted |
| Messy real exports fail to load | Alias and CamelCase column names, two layouts, encoding detection, detailed errors and downloadable removed rows |
| Returns are counted as purchases | Rows with a negative quantity are removed with their own reason |
| Dates are misread (day/month versus month/day) | The order is detected per file; ambiguous files use day/month/year with a warning |
| Very low thresholds make mining slow or huge | A slider minimum of 0.2 % support; an upper bound on the itemset count is checked before mining (limit 200,000) and the candidate-rule count before rule generation (limit 500,000), each with a clear message |
| Popular products dominate suggestions | Ranking options (lift, support), ignored products (which also apply to the best-seller fallback), and a baseline comparison that exposes the effect |

## Future improvements

These are not implemented:
- Time-based evaluation (train on earlier months, test on later ones) and scheduled re-mining.
- Margin-aware or stock-aware ranking.
- Statistical significance filtering of rules.
- Saving analyses, and multi-user deployment with authentication.
- Direct import from common billing or point-of-sale exports.
- An A/B test toolkit for measuring the real impact of recommendations.
