# BasketWise — Product decisions

Each entry records a decision made while building the current version: the reason for it, the alternatives considered, and the evidence available. "Evidence" means something measured in this project. Anything else is labelled as reasoning or assumption.

## 1. Use association-rule mining for the core analysis

- **Decision:** find patterns with frequent itemsets and association rules.
- **Why:** the input is anonymous baskets (transaction ID plus products), which is what retailers' billing exports typically contain. Association rules need nothing else and produce results a shop owner can read ("baskets with A often contain B").
- **Alternatives:**
  - Collaborative filtering and matrix factorisation need stable customer IDs and purchase histories.
  - Sequence models need ordered timestamps.
  - Deep learning adds complexity and reduces explainability for a small-retailer audience.
- **Trade-off:** rules capture co-occurrence only; there is no personalisation and no time ordering.

## 2. Use mlxtend instead of implementing Apriori and FP-Growth

- **Decision:** call mlxtend's `TransactionEncoder`, `apriori`, `fpgrowth` and `association_rules`.
- **Why:** they are mature, widely used implementations, and rewriting them would add risk without product value. The project's own work is the product logic around them: validation, settings, ranking, explanations, recommendations, fallback, evaluation, insights and reporting.
- **Evidence:**
  - Tests confirm that Apriori and FP-Growth return identical itemsets and rules on the demo data.
  - A plain-Python recount reproduces every rule's support, confidence and lift.
  - Hand calculations on a textbook example match exactly.

## 3. Default algorithm: FP-Growth

- **Decision:** FP-Growth is the default and Apriori is selectable.
- **Evidence (`scripts/benchmark.py`, one laptop; timings vary between computers):**
  - At the app's default settings (support 1 %, at most 3 products) on the 47-product catalogue, Apriori was faster on the dense matrix (0.40 s vs 0.88 s at 120,000 transactions) and about equal on the sparse one.
  - At a low minimum support of 0.2 % with no size limit (1,741 itemsets, 40,000 transactions), FP-Growth was 2.4× faster on the dense matrix and 6.8× faster on the sparse one.
- **Why:** both take under a second at the defaults, and FP-Growth degrades far less when users lower the thresholds, which is the risky case. The in-app help text states this trade-off.

## 4. Generate synthetic demo data

- **Decision:** ship a generated dataset (`scripts/generate_data.py`, seed 42) instead of a public dataset.
- **Why:**
  - The previously used dataset had an unknown origin and licence.
  - Synthetic data has known ground truth: the planted shopping missions let tests check that the pipeline finds real associations (Pasta → Pasta Sauce) and does not link unrelated products (Pasta → Shampoo).
- **Trade-off:** synthetic associations are cleaner than real ones, so the demo flatters the method. This is labelled in the app, the README and the evaluation report.

## 5. Always compare against a best-seller baseline

- **Decision:** every evaluation includes "recommend the most popular products not already in the basket".
- **Why:** best sellers are the simplest thing a shop could do without any analytics. A recommendation method is only worth building if it beats them, and the comparison stops the demo from overstating results.

## 6. Confidence as the default recommendation ranking

- **Decision:** rank applicable rules by confidence by default; lift and support are options.
- **Evidence:** in the current evaluation (5 splits), confidence ranking beat lift ranking. Hit Rate@3 was 42.5 % vs 38.4 %; Hit Rate@5 was 50.2 % vs 48.2 %.
- **Reasoning:**
  - Confidence directly estimates "how often customers with this basket also bought the product".
  - Lift favours rare pairings: a niche product can have high lift but low confidence. That is useful for discovering cross-sell ideas, so lift remains available for that purpose and in the rule explorer.
- **Caveat:** this choice was made by looking at the evaluation on the same demo data, so the reported lead of the default method is slightly optimistic (a selection effect). The app and the evaluation report say so.

## 7. Best sellers as a labelled fallback

- **Decision:** when fewer rules apply than suggestions requested, fill the slots with best sellers, and label them "best seller" with their own reason.
- **Evidence:**
  - Confidence-ranked rules alone produced at least one suggestion in 98.8 % of evaluated cases.
  - With the fallback, availability is 100 %, and Hit Rate@5 rose from 50.2 % to 53.1 %.
- **Why the label matters:** best-seller suggestions are not based on the basket, and users should know that.

## 8. Every recommendation carries an explanation

- **Decision:** show the supporting rule's antecedent, consequent, support, confidence and lift, plus a sentence such as "Customers whose basket contained Bread and Milk also bought Butter in 41% of those transactions (3.8x its usual rate) in the observed data."
- **Why:** a merchandiser will not act on a black-box suggestion. The explanation lets them sanity-check the pattern. The wording describes observations only, never promised sales.

## 9. Bundles: filtered by lift, ranked by all-confidence

- **Decision:** offer only sets whose lift against independence is above 1 and at least the minimum lift setting, then rank them (pairs and triples) by all-confidence, the support of the set divided by the largest single-product support in it.
- **Why:**
  - All-confidence ignores how often products would meet by chance, so on its own it can rank two very popular but unrelated (or even negatively associated) products highly. An audit showed a pair with lift 0.92 ranked second on constructed data. The lift filter makes "bought together more often than chance" true.
  - Lift against full independence grows mechanically with set size, so it is not used for ranking; triples would always look better than pairs.
  - All-confidence means "at least this share of buyers of any product in the set bought the whole set", which is a practical bundle question.

## 10. Never drop data silently

- **Decision:** every removed row is kept with its reason and can be downloaded. A prominent warning appears when more than 5 % of rows are removed.
- **Why:** an analysis is only trustworthy if the user can see what was excluded.

## 11. Offline evaluation is not proof of business value

- **Decision:** present offline hit rate as evidence about relevance only, and recommend an A/B test for impact.
- **Why:** hit rate rewards predicting products the customer would have bought anyway. Business value is the extra purchases a suggestion causes, which needs a controlled experiment: randomly assign customers or sessions to see recommendations or not, then compare basket value and attach rate.

## 12. Keep the stack simple and local

- **Decision:** use Python, pandas, NumPy, mlxtend, Plotly, NetworkX and Streamlit. There is no database, cloud, authentication or paid API.
- **Why:** the target user needs a tool that runs on a laptop with a CSV. Each extra service adds setup cost without solving the core problem.

## 13. "Ignore products" applies to every suggestion, not to sales statistics

- **Decision:** an ignored product (for example a carrier bag) is left out of rules, recommendations (including the best-seller fallback), bundles, cross-sell pairs, merchandising groups and the evaluation. Descriptive statistics (best sellers chart, product frequency table) still include it.
- **Why:** the point of ignoring a product is that it should never be suggested. Before this was fixed, an ignored best seller still appeared as a "best seller" suggestion in almost every basket. Sales statistics describe what happened, so hiding a product there would misreport the data.
- **Evaluation:** every method treats ignored products the same way (never recommended, never used as the hidden product), so the comparison stays fair.

## 14. Returns are removed; duplicate lines are added up

- **Decision:** rows with a negative quantity are removed as returns or cancellations, with that reason in the removed-rows download. Several lines for the same product in one transaction are merged and their quantities added.
- **Why:** counting a returned product as a purchase would invent associations (a cancellation invoice would look like a basket). Duplicate lines are treated as separate scans of the same product, which is how most tills record them; rules only use whether a product is in the basket, so this choice cannot change any rule.

## 15. Ambiguous dates default to day/month/year

- **Decision:** the order of day and month is detected per file; if no date settles it, day/month/year is used and a warning says so.
- **Why:** the target users are Indian retailers, whose exports use day/month/year. A US file in which no day is above 12 would be misread, which is why the warning is shown. Only the date range and the weekly chart use dates; the rules do not.

## 16. Refuse very large settings before mining, not after

- **Decision:** before mining, compute a guaranteed upper bound on the number of frequent itemsets and refuse settings above 200,000; before rule generation, refuse more than 500,000 candidate rules.
- **Why:** the earlier check counted itemsets only after mining had finished, so a dense file at the lowest support with no size limit could use more than a gigabyte of memory first. The bound is cheap (it only needs per-product counts and basket sizes), never refuses a setting the app offers on the demo data, and turns a long freeze into an immediate, actionable message.

## 17. Recognise generated data by its IDs

- **Decision:** a file is labelled synthetic when every transaction ID and every customer ID has the generator's format (`SYN-T…`, `SYN-C…`), wherever the file comes from.
- **Why:** the demo upload is itself generated data, and reports about it must carry the synthetic-data disclaimer. File names are not reliable, and requiring both ID formats in full means real data is not labelled synthetic by accident.
