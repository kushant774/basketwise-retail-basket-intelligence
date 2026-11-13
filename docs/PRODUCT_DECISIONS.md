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
- **Evidence (`scripts/benchmark.py`):**
  - At the default settings on the 47-product catalogue, Apriori was slightly faster: 0.38 s vs 0.59 s at 120,000 transactions.
  - At a low minimum support of 0.2 %, with 1,741 itemsets, FP-Growth was 2.5× faster on the dense matrix and 7× faster on the sparse one.
- **Why:** both are fast at the defaults, and FP-Growth degrades far less when users lower the thresholds, which is the risky case. The in-app help text states this trade-off.

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

## 7. Best sellers as a labelled fallback

- **Decision:** when fewer rules apply than suggestions requested, fill the slots with best sellers, and label them "best seller" with their own reason.
- **Evidence:**
  - Confidence-ranked rules alone produced at least one suggestion in 98.8 % of evaluated cases.
  - With the fallback, availability is 100 %, and Hit Rate@5 rose from 50.2 % to 53.1 %.
- **Why the label matters:** best-seller suggestions are not based on the basket, and users should know that.

## 8. Every recommendation carries an explanation

- **Decision:** show the supporting rule's antecedent, consequent, support, confidence and lift, plus a sentence such as "Customers whose basket contained Bread and Milk also bought Butter in 41% of those transactions (3.8x its usual rate) in the observed data."
- **Why:** a merchandiser will not act on a black-box suggestion. The explanation lets them sanity-check the pattern. The wording describes observations only, never promised sales.

## 9. Bundles ranked by all-confidence, not lift

- **Decision:** rank bundle candidates (pairs and triples) by all-confidence, the support of the set divided by the largest single-product support in it.
- **Why:**
  - Lift against full independence grows mechanically with set size, so triples would always look better than pairs.
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
