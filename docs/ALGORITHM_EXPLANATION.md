# BasketWise — How the analysis works

## Market basket analysis in one paragraph

Each transaction is a basket of products. Market basket analysis looks for sets of products that appear together often (**frequent itemsets**). It then turns them into **association rules** of the form *A → B*, read as "baskets that contain A often also contain B". Each rule is scored with a few simple ratios that say how common it is, how reliable it is, and how much stronger it is than chance.

## Metrics

For a rule A → B over N transactions:

| Metric | Formula | Meaning |
|---|---|---|
| Support | support(A) = transactions containing A / total transactions | How common a product or set is |
| Confidence | confidence(A → B) = support(A ∪ B) / support(A) | Of the baskets with A, the share that also had B (directional) |
| Lift | lift(A → B) = confidence(A → B) / support(B) | How many times more often B appears with A than overall (symmetric) |
| Leverage | support(A ∪ B) − support(A) × support(B) | Observed minus expected co-occurrence if A and B were independent |
| Conviction | (1 − support(B)) / (1 − confidence(A → B)) | How much more often the rule would be wrong if A and B were independent; infinite when confidence is 100 % |
| All-confidence | support(set) / largest single-product support in the set | Used for bundles: at least this share of buyers of any product in the set bought the whole set |

**Interpreting lift:**
- **Above 1** means B appears with A more often than chance in the observed data: a positive association.
- **Equal to 1** means the two look independent.
- **Below 1** means they appear together less often than chance.

Lift does not show that A causes the purchase of B. Both products might be driven by something else, such as a cooking occasion or a promotion.

## Worked example

This is the five-basket example from Tan, Steinbach and Kumar's textbook *Introduction to Data Mining*, used in `tests/test_mining.py`:

| Basket | Products |
|---|---|
| T1 | Bread, Milk |
| T2 | Bread, Diapers, Beer, Eggs |
| T3 | Milk, Diapers, Beer, Cola |
| T4 | Bread, Milk, Diapers, Beer |
| T5 | Bread, Milk, Diapers, Cola |

For the rule Beer → Diapers:
- support(Beer ∪ Diapers) = 3 / 5 = 0.6
- confidence = 0.6 / support(Beer) = 0.6 / 0.6 = 1.0
- lift = 1.0 / support(Diapers) = 1.0 / 0.8 = 1.25
- leverage = 0.6 − 0.6 × 0.8 = 0.12
- conviction = (1 − 0.8) / (1 − 1.0) = infinite

The reverse rule, Diapers → Beer, has confidence 3 / 4 = 0.75 and the same lift of 1.25. This shows that confidence depends on direction and lift does not.

## Apriori and FP-Growth (both from mlxtend)

Both algorithms find every itemset whose support is at least the minimum support. They return the same itemsets; they only differ in how they search.

**Apriori** builds itemsets level by level:
1. Find the frequent single products.
2. Combine frequent sets of size *k* into candidates of size *k + 1*.
3. Discard any candidate that has an infrequent subset. This is valid because every subset of a frequent set must itself be frequent (the "downward-closure" property).
4. Count the remaining candidates in the data, and repeat.

**FP-Growth** compresses the transactions into a prefix tree (an FP-tree), where shared beginnings of baskets are stored once. It then mines frequent sets directly from the tree, without generating candidates. This usually helps when there are many frequent itemsets.

**Benchmark (`scripts/benchmark.py`):**
- At the default settings on the 47-product demo catalogue, Apriori was slightly faster (0.38 s vs 0.59 s at 120,000 transactions).
- At 0.2 % support (1,741 itemsets, 40,000 transactions), FP-Growth was 2.5–7× faster.
- FP-Growth is the default because it handles the expensive low-threshold case better.

## What BasketWise adds around mlxtend

| Step | Library or own code |
|---|---|
| One-hot encoding of baskets | mlxtend `TransactionEncoder` (BasketWise chooses dense or sparse output by size) |
| Frequent itemsets | mlxtend `apriori` / `fpgrowth` |
| Rules and their metrics | mlxtend `association_rules` |
| Validation, cleaning, synthetic data | BasketWise |
| Confidence threshold, product exclusions, the too-many-itemsets guard | BasketWise |
| Rule ranking with deterministic tie-breaks, plain-English explanations | BasketWise |
| Recommendation engine and best-seller fallback | BasketWise |
| Bundles (all-confidence), cross-sell pairs, merchandising groups (NetworkX community detection) | BasketWise (NetworkX provides the community algorithm) |
| Offline evaluation framework | BasketWise |

## How the recommender chooses

1. Keep the rules that have a single product on the right-hand side.
2. Sort them by the chosen metric. Confidence is the default; lift and support are alternatives. Ties are broken by the other metrics and then by product name.
3. Walk the sorted list. A rule A → B is used if every product in A is in the basket and B is not. Each product is suggested only once, through its first (best) matching rule.
4. If fewer than *n* products were found, add the best sellers that are not already in the basket. Label each as "best seller" and give the reason.

Every rule-based suggestion includes its supporting rule and a sentence describing what was observed. The tests check that each sentence matches the actual rule.

## Merchandising groups

Each single-product rule with a lift of at least 2 (the threshold can be adjusted in the app) becomes a weighted link between two products. NetworkX's greedy modularity community detection then splits this graph into groups of products that are linked more strongly to each other than to the rest. The groups are suggestions for shelf adjacency or themed promotions. They change with the threshold.
