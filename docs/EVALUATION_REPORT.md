# BasketWise — Evaluation report

To reproduce these results, run `python -m market_basket evaluate --splits 5` (and `--splits 10`), or use the Evaluation Lab page with the defaults. They use the synthetic demo data (8,000 transactions) and the default mining settings: FP-Growth, minimum support 1 %, confidence 20 %, lift 1.2, at most 3 products per itemset.

## What was measured

The question: *if a customer has already put some products in the basket, do BasketWise's suggestions include another product that was actually in the same basket?* (Baskets have no order, so this is not "the product bought next".) Three recommendation methods are compared with a simple alternative, suggesting the best sellers.

### Method

1. **Split.** Shuffle the transactions with a fixed seed and use 80 % for training and 20 % for testing (1,600 test transactions). Repeat with seeds 42, 43, 44, ... for several splits.
2. **Train.** Mine rules and compute best-seller shares from the **training split only**.
3. **Hide one product.** For every test transaction with at least two products, hide each product in turn. The remaining products are the query basket, and the hidden product is the target. Single-product transactions cannot be evaluated this way and are skipped. This gives 5,760 cases per split on average (5,683 to 5,896 in the five splits).
4. **Score.** Each method returns up to 5 suggestions for the query basket. Hit Rate@k is the share of cases where the target is among the first k.

### Leakage controls (tested in `tests/test_evaluation.py`)

- The mining function and the best-seller calculation are checked to receive exactly the training transactions for every split.
- A product that appears only in test data is never recommended.
- Every method is scored on exactly the same cases.
- Products the user chose to ignore are treated the same way by every method: no method may recommend them, cases that would hide them are skipped, and they are left out of coverage. With the default settings nothing is ignored.

## Methods compared

Three recommendation methods and the baseline:

| Method | Description |
|---|---|
| Best-seller baseline | The most popular training products that are not already in the basket |
| Rules ranked by confidence | Association rules only, highest confidence first |
| Rules ranked by lift | Association rules only, highest lift first |
| Rules by confidence + best-seller fallback | Confidence-ranked rules, with best sellers filling any free slots |

## Metrics

| Metric | Definition |
|---|---|
| Hit Rate@3 / @5 | Share of cases where the hidden product is in the top 3 / top 5 suggestions |
| Availability | Share of cases with at least one suggestion |
| Coverage | Share of training products the method recommended at least once across all cases |
| Splits beating the baseline | Number of splits where the method's hit rate exceeded the baseline's on the same test cases |

## Results

**5 splits (seeds 42–46), 5,760 cases per split on average:**

| Method | Hit Rate@3 | Hit Rate@5 | Availability | Coverage | Splits beating the baseline (@3 / @5) |
|---|---|---|---|---|---|
| Best-seller baseline | 19.7 % (±0.6) | 30.1 % (±0.5) | 100 % | 26.0 % | – |
| Rules ranked by confidence | 42.5 % (±0.2) | 50.2 % (±0.1) | 98.8 % | 76.6 % | 5 / 5 |
| Rules ranked by lift | 38.4 % (±0.5) | 48.2 % (±0.5) | 98.8 % | 76.6 % | 5 / 5 |
| Rules by confidence + fallback | **43.5 %** (±0.2) | **53.1 %** (±0.2) | 100 % | 76.6 % | 5 / 5 |

The numbers in brackets are the standard deviation across splits, in percentage points.

**10 splits (seeds 42–51):** the results were almost the same.
- Baseline: 19.8 % / 30.1 %.
- Confidence: 42.3 % / 49.9 %.
- Lift: 37.9 % / 47.7 %.
- Confidence + fallback: 43.3 % / 52.9 %.
- Every rule method beat the baseline on all 10 splits.

## Interpretation

- On this data, all rule-based methods found the hidden product far more often than best sellers. The best method was confidence plus fallback: +23.8 points at Hit Rate@3 and +23.0 points at Hit Rate@5.
- Confidence ranking beat lift ranking. Lift favours rare pairings, which helps when discovering niche cross-sell ideas but is less likely to match what most customers buy.
- The fallback adds suggestions when no rule applies (availability rises from 98.8 % to 100 %) and improves Hit Rate@5 by about 3 points.
- Rule-based methods recommended 77 % of the catalogue at least once, compared with 26 % for best sellers. Rules spread attention over far more products.

## Why these results should not be over-interpreted

1. **The data is synthetic and its associations were planted.** The generator builds baskets from shopping missions such as "pasta night" (Pasta, then Pasta Sauce 60 % of the time). Rules recovering these patterns shows that the pipeline works; it does not show that real stores have patterns this strong. Real retail data is noisier, so the gap to best sellers may be much smaller or even reversed.
2. **Offline hit rate is not business impact.** A hit means the suggestion matched something the customer bought anyway. A recommendation creates value only if it causes an extra purchase, and offline data cannot measure that.
3. **The split is random, not by time.** In practice, rules learned from past months are used on future months, and preferences change (seasonality, new products).
4. **Each basket contributes several cases.** The cases within one split are not independent. The spread across splits is the more honest guide to variability.
5. **The defaults were chosen on this same data.** Within one run, the test transactions are never used to build recommendations. However, BasketWise's default ranking (confidence) and thresholds were chosen after looking at results like these on the demo data, so the default method's lead over the other rule methods is slightly optimistic. The effect is small next to the 20-point gap to the baseline, but this is not a fully held-out test. A clean check would fix the settings first and then evaluate on new data, such as a later time period.

## What would establish real business impact: an A/B test

1. **Randomise.** Assign customers, or checkout sessions, at random to a control group (no suggestions, or the best-seller list) or a treatment group (BasketWise suggestions).
2. **Choose the metrics in advance.**
   - Primary: average basket value or items per basket.
   - Secondary: the attach rate of suggested products, and margin per basket.
   - Guardrails: checkout time, and complaints or returns.
3. **Run long enough.** Cover full weekly cycles, with a sample size calculated from the smallest effect worth acting on.
4. **Compare.** Use a confidence interval for the difference, and check for novelty effects before rolling out.
