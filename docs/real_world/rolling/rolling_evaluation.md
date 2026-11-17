# BasketWise multi-cutoff chronological validation (UCI Online Retail)

Generated 2026-11-17 20:44 UTC by BasketWise 1.2.0. Every number in this report was computed by the code in this repository from the file named in section 1; none was typed by hand.

**Question.** Does the single-split result hold at other times? At every monthly cutoff, do association rules learned only from earlier invoices complete the next month's baskets more often than best sellers?

**Answer for this data.** In all 10 folds the association-rule recommender found the hidden product in the top 3 more often than the best sellers, and every fold's 95% interval for the difference is above zero. The difference ranged from +6.2 points (test period from 2011-12-01) to +8.8 points (from 2011-08-01). Pooled over all folds (10 folds, test periods from 2011-03-01 to 2011-12-09): 8.9% against 1.4%, +7.4 points (95% interval +7.2 to +7.7 points). Pooled over folds before the single-split test period (7 folds, test periods from 2011-03-01 to 2011-09-30): 9.5% against 1.7%, +7.8 points (95% interval +7.6 to +8.0 points).

**What this does not show:** next-purchase or next-order prediction (the hidden product comes from the same invoice); extra sales, revenue or any causal effect of showing recommendations (that needs an A/B test); real customer adoption (no recommendation was shown to any customer); how other shops or other periods would behave (the data comes from one online retailer).

This report extends the single-split report (`real_world_evaluation.md`), which has one test period. It uses the same data, preprocessing, methods and metrics, but its numbers come from different training and test periods, so they must not be added to the single split's numbers. The synthetic benchmark in `docs/EVALUATION_REPORT.md` is a separate experiment again.

## 1. Dataset

| item | value |
|---|---|
| Dataset | Online Retail, UCI Machine Learning Repository |
| Citation | Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/legalcode |
| Retrieved (UTC) | 2026-11-14T05:10:06+00:00 |
| File used | online_retail.csv (SHA-256 7615e19743f2c3e8e641375743c276dbb0ed64a70cf210464ce72c335db3d221) |
| Same file as the published results | yes |
| Valid invoices after preprocessing | 19,773 |

Data quality and preprocessing follow the single-split pipeline; `uci-prepare` with the same options writes their reports. Rows without a customer ID were kept in this run.

## 2. Protocol

1. **Cutoffs:** first day of every month, from 3 months after the month of the first invoice: 2011-03-01, 2011-04-01, 2011-05-01, 2011-06-01, 2011-07-01, 2011-08-01, 2011-09-01, 2011-10-01, 2011-11-01, 2011-12-01.
2. **Folds:** each fold learns from every invoice before its cutoff (an expanding window) and tests on the invoices up to the next cutoff; the last fold tests up to the last invoice. No invoice is tested twice.
3. **Inside each fold** the single-split protocol runs unchanged: the 9 candidate settings are compared on the latest 20% of that fold's training invoices, rules and best sellers are learned from that fold's training invoices, and every product of every test invoice with at least two products is hidden once. K = 3.
4. **Intervals:** 1,000 bootstrap resamples of whole test invoices (seed 42), the same resamples for both methods, per fold and for each pooled group.
5. **Pooled groups:** all folds, and the folds whose test period ends on or before the single-split cutoff (2011-10-26), which share no test invoice with the single split.

The recommender is BasketWise's default (rules ranked by confidence, best sellers filling free slots). The baseline suggests the fold's training-period best sellers that are not already in the basket.

## 3. Results per fold

| fold | test period | training invoices | test invoices | cases | support / confidence | rules | best-seller baseline | association-rule recommender | difference (95% interval) | before single-split test |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2011-03-01 to 2011-03-31 | 3,724 | 1,440 | 34,929 | 1.0% / 20% | 3,266 | 1.6% | 8.1% | +6.5 (+6.0 to +7.2 points) | yes |
| 2 | 2011-04-01 to 2011-04-28 | 5,164 | 1,235 | 28,473 | 1.0% / 30% | 1,092 | 1.6% | 8.5% | +6.9 (+6.3 to +7.6 points) | yes |
| 3 | 2011-05-01 to 2011-05-31 | 6,399 | 1,668 | 35,363 | 1.0% / 20% | 1,701 | 1.9% | 8.7% | +6.8 (+6.2 to +7.4 points) | yes |
| 4 | 2011-06-01 to 2011-06-30 | 8,067 | 1,525 | 35,160 | 0.5% / 20% | 7,158 | 1.8% | 10.2% | +8.5 (+7.7 to +9.2 points) | yes |
| 5 | 2011-07-01 to 2011-07-31 | 9,592 | 1,452 | 37,835 | 0.5% / 10% | 12,685 | 1.7% | 9.5% | +7.9 (+7.2 to +8.7 points) | yes |
| 6 | 2011-08-01 to 2011-08-31 | 11,044 | 1,339 | 33,737 | 0.5% / 10% | 11,943 | 1.8% | 10.6% | +8.8 (+8.1 to +9.6 points) | yes |
| 7 | 2011-09-01 to 2011-09-30 | 12,383 | 1,818 | 48,199 | 0.5% / 10% | 12,157 | 1.4% | 10.2% | +8.8 (+8.2 to +9.4 points) | yes |
| 8 | 2011-10-01 to 2011-10-31 | 14,201 | 2,005 | 57,320 | 0.5% / 20% | 5,534 | 1.2% | 8.3% | +7.1 (+6.4 to +7.8 points) |  |
| 9 | 2011-11-01 to 2011-11-30 | 16,206 | 2,751 | 80,168 | 0.5% / 20% | 5,450 | 1.0% | 7.9% | +6.9 (+6.3 to +7.5 points) |  |
| 10 | 2011-12-01 to 2011-12-09 | 18,957 | 816 | 24,308 | 0.5% / 10% | 12,990 | 1.0% | 7.2% | +6.2 (+5.0 to +7.4 points) |  |

Columns: Hit Rate@3 for each method on that fold's test period, and the difference in percentage points. 'support / confidence' is the setting chosen on that fold's validation period.

## 4. Summary over the folds

- recommender higher: 10; no clear difference: 0; baseline higher: 0 (out of 10 folds).
- Recommender Hit Rate@3 per fold: from 7.2% to 10.6%; best sellers: from 1.0% to 1.9%.

Pooled results (every case of the folds in the group, each method on the same cases):

| folds | method | cases | Hit Rate@3 | 95% interval | difference vs baseline (95% interval) | invoice-averaged Hit Rate@3 | availability |
|---|---|---|---|---|---|---|---|
| all folds (10) | Best-seller baseline | 415,492 | 1.4% | 1.4% to 1.5% | - | 1.9% | 100.0% |
| all folds (10) | Rules ranked by confidence | 415,492 | 8.9% | 8.6% to 9.1% | +7.2 to +7.6 points | 12.5% | 98.8% |
| all folds (10) | Rules ranked by lift | 415,492 | 6.8% | 6.6% to 7.0% | +5.2 to +5.6 points | 10.2% | 98.8% |
| all folds (10) | Rules by confidence + best-seller fallback | 415,492 | 8.9% | 8.6% to 9.1% | +7.2 to +7.7 points | 12.6% | 100.0% |
| folds before the single-split test period (7) | Best-seller baseline | 253,696 | 1.7% | 1.6% to 1.7% | - | 2.2% | 100.0% |
| folds before the single-split test period (7) | Rules ranked by confidence | 253,696 | 9.4% | 9.2% to 9.7% | +7.5 to +8.0 points | 12.6% | 98.6% |
| folds before the single-split test period (7) | Rules ranked by lift | 253,696 | 7.2% | 7.0% to 7.5% | +5.3 to +5.8 points | 10.2% | 98.6% |
| folds before the single-split test period (7) | Rules by confidence + best-seller fallback | 253,696 | 9.5% | 9.2% to 9.7% | +7.6 to +8.0 points | 12.7% | 100.0% |

## 5. Threshold selection per fold

| fold | validation period | settings compared | chosen support / confidence / lift | validation Hit Rate@3 (chosen) | validation Hit Rate@3 (baseline) |
|---|---|---|---|---|---|
| 1 | 2011-02-09 to 2011-02-28 | 6 of 9 | 1.0% / 20% / 1.2 | 8.2% | 1.8% |
| 2 | 2011-03-10 to 2011-03-31 | 6 of 9 | 1.0% / 30% / 1.2 | 8.8% | 1.4% |
| 3 | 2011-03-31 to 2011-04-28 | 6 of 9 | 1.0% / 20% / 1.2 | 8.6% | 1.7% |
| 4 | 2011-05-03 to 2011-05-31 | 9 of 9 | 0.5% / 20% / 1.2 | 9.8% | 1.9% |
| 5 | 2011-05-23 to 2011-06-30 | 9 of 9 | 0.5% / 10% / 1.2 | 10.1% | 1.8% |
| 6 | 2011-06-15 to 2011-07-31 | 9 of 9 | 0.5% / 10% / 1.2 | 9.5% | 1.6% |
| 7 | 2011-07-07 to 2011-08-31 | 9 of 9 | 0.5% / 10% / 1.2 | 9.9% | 1.8% |
| 8 | 2011-08-08 to 2011-09-30 | 9 of 9 | 0.5% / 20% / 1.2 | 10.3% | 1.5% |
| 9 | 2011-09-11 to 2011-10-31 | 9 of 9 | 0.5% / 20% / 1.2 | 8.8% | 1.2% |
| 10 | 2011-10-16 to 2011-11-30 | 9 of 9 | 0.5% / 10% / 1.2 | 7.8% | 1.1% |

Every candidate of every fold is in `rolling_threshold_selection.csv`.

## 6. Limitations

- One retailer: every fold comes from the same online retailer and the same year, so the folds say nothing certain about other shops.
- The folds are not independent. Their training periods overlap (each one contains the previous one), neighbouring months share customers and seasons, and the pooled intervals resample test invoices, not periods. Each fold's interval is reported separately and is not corrected for making several comparisons: with many folds, one interval can exclude zero by chance.
- Like the single split, this is offline basket completion: the hidden product comes from the same invoice, and a hit means the suggestion matched a product the customer bought anyway. It shows no extra sales, revenue or causal effect.
- Each fold chose its own thresholds on its own validation period, so every fold uses only data before its cutoff. 4 different settings were chosen, so the folds do not all use the same rules settings.
- The earliest folds learn from little data: the smallest training period has 3,724 invoices (fold 1).
- Rules have at most 2 products and the threshold grid is small, as in the single split.
- In folds 1, 2 and 3, 9 candidate settings were refused by BasketWise's safety limits (the reasons are in rolling_threshold_selection.csv), so those folds chose among the remaining settings.
- In folds 4, 5, 6, 7, 8, 9 and 10, the chosen minimum support (0.50%) is the lowest value in the grid, so a value outside the grid might have scored better on validation.
- The last fold's test period is short: it runs from 2011-12-01 to 2011-12-09 (9 days), where the data ends.
- The test periods of 3 folds (from 2011-10-01) run past the single-split cutoff (2011-10-26), so they reuse test invoices that the single split had already scored. The pooled result over the earlier folds avoids them.
- This run matches the multi-cutoff protocol of BasketWise 1.2.0 on the official UCI file. That protocol (monthly cutoffs from three months after the first invoice, expanding training windows, the 3 x 3 threshold grid chosen per fold, at most 2 products per rule, K = 3, 1,000 bootstrap resamples with seed 42, and the summaries reported here) was written into docs/PRODUCT_DECISIONS.md (decision 25) before it was first run on the real data. It was added after the single-split test period had been scored.

## 7. How to reproduce

```
python scripts/download_uci_online_retail.py   # official ZIP -> Data/external/uci_online_retail/
python -m market_basket uci-rolling            # this report and the rest of outputs/real_world/rolling/
```

With the same file and settings the results are identical on every run; only the generation time changes.
