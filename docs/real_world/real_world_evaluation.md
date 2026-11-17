# BasketWise real-world chronological validation (UCI Online Retail)

Generated 2026-11-14 06:03 UTC by BasketWise 1.2.0. Every number in this report was computed by the code in this repository from the file named in section 1; none was typed by hand.

**Question.** When part of a later invoice is known, do association rules learned only from earlier invoices suggest the invoice's remaining product more often than simply suggesting best sellers?

**Answer for this data.** On this test period the association-rule recommender found the hidden product in the top 3 more often than the best sellers did (7.6% against 1.1%, +6.6 points; 95% interval +6.0 to +7.1 points). In absolute terms, the hidden product was not among the 3 suggestions in 92.4% of cases for the association-rule recommender and in 98.9% for the best sellers.

**What this does not show:** next-purchase or next-order prediction (the hidden product comes from the same invoice); extra sales, revenue or any causal effect of showing recommendations (that needs an A/B test); real customer adoption (no recommendation was shown to any customer); how other shops or other periods would behave (the data comes from one online retailer).

This is a separate experiment from the synthetic benchmark in `docs/EVALUATION_REPORT.md`: the data, the split (by time instead of random) and the settings differ, so the two sets of numbers must not be combined or compared as one benchmark.

## 1. Dataset source and attribution

| item | value |
|---|---|
| Dataset | Online Retail, UCI Machine Learning Repository |
| Citation | Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BW33. |
| Licence | Creative Commons Attribution 4.0 International (CC BY 4.0) — https://creativecommons.org/licenses/by/4.0/legalcode |
| Source page | https://archive.ics.uci.edu/dataset/352/online+retail |
| Download URL | https://archive.ics.uci.edu/static/public/352/online+retail.zip |
| Retrieved (UTC) | 2026-11-14T05:10:06+00:00 |
| File used | online_retail.csv (SHA-256 7615e19743f2c3e8e641375743c276dbb0ed64a70cf210464ce72c335db3d221) |
| Official spreadsheet SHA-256 | 43465a06f2ccf7c8b5bd2892bc7defb52f97487934fe93b16ae4c3936424676d |

The raw data is not stored in this repository. This file was made by `scripts/download_uci_online_retail.py`, which records the source, retrieval time and checksums in `download_manifest.json`, and it is the same file the published results in `docs/real_world/` were computed from.

## 2. Data-quality findings (raw table)

| check | observed |
|---|---|
| Rows | 541,909 |
| Invoices (all) | 25,900 |
| Cancellation invoices (number starts with C) | 3,836 (9,288 rows, of which 9,288 have a negative quantity) |
| Other non-numeric invoices | 3 (A563185, A563186, A563187) |
| Stock codes (as written / ignoring case) | 4,070 / 3,958 |
| Distinct descriptions | 4,202 |
| Stock codes with more than one description | 612 |
| Descriptions shared by several stock codes | 68 |
| Customers (distinct CustomerID) | 4,372 |
| Rows without CustomerID | 135,080 |
| Missing or placeholder descriptions | 1,501 |
| Note-like descriptions (lower case, no word in capitals) | 1,377 (most common: 'Manual', 'check', 'Next Day Carriage', 'Discount', 'damages') |
| Exact duplicate rows | 5,268 |
| Negative quantities (all / outside cancellations) | 10,624 / 1,336 |
| Zero quantities | 0 |
| Zero / negative unit prices | 2,515 / 2 |
| Invalid or missing dates | 0 |
| Date range | 2010-12-01T08:26:00 to 2011-12-09T12:50:00 (305 days with invoices) |
| Service and accounting lines (postage, fees, ...) | 3,028 |
| Countries | 38 |

Rows per country (top 10): United Kingdom 495,478, Germany 9,495, France 8,557, EIRE 8,196, Spain 2,533, Netherlands 2,371, Belgium 2,069, Switzerland 2,002, Portugal 1,519, Australia 1,259.

Regular invoices per month: 2010-12 1,699, 2011-01 1,216, 2011-02 1,174, 2011-03 1,665, 2011-04 1,504, 2011-05 1,848, 2011-06 1,683, 2011-07 1,657, 2011-08 1,456, 2011-09 1,994, 2011-10 2,275, 2011-11 3,021, 2011-12 869.

## 3. Preprocessing decisions

- **Baskets are invoices.** Products are identified by stock code, trimmed and in upper case: 112 of the 117 lower-case codes (such as 85123a) also appear in upper case, and 110 of those share a description with their upper-case twin. Descriptions are display names only (the most frequent description of a code is used), because 68 descriptions belong to several different codes.
- **Returns and cancellations are never purchases.** Every row of a cancellation invoice is removed, and so is every row with a zero or negative quantity elsewhere (returns or stock adjustments).
- **Non-sale invoices and service lines are removed:** invoice numbers with another letter (bad-debt adjustments), and stock codes for postage, carriage, fees, commissions, manual entries, discounts, samples and gift vouchers (22016, 23444, AMAZONFEE, B, BANK CHARGES, C2, CRUK, D, DOT, M, POST, S, GIFT_0001_*).
- **Rows with a missing, zero or negative unit price are removed.** The raw table has 2,515 zero-price rows (40 of them with a customer ID) and 2 negative prices; rows already removed for an earlier reason are counted only once.
- **Missing CustomerID is kept**: a basket is an invoice, and customer IDs are not used by the methods.
- **Duplicates:** exact duplicate rows are removed; other repeated lines of the same product in one invoice are merged into one basket line (quantities added). Basket analysis only uses whether a product is in the basket.
- **Dates** must be ISO date-times (as written by the download script); anything else is reported as invalid, never guessed. An invoice's time is its earliest line (42 invoices have lines with more than one timestamp, at most 1 minute apart).
- Each removed row keeps its first applicable reason in `removed_rows.csv.gz`, with its line number in the CSV.

| step | count |
|---|---|
| Original rows | 541,909 |
| Removed: missing invoice number | 0 |
| Removed: cancellation invoice (invoice number starts with C) | 9,288 |
| Removed: non-sale invoice (invoice number with another letter, e.g. A = bad-debt adjustment) | 3 |
| Removed: missing or invalid invoice date | 0 |
| Removed: missing stock code | 0 |
| Removed: non-merchandise line (postage, carriage, fees, discounts, manual entries, samples, vouchers) | 2,440 |
| Removed: missing, zero or negative quantity (return or stock adjustment) | 1,336 |
| Removed: missing, zero or negative unit price | 1,161 |
| Valid purchase rows | 527,681 |
| Removed: exact duplicate of an earlier row | 5,221 |
| Merged: repeated product in the same invoice (merged into one basket line) | 5,250 |
| Basket lines (invoice-product pairs) | 517,210 |
| Valid invoices (transactions) | 19,773 |
| Invoices with two or more products | 18,267 |
| Unique products (stock codes) | 3,797 |
| Unique customers | 4,334 |
| Valid invoices without a customer ID (kept) | 1,371 |

## 4. Training and test periods

| item | value |
|---|---|
| Cutoff (first moment of the test period) | 2011-10-26T00:00:00 |
| How the cutoff was chosen | start of the day containing the first invoice of the latest 20% of invoices |
| Training period | 2010-12-01T08:26:00 to 2011-10-25T18:39:00 |
| Training invoices | 15,769 |
| Test period | 2011-10-26T08:30:00 to 2011-12-09T12:50:00 |
| Test invoices (with 2+ products) | 4,004 (3,728) |
| Share of invoices in the test period | 20.2% |

Every training invoice is earlier than every test invoice, and no invoice is split between the periods.

## 5. Evaluation method

1. Rules and best sellers are learned from training-period invoices only.
2. In every test invoice with at least two products, each product is hidden once. The other products of that invoice are the query, and each method suggests up to 3 products that are not already in the query.
3. All methods are scored on exactly the same 116,322 cases from 3,728 test invoices (1,999 distinct customers among invoices with a customer ID).
4. **Candidate universe:** the 3,737 products bought in the training period; nothing else can be suggested. **Denominator:** every case above, including cases whose hidden product was never sold in training (a miss for every method).

Metrics in simple words:

- **Hit Rate@3**: the share of cases where the hidden product is among the 3 suggestions.
- **Precision@3**: the share of the 3 suggestion slots that hold the hidden product, averaged over cases. With one hidden product per case it is Hit Rate@3 divided by 3, so it can be at most 33.3%.
- **Recall@3**: the share of hidden products that were suggested. With one hidden product per case it equals Hit Rate@3.
- **Coverage**: the share of the candidate universe that a method suggested at least once.
- **Availability**: the share of cases with at least one suggestion.
- **Invoice-averaged Hit Rate@3**: the hit rate of each invoice, averaged over invoices, so that large invoices do not dominate.
- **95% intervals**: from 1,000 bootstrap resamples of whole test invoices (seed 42), the same resamples for every method.

## 6. Threshold selection (validation period inside the training period)

Candidate settings were compared on a validation period from 2011-09-05T09:07:00 to 2011-10-25T18:39:00 (3,188 invoices, 88,734 cases), with rules learned from 2010-12-01T08:26:00 to 2011-09-04T16:12:00. The test period was not used. On validation, the best-seller baseline scored 1.2%.

| min_support | min_confidence | min_lift | rules | validation hit rate@3 | validation coverage | validation rank | status | chosen |
|---|---|---|---|---|---|---|---|---|
| 0.50% | 10% | 1.2 | 12,104 | 9.27% | 13.19% | 2 | compared |  |
| 0.50% | 20% | 1.2 | 5,948 | 9.27% | 13.19% | 1 | compared | yes |
| 0.50% | 30% | 1.2 | 2,629 | 9.25% | 12.71% | 3 | compared |  |
| 1.00% | 10% | 1.2 | 1,931 | 7.40% | 5.93% | 5 | compared |  |
| 1.00% | 20% | 1.2 | 1,382 | 7.41% | 5.93% | 4 | compared |  |
| 1.00% | 30% | 1.2 | 772 | 7.35% | 5.71% | 6 | compared |  |
| 2.00% | 10% | 1.2 | 218 | 4.07% | 1.58% | 7 | compared |  |
| 2.00% | 20% | 1.2 | 198 | 4.05% | 1.53% | 8 | compared |  |
| 2.00% | 30% | 1.2 | 172 | 3.99% | 1.47% | 9 | compared |  |

## 7. Results on the test period

Chosen setting: minimum support 0.50%, minimum confidence 20%, minimum lift 1.2, at most 2 products per rule (FP-Growth). Learned from the training period: 8,290 frequent itemsets and 5,516 rules.

The association-rule recommender below is BasketWise's default: rules ranked by confidence, with training-period best sellers filling any free slots. The best-seller baseline suggests the training period's most popular products that are not already in the basket.

| Metric | Best-seller baseline | Association-rule recommender |
|---|---|---|
| Hit Rate@3 | 1.1% | 7.6% |
| Precision@3 | 0.4% | 2.5% |
| Recall@3 | 1.1% | 7.6% |
| Coverage | 2.2% | 12.4% |
| Availability | 100.0% | 100.0% |
| Invoice-averaged Hit Rate@3 | 1.4% | 12.2% |
| Hit Rate@3 95% interval | 1.0% to 1.2% | 7.1% to 8.1% |
| Evaluated cases | 116,322 | 116,322 |

All methods (rules-only methods leave a slot empty when no rule applies):

| method | hit rate@3 | precision@3 | recall@3 | availability | coverage | invoice-averaged hit rate@3 | hit rate@3, hidden product seen in training | difference vs baseline (95% interval) |
|---|---|---|---|---|---|---|---|---|
| Best-seller baseline | 1.1% | 0.4% | 1.1% | 100.0% | 2.2% | 1.4% | 1.1% | - |
| Rules ranked by confidence | 7.6% | 2.5% | 7.6% | 99.3% | 12.4% | 12.1% | 7.7% | +6.0 to +7.1 points |
| Rules ranked by lift | 6.0% | 2.0% | 6.0% | 99.3% | 14.9% | 10.1% | 6.0% | +4.4 to +5.4 points |
| Rules by confidence + best-seller fallback | 7.6% | 2.5% | 7.6% | 100.0% | 12.4% | 12.2% | 7.7% | +6.0 to +7.1 points |

Hit Rate@3 per test week (the first and last weeks can be partial):

| week starting | invoices | cases | best-seller baseline | association-rule recommender |
|---|---|---|---|---|
| 2011-10-24 | 338 | 8,525 | 0.9% | 9.8% |
| 2011-10-31 | 495 | 15,768 | 1.6% | 7.5% |
| 2011-11-07 | 596 | 18,353 | 1.1% | 8.8% |
| 2011-11-14 | 641 | 19,978 | 1.0% | 8.1% |
| 2011-11-21 | 571 | 18,800 | 1.0% | 6.8% |
| 2011-11-28 | 608 | 17,939 | 1.0% | 7.8% |
| 2011-12-05 | 479 | 16,959 | 1.0% | 5.7% |

Examples of rules learned from the training period (strongest confidence first):

| if the basket has | suggest | support | confidence | lift |
|---|---|---|---|---|
| HERB MARKER ROSEMARY (22917) | HERB MARKER THYME (22916) | 0.012 | 0.933 | 70.036 |
| REGENCY TEA PLATE PINK (23172) | REGENCY TEA PLATE GREEN (23171) | 0.015 | 0.928 | 46.738 |
| HERB MARKER THYME (22916) | HERB MARKER ROSEMARY (22917) | 0.012 | 0.924 | 70.036 |
| HERB MARKER CHIVES (22921) | HERB MARKER PARSLEY (22918) | 0.011 | 0.919 | 69.666 |
| HERB MARKER PARSLEY (22918) | HERB MARKER THYME (22916) | 0.012 | 0.909 | 68.231 |
| HERB MARKER PARSLEY (22918) | HERB MARKER MINT (22919) | 0.012 | 0.909 | 67.270 |
| HERB MARKER CHIVES (22921) | HERB MARKER THYME (22916) | 0.011 | 0.908 | 68.190 |
| HERB MARKER CHIVES (22921) | HERB MARKER MINT (22919) | 0.011 | 0.903 | 66.830 |
| SET/6 COLLAGE PAPER CUPS (21084) | SET/6 COLLAGE PAPER PLATES (21090) | 0.009 | 0.902 | 74.456 |
| BISCUITS SMALL BOWL LIGHT BLUE (84596L) | SMALL DOLLY MIX DESIGN ORANGE BOWL (84596B) | 0.010 | 0.902 | 33.775 |

## 8. Evaluation settings

| setting | value |
|---|---|
| Recommendations per case (K) | 3 |
| Test share / fixed cutoff | 20% |
| Validation share of the training period | 20% |
| Candidate minimum supports | 0.50%, 1.00%, 2.00% |
| Candidate minimum confidences | 10%, 20%, 30% |
| Candidate minimum lifts | 1.2 |
| Chosen minimum support / confidence / lift | 0.50% / 20% / 1.2 |
| Maximum products per rule | 2 |
| Algorithm | FP-Growth (mlxtend) |
| Recommender ranking | confidence, with best-seller fallback (BasketWise default) |
| Bootstrap resamples / seed | 1,000 / 42 |
| Customer IDs required | no |

## 9. Limitations

- One retailer, one split: the data covers 2010-12-01 to 2011-12-09 for one online retailer (UCI describes it as a UK-based online gift retailer with many wholesale customers), evaluated on a single chronological split. The results say nothing certain about other shops or other periods.
- This is offline basket completion, not next-purchase prediction: the hidden product comes from the same invoice, and the rest of that invoice is used as context. In a live shop the order in which products are added is unknown, so less context may be available.
- A hit means the suggestion matched a product the customer bought anyway. It does not show extra sales, revenue or any causal effect; that needs a controlled experiment such as an A/B test.
- Large invoices contribute many cases (the largest test invoice gives 1,106 cases; the median gives 17). The invoice-averaged hit rate, which weights every invoice equally, is reported as a check.
- 1.1% of the test cases hide a product that was never sold in the training period. No method can suggest such a product; these cases stay in the denominator for every method.
- Cancellations are removed as rows, but the data does not link a cancellation to its original order, so an order that was later cancelled still counts as a basket.
- Rules have one product on each side (at most 2 products per rule). On this training period, allowing three products at the chosen support would exceed BasketWise's pre-mining safety limit of 200,000 itemsets, so longer rules were not possible with these settings.
- The chosen minimum support (0.50%) is the lowest value in the grid, so a value outside the grid might have scored better on validation.
- The thresholds were chosen from a small grid on a validation period at the end of the training period. Other values, longer rules or other ranking methods were not tried, so better settings may exist.
- 1,371 valid invoices have no customer ID. They are kept, because a basket is an invoice; customer-level questions cannot be answered with them.
- The confidence intervals resample test invoices only. They do not cover changes between periods, such as seasonal peaks inside or outside the test period.
- This run matches the default protocol of BasketWise 1.2.0 on the official UCI file (latest 20% of invoices as the test period, 20% of the training invoices for validation, the 3 x 3 threshold grid, at most 2 products per rule, K = 3, 1,000 bootstrap resamples with seed 42). In this project that protocol was fixed before the test period was first scored and was not changed after seeing test results (docs/PRODUCT_DECISIONS.md, decisions 20 to 22).

## 10. How to reproduce

```
python scripts/download_uci_online_retail.py   # official ZIP -> Data/external/uci_online_retail/
python -m market_basket uci-prepare            # data-quality report and preprocessing summary
python -m market_basket uci-evaluate           # this report and the other files in outputs/real_world/
```

The same evaluation can be run from the Evaluation Lab page of the app (tab 'Real-world validation'). With the same file and settings the results are identical on every run; only the generation time changes.
