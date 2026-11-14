# BasketWise — Interview preparation

These answers match the code in this repository. Before an interview, run the app, open each module named here and rerun the evaluation yourself, so you can explain every point from first-hand experience.

## What I built, what I used from libraries, and what AI assistance contributed

| Area | Who or what |
|---|---|
| Product idea, requirements, constraints, acceptance criteria | Me (the product brief and the review decisions) |
| Apriori, FP-Growth, one-hot encoding, rule metrics | **mlxtend** library (`TransactionEncoder`, `apriori`, `fpgrowth`, `association_rules`) |
| Community detection for merchandising groups | **NetworkX** library |
| Charts and app framework | **Plotly**, **Streamlit** |
| Project code: data validation, synthetic data generator, settings, rule ranking and explanations, recommender and fallback, evaluation framework, insights, reports, app pages, CLI, tests | Written for this project with substantial **AI coding assistance** (Claude Code), directed by my brief; I reviewed the results and am responsible for being able to explain them |
| Verification | 247 automated tests, a browser walkthrough and a clean-environment install, all run during development; rerun them yourself before relying on them |
| Real-world validation (UCI Online Retail, split by time) | Same AI-assisted process: download script, preprocessing, chronological evaluation, CLI, app tab, tests and report, reviewed by an independent review agent before the single final run; the dataset is by Chen (2015), CC BY 4.0, and is not redistributed |

**An honest one-line version:** "I specified and directed BasketWise and used an AI coding assistant to implement it. The mining algorithms come from mlxtend; the product logic, evaluation and app are this project's own code."

## Questions and answers

**1. What is BasketWise?**
A local web app that turns retail transaction data into product associations, explained recommendations, bundle ideas, and an honest check of whether those recommendations beat simply suggesting best sellers.

**2. What problem does it solve?**
Retailers have transaction data but rarely know which products are bought together. BasketWise answers "what should we suggest, place together or bundle?" without the user needing to know Python or association-rule metrics.

**3. Who are its target users?**
Small shop owners, supermarket managers, retail analysts, and merchandising and inventory teams. The personas in the requirements are assumptions; there was no user research and the product has no real users yet.

**4. Why did you build it?**
To show product thinking end to end on a real retail problem: define users and requirements, build something usable, and measure it honestly against a baseline, not just show attractive charts.

**5. What is market basket analysis?**
Finding sets of products that often appear in the same transaction, and expressing them as rules of the form "baskets with A often also contain B", each with a strength score.

**6. What are Apriori and FP-Growth?**
Both find every itemset whose support is above a threshold.
- **Apriori** grows itemsets one size at a time. It prunes candidates that have an infrequent subset, because every subset of a frequent set must also be frequent.
- **FP-Growth** compresses the transactions into a prefix tree and mines it without generating candidates.

Both return the same itemsets; the tests check this. In my benchmark, both took under a second at the app's default settings (Apriori was faster on the dense matrix), but at very low support with no size limit FP-Growth was 2.4–6.8× faster. That is why FP-Growth is the default. Timings depend on the computer.

**7. Why did you use mlxtend?**
It is a mature, widely used implementation. Rewriting these algorithms would add risk without adding product value. The value is in what surrounds them: validation, explanations, recommendations and evaluation. I verified that the library's metrics are correct with hand calculations and a brute-force recount in the tests.

**8. What are support, confidence and lift?**
- **Support:** the share of all transactions containing the products.
- **Confidence(A → B):** of the baskets with A, the share that also had B.
- **Lift:** confidence divided by B's overall support, so how many times more often B appears with A than usual.

Lift above 1 indicates a positive association in the data, not causation. Example from the demo: 41 % of baskets with Milk and Bread also had Butter, which is 3.8 times Butter's usual rate.

**9. How does the recommendation engine work?** (`market_basket/recommend.py`)
1. Take rules with a single product on the right-hand side and sort them by confidence (lift or support can be chosen instead).
2. For the basket, use each rule whose left-hand products are all in the basket, skipping products already in the basket.
3. Suggest each product once, through its best rule, with a reason and the rule's numbers.
4. If there are not enough suggestions, fill the remaining slots with best sellers, labelled as such.

**10. Why use a best-seller fallback?**
Some baskets have no applicable rule. Without a fallback, rules alone produced a suggestion in 98.8 % of evaluated cases. With the fallback this rises to 100 %, and Hit Rate@5 goes from 50.2 % to 53.1 %. The label keeps it honest: those suggestions are not based on the basket.

**11. How did you evaluate the recommendations?** (`market_basket/evaluation.py`)
1. Split the transactions 80/20 at random with a fixed seed.
2. Mine rules and best sellers on the training part only.
3. In each test basket with at least two products, hide each product in turn, recommend from the rest, and check whether the hidden product is in the top k.
4. Repeat on 5 or 10 splits and compare three recommendation methods with the best-seller baseline.

The tests guard against leakage: they check that mining only ever sees the training data, and that products seen only in the test set are never recommended. One honest caveat: I chose confidence as the default ranking after seeing these results on the same demo data, so its lead over the other rule methods is slightly optimistic. That selection effect is small next to the 20-point gap to the baseline.

**12. What is Hit Rate@3?**
The share of test cases where the hidden product appears in the top 3 suggestions. On the demo data:
- best sellers: 19.7 %
- confidence-ranked rules with fallback: 43.5 %

These are means over 5 splits, and the rule methods won on every split.

**13. Why can lift-based recommendations underperform?**
Lift rewards unusual pairings. A rare product that almost always appears with another gets a very high lift, even though few customers buy it. Ranking by lift therefore puts niche items first, which is less likely to match what a typical customer adds. On the demo, lift ranking scored 38.4 % at Hit Rate@3 against 42.5 % for confidence. Lift is still useful for finding cross-sell ideas.

**14. What were the most difficult problems?**
- **Making the evaluation honest.** Preventing leakage, choosing a baseline, and being clear that synthetic data with planted patterns flatters the method.
- **Messy input data.** Two layouts, alias column names, comma, semicolon, tab and pipe separators, blank and duplicate rows, capitalisation variants, and a pandas behaviour that silently misreads rows longer than the header. A generated demo file with known injected problems lets the tests check that every problem is counted exactly.
- **Bundle scoring.** Lift against independence grows with set size, so I used all-confidence to compare pairs and triples fairly. A later review showed that all-confidence alone could offer two popular but unrelated products as a bundle, so bundles must now also have lift above 1.
- **Small datasets.** Interface tests found a slider that crashed on very small catalogues, which I fixed.
- **Bugs a release audit found.** An independent review of the finished app found problems the tests had missed, because the interface tests ran one page at a time:
  - a second "Apply settings" in a row was ignored;
  - the basket was lost when changing pages;
  - an ignored product could still be suggested as a best seller;
  - returns were counted as purchases, and day/month dates were misread.

  Each fix has a regression test that fails on the old code. The lesson: test the journeys users take across pages, not just each page.

**15. What would you improve next?**
- Test on real retail data (with permission).
- Evaluate by time (train on past months, test on later ones).
- Add margin-aware ranking.
- Add significance filtering for rules.
- Build a simple A/B testing workflow to measure real impact.

**16. How would you measure real business impact?**
With a controlled experiment, comparing customers who see recommendations with those who do not.
- Primary metric: average basket value or items per basket.
- Secondary metrics: the attach rate of suggested products, and margin.
- Guardrails: checkout time, and returns or complaints.

Offline hit rate only shows relevance, not extra sales.

**17. How would you conduct an A/B test?**
1. Randomly assign customers or checkout sessions to control (no suggestions or best sellers) and treatment (BasketWise suggestions).
2. Fix the primary metric and the minimum effect worth detecting, and calculate the sample size.
3. Run for full weekly cycles.
4. Check guardrails, and compare with a confidence interval.
5. Watch for novelty effects before rolling out.

**18. What would you prioritise with one more month?**
1. Pilot with one retailer's real export, and fix whatever the data reveals.
2. Add a time-based evaluation.
3. Make the Evaluation Lab show results per category.
4. Design the A/B test with the retailer.

Real data and real feedback come before new features.

**19. How would you onboard a retailer?**
1. Ask for one month of billing export as a CSV.
2. Load it on the Data Explorer page and review the validation report together.
3. Agree thresholds on the Association Rules page.
4. Pick two or three cross-sell or bundle candidates to test in store.
5. Agree up front how success will be measured.

The downloadable templates and validation warnings are designed for exactly this first session.

**20. How would you decide whether to add a new feature?**
Ask which user problem it solves and how often it occurs, whether evidence shows the need (requests, usage data, failed tasks), what it costs to build and maintain, and how we would measure that it worked. For example, I would add margin-aware ranking only if retailers said that low-margin suggestions are a real problem.

**21. Isn't your 43 % circular, since you planted the associations yourself?**
Partly, yes, and I say so. The result shows that the pipeline recovers known structure without leakage and beats a sensible baseline on the same cases. It does not show how much better rules are on real data, where associations are weaker. That is why I added a separate, time-based validation on the real UCI Online Retail data (question 23). I keep the two experiments apart and never quote them as one benchmark.

**22. Why association rules and not collaborative filtering or embeddings?**
The input is anonymous baskets, which is what small retailers' billing exports contain. Collaborative filtering needs stable customer histories, and embeddings need much more data and lose explainability. Rules give a reason for every suggestion, which matters to a shop owner. With customer IDs and more data, I would compare item-based collaborative filtering against these rules in the same Evaluation Lab.

**23. Would it work on a real export, such as the UCI Online Retail dataset?**
Yes, and it is now measured, in a separate experiment. I learned rules and best sellers from invoices up to 25 October 2011, chose the thresholds on a validation period inside that training period, and tested on 26 October to 9 December 2011. On the same 116,322 leave-one-out cases, Hit Rate@3 was 7.6 % for the rules against 1.1 % for best sellers (95 % interval for the difference: +6.0 to +7.1 points). I always add three points. Both rates are low with 3,797 products, so the rules still missed 92 % of hidden products. Their weekly hit rate fell from 9.8 % to 5.7 % across the test period. And this is basket completion on one retailer's data, not next-purchase prediction or extra sales. See `docs/REAL_WORLD_VALIDATION.md`.

**24. How did you prevent leakage in the time-based test?**
- **Split by time:** every training invoice is earlier than every test invoice; the cutoff is a midnight, so no invoice or day is split.
- **Training data only:** rules, best sellers and the list of candidate products come from training invoices only.
- **Thresholds chosen on validation:** they were chosen on a validation period cut from the end of the training period, never on the test period.
- **Protocol fixed in advance:** the protocol was fixed before the test period was first scored.
- **Same cases for every method:** both methods are scored on exactly the same cases, and products that were never sold in training stay in the denominator as misses.
- **Tests:** they check each point, for example by recording what the mining and popularity functions receive.

**25. Why did you keep invoices without a customer ID, and why products by stock code?**
A basket is an invoice, and the methods never use customer IDs, so dropping those invoices would only lose data. There is an option to drop them. Stock codes are the stable identifier. In the data, 68 descriptions belong to several different codes, and many codes have several spellings, so matching by description would merge different products.

## Numbers to remember (real-world validation, UCI Online Retail; a separate experiment)

| Item | Value |
|---|---|
| Raw rows → valid invoices, products | 541,909 → 19,773 invoices, 3,797 products |
| Training / test periods | 1 Dec 2010 – 25 Oct 2011 (15,769 invoices) / 26 Oct – 9 Dec 2011 (4,004 invoices) |
| Chosen thresholds (on validation) | support 0.5 %, confidence 20 %, lift 1.2, at most 2 products per rule |
| Hit Rate@3: best sellers vs rules + fallback | 1.1 % vs 7.6 % on 116,322 cases (+6.6 points, 95 % interval +6.0 to +7.1) |
| Coverage: best sellers vs rules | 2.2 % vs 12.4 % of training products |

## Numbers to remember (demo data, default settings)

| Item | Value |
|---|---|
| Synthetic transactions, products, categories | 8,000 · 47 · 8 (seed 42) |
| Default thresholds | support 1 %, confidence 20 %, lift 1.2, at most 3 products |
| Frequent itemsets / rules | 265 / 262 |
| Hit Rate@3: best sellers vs confidence + fallback | 19.7 % vs 43.5 % (5 splits) |
| Hit Rate@5: best sellers vs confidence + fallback | 30.1 % vs 53.1 % |
| Coverage: best sellers vs rules | 26 % vs 77 % of products |
| Automated tests | 247 |

Always add: "the demo data is synthetic with planted associations, so real data would show a smaller gap."
