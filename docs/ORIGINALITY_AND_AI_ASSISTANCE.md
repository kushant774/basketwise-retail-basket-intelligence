# Originality and AI-assistance disclosure

## Repository history

This repository was created in 2025. Its first two commits are kept as they were; the Git history was not rewritten, and neither commit dates nor authors were changed. A November 2026 audit found that those commits contain material that was not written for this project:

| File in earlier commits | Source found in the audit |
|---|---|
| `Data/store_data.csv` | Byte-identical to files in the public repositories ashishpatel26/Market-Basket-Analysis and Debasishsaha123/MARKET-BASKET-ANALYSIS; the dataset's original source and licence are unknown |
| `Market Basket Analysis Using apyori  package.ipynb` | Byte-identical to ashishpatel26/Market-Basket-Analysis; its code follows the Stack Abuse tutorial "Association Rule Mining via Apriori Algorithm in Python" (Usman Malik), and its metric text is quoted from a 2016 KDnuggets tutorial |
| `Market Basket Analysis Using mlxtend package.ipynb` | Byte-identical to ashishpatel26/Market-Basket-Analysis; it is based on the mlxtend user guide example |
| `Market_Basket_Analysis(Data Gathering and Visualization).ipynb` and `.py` | Byte-identical to Debasishsaha123/MARKET-BASKET-ANALYSIS |
| `README.md` (as committed in 2025) | Byte-identical to ashishpatel26/Market-Basket-Analysis |

## Current implementation

All of the material listed above has been removed from the current working tree. It was not moved into another folder, and the current application, tests, charts and screenshots do not use it.

The current BasketWise code was written for this project. This includes:
- the package, the app and the scripts
- the synthetic data generator and the generated dataset
- the tests, the notebook and the documentation

It uses established open-source libraries for standard algorithms:
- **mlxtend**: `TransactionEncoder`, `apriori`, `fpgrowth`, `association_rules`
- **NetworkX**: greedy modularity community detection
- **pandas, NumPy, Plotly and Streamlit**
- **openpyxl**: reading the official UCI spreadsheet (real-world validation only)

These libraries are used, not reimplemented, and the documentation says so wherever they appear. The algorithms (Apriori, FP-Growth, association-rule metrics) are standard and widely documented. The five-basket example in the tests is a well-known textbook example (Tan, Steinbach and Kumar), and it is cited.

## Who contributed what

| Contribution | Source |
|---|---|
| **Product decisions by the author** | The product idea and name; target users; the required features and pages; the constraints (local, no LLM features, synthetic data, removal of copied material, honest disclosure); the evaluation requirements (best-seller baseline, Hit Rate@3 and @5, no leakage, fixed seeds); and the review and acceptance of each stage |
| **AI-assisted implementation** | Detailed design and code for the package, the app, the tests and the docs, written by an AI coding assistant to the author's specification. Implementation choices made in that process (for example all-confidence for bundles, FP-Growth as the default, confidence as the default ranking) are documented, with their evidence, in `docs/PRODUCT_DECISIONS.md` |
| **Algorithms provided by mlxtend** | `TransactionEncoder`, `apriori`, `fpgrowth`, `association_rules` (the metric calculations). NetworkX provides community detection |
| **Original project code** | Everything else in `market_basket/`, `ui/`, `app.py`, `scripts/` and `tests/`: synthetic data, validation, ranking, explanations, recommendations, fallback, evaluation, insights, reports and the interface. It was written for this project, not copied |

## AI assistance

The current implementation was developed with substantial help from an AI coding assistant (Claude Code) in sessions directed by the repository author. In those sessions:
- **The author** provided the product brief, requirements, constraints and acceptance criteria. The author also reviewed the progress reports and decided what to keep.
- **The assistant** did the following:
  - audited the repository
  - wrote most of the code, tests and documentation
  - ran the tests, the evaluation, the benchmark and the browser checks
  - reported the results, including failures and limitations
- **Responsibility.** The author is responsible for understanding, verifying and being able to explain every part before presenting it as their work. `docs/INTERVIEW_PREPARATION.md` separates what was built, what comes from libraries, and what AI assistance contributed.

### Release audit (November 2026)

At the author's request, the finished project was audited in AI-assisted sessions. Separate review agents checked algorithm correctness, code quality and data handling, and the Streamlit interface, and each finding was reproduced before it was fixed. The resulting fixes and their regression tests are described in `docs/TESTING_REPORT.md`; the same assistant implemented them, and the author authorised the changes and their publication. The audit independently recomputed every rule metric and reproduced every published evaluation number. It confirmed that the mlxtend-provided algorithms were used correctly, so they were not changed.

No result in the documentation is hand-written. Every number comes from running the code in this repository.

### Real-world validation extension (November 2026)

At the author's request, a chronological validation on real data was added in a further AI-assisted session (Claude Code), on a separate branch for the author's review. The assistant inspected the dataset, wrote the download script, the preprocessing, the time-based evaluation, the CLI commands, the Evaluation Lab tab, the tests and the documentation, and ran the evaluation. The author set the requirements: an official source, a chronological split, no leakage, the same cases for every method, honest reporting, and no changes to the published synthetic benchmark.

**Third-party data.** The real-world validation uses the *Online Retail* dataset: Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5BW33>, licensed under Creative Commons Attribution 4.0 International (CC BY 4.0). The dataset is not included in this repository. It is downloaded from the UCI Machine Learning Repository by `scripts/download_uci_online_retail.py`. The committed files in `docs/real_world/` are summaries and evaluation results derived from it, with this attribution. The licence of BasketWise itself has not been chosen yet; that is the author's decision and is not implied by the dataset's licence.
