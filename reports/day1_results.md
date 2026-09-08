# Day 1 verification and findings

Completed 2026-09-07 using Python 3.12.14 and requirements-day1.txt.

## UCI training run

Command: `python scripts/train_rfm.py --input "data/Online Retail.xlsx" --as-of 2011-12-10`

- Raw transaction lines: 541,909
- Missing required fields or blank invoice: 135,080 removed
- Cancellation lines: 8,905 removed
- Invalid/nonpositive sales: 40 removed
- Exact full-row duplicates: 5,192 removed
- Eligible transaction lines: 392,692
- Customer profiles: 4,338
- Snapshot: 2011-12-10, exclusive cutoff

Removal counts are sequential and do not overlap. Monetary is positive purchase value in GBP, not net sales.

## Candidate comparison

| k | Sampled silhouette | Mean seed stability ARI | Smallest cluster share |
|---|---:|---:|---:|
| 2 | 0.425188 | 1.000000 | 38.45% |
| 3 | 0.329679 | 0.988593 | 17.57% |
| 4 | 0.330213 | 0.949797 | 16.85% |
| 5 | 0.319695 | 0.994688 | 7.93% |
| 6 | 0.318532 | 0.958266 | 6.80% |

The documented maximum-silhouette rule selected k=2. This is a coarse behavioral split,
not evidence that the business should run exactly two campaigns. A more granular k may be
useful after reviewing actionable differences, but four named segments are not claimed here.

| Cluster | Customers | Median recency (days) | Median invoices | Median spend (GBP) |
|---|---:|---:|---:|---:|
| 0 | 1,668 | 17 | 6 | 2,058.17 |
| 1 | 2,670 | 96 | 1 | 362.01 |

Cluster 0 contains more recent, repeat, higher-spend customers. Cluster 1 is generally less
recent with fewer invoices and lower spend. These descriptive differences could inform
retention experiments, but cannot establish churn risk or treatment effectiveness.

## Verification evidence

- `python -m pytest tests -q`: 3 passed.
- Tests verify an explicit cutoff excludes a later purchase, invoice frequency counts
  distinct invoices, gross spend is correct, full-row duplicate handling, schema errors,
  invalid feature rejection, and joblib round-trip predictions.
- Full UCI training reloaded the saved pipeline and reproduced all 4,338 assignments.
- Synthetic CLI training also completed: 17,907 lines, 1,200 customers, k=3 selected,
  silhouette 0.353657, and matching reloaded predictions. Synthetic artifacts are kept
  outside the UCI artifact directory and are not used as real findings.
- Seed checks test initialization sensitivity on the same data, not generalization,
  bootstrap stability, temporal stability, or causal retention lift.

## Remaining scope

Day 2 dashboard verification and deployment remain pending. No public app link exists yet.
