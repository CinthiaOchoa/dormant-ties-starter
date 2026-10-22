# Ranking Dormant Ties in a Timestamped Email Network
## Phase 2 — Progress Report

CSC 4740/6740 Data Mining — Team 3 (Undergraduate)
Cinthia Ochoa Torre
October 23, 2026

---

## 1. Summary

The full pipeline from the Phase 1 proposal now runs end to end on the SNAP files:
data audit, node alignment, the pair table, the four signals, the Reconnect Score, three
comparison procedures, ReconnectScoreNet, the ablation, the error analysis, the
one-sender case study, and the out-of-time check.

Two problems in the data forced changes to the plan. Both are documented in Section 6.

1. **The department labels use a different node numbering from the temporal edge list.**
   A direct join would have attached the wrong department to almost every person. I
   recovered the correct mapping by graph alignment and verified it on every edge.
2. **The trace has no email between day 525.5 and day 797.3.** The proposed decision
   time (day 600) falls inside that silence, so its label window is empty. The decision
   times move to day 400 (main) and day 460 (out-of-time), chosen by a rule that does not
   look at labels.

Main result on 972 held-out test pairs (115 resumptions, 11.8%): ReconnectScoreNet and
logistic regression tie (ROC-AUC 0.676, average precision 0.227 and 0.225). Both beat
the hand-weighted Reconnect Score (ROC-AUC 0.630, AP 0.201), and all three beat the
majority baseline (AP 0.118). The ranking holds at the out-of-time decision time. The
weighted score loses mainly because Overdue has the wrong sign: in this network, the
longer a pair has been silent, the *less* likely it is to write again.

## 2. Data and preprocessing

| Quantity | Value |
|---|---|
| Emails in the temporal file | 332,334 |
| Self-mail removed | 0 |
| People in the temporal file | 986 |
| People with a verified department | 965 (21 unresolved, see 2.1) |
| Departments | 42 |
| Last email in the contiguous record | day 525.5 |
| Emails after the gap (days 797.3–803.9) | 2,424 (never used) |

### 2.1 Node-ID alignment

SNAP publishes the department labels with the static release (`email-Eu-core`). The
temporal release numbers people differently. The evidence:

- Joining the two files on the integer ID, only 571 of the 24,929 directed temporal edges
  appear in the static graph. A random relabeling gives 638. The naive join is no better
  than chance.
- With self-loops removed, the static graph has exactly 24,929 directed edges, the same
  as the collapsed temporal graph. The 642 self-loops shift every degree by one.

I built the crosswalk (`src/crosswalk.py`) with joint color refinement (1-dimensional
Weisfeiler–Leman) on both directed graphs, then individualized and refined until the
match was one-to-one. The resulting mapping carries **all 24,929 of 24,929** temporal edges
onto static edges, so the two graphs are isomorphic under it.

954 people are identified uniquely. The other 32 fall into 14 small groups of structural
twins (two to four people with identical connection patterns) that no method using the
graph alone can tell apart. In 5 of those groups every member has the same department, so
11 more people get a safe department. The remaining **21 people** keep an unresolved
department, and, as the proposal specifies, any pair that includes one of them gets
S = 0. This affects 13 of the 6,480 dormant pairs.

### 2.2 Decision times

Emails per 30 days are steady (7,000–26,000) from day 0 to day 525, then drop to zero
until a 7-day burst at day 797. SNAP's "803 days" counts the gap.

Rule: t₀ is the latest multiple of 10 for which both label windows end inside the
contiguous record, t₀ + 2 · 60 ≤ 525.5. That gives **t₀ = 400** and an out-of-time
decision time of **460**, with label windows (400, 460] and (460, 520]. The 90-day gate
and 60-day horizon are unchanged. The pipeline checks the rule and stops if a label window
runs past the record.

### 2.3 Pair counts and class balance

| | t₀ = 400 | t₀ = 460 (out-of-time) |
|---|---|---|
| Pairs with an email at or before t₀ | 13,195 | 14,462 |
| Excluded by the 90-day gate (still active) | 6,715 | 6,612 |
| **Dormant pairs** | **6,480** | **7,850** |
| Resumed within 60 days | 769 (11.9%) | 1,215 (15.5%) |
| Mutual history (C = 80) | 2,620 | 3,362 |
| One-way history (C = 40) | 3,860 | 4,488 |

The dormant pairs at t₀ = 400 are split at random, stratified on the label:

| Fold | Pairs | Resumptions |
|---|---|---|
| Train | 4,536 | 538 |
| Validation | 972 | 116 |
| Test | 972 | 115 |

Resumptions are rare (about 1 in 8), so the report leads with precision, recall,
average precision and the precision–recall curve, as planned.

## 3. Signals

The four signals follow Section 3.3 of the proposal; `docs/data_dictionary.md` lists the
exact definitions. Every signal was checked to lie in 0–100.

| Signal | Mean | Median | Resumption rate across levels (training pairs) |
|---|---|---|---|
| S, similarity | 48.8 | 40 | rises from 5.9% (S = 0) to 18.1% (S = 100) |
| H, closeness | 35.2 | 20 | rises from 8.5% (H = 20) to 20.1% (H = 90) |
| C, history | 56.2 | 40 | 9.5% one-way vs 15.3% mutual |
| T, overdue | 84.4 | 100 | **falls** from 17.9% (T ≤ 25) to 10.4% (T = 100) |

Closeness bin edges, fit on the training pairs and frozen: 1–2 emails → 20, 3–4 → 35,
5–7 → 60, more than 7 → 90.

S, H and C move in the direction the proposal assumed. T does not. The proposal treated
a longer silence as more "overdue" for contact; in this data a longer silence means the
tie is more likely to be over. 63% of dormant pairs sit at the T = 100 cap (silent at
least 180 days).

## 4. Results

### 4.1 Test pairs, t₀ = 400

Thresholds for every procedure except the majority class are chosen on the validation
pairs by maximum F1. ROC-AUC and average precision do not depend on a threshold.

| Procedure | Accuracy | Precision | Recall | F1 | ROC-AUC | Avg. precision | Brier |
|---|---|---|---|---|---|---|---|
| Majority class | 0.882 | 0.000 | 0.000 | 0.000 | 0.500 | 0.118 | 0.104 |
| Weighted sum, s ≥ 48 (first pass) | 0.455 | 0.155 | 0.809 | 0.260 | 0.630 | 0.201 | 0.291* |
| Weighted sum, validation cutoff s ≥ 41.7 | 0.341 | 0.138 | 0.870 | 0.238 | 0.630 | 0.201 | 0.291* |
| Logistic regression (L2) | 0.736 | 0.218 | 0.478 | **0.300** | **0.676** | 0.225 | **0.100** |
| ReconnectScoreNet (4–16–8–1) | 0.718 | 0.205 | 0.478 | 0.287 | **0.676** | **0.227** | **0.100** |

\* The weighted sum is a ranking score, not a probability. Its Brier score uses s / 100
and is shown only for completeness.

Over five training seeds, ReconnectScoreNet scores ROC-AUC 0.677 ± 0.002 and average
precision 0.229 ± 0.008, so the tie with logistic regression is not a seed effect.

![Precision-recall curves on the test pairs](figures/pr_curve_test.png)

The validation-tuned cutoff for the weighted sum (41.7) is lower than the first-pass
cutoff (48). On test, the first-pass cutoff has the better F1; both are reported, as
planned.

### 4.2 Out-of-time check, t₀ = 460

The same fitted models and thresholds are applied to the 7,850 dormant pairs at day 460.
Nothing was tuned on these pairs.

| Procedure | Precision | Recall | F1 | ROC-AUC | Avg. precision | Brier |
|---|---|---|---|---|---|---|
| Majority class | 0.000 | 0.000 | 0.000 | 0.500 | 0.155 | 0.132 |
| Weighted sum, s ≥ 48 | 0.182 | 0.735 | 0.292 | 0.602 | 0.216 | 0.293* |
| Logistic regression (L2) | 0.263 | 0.515 | 0.348 | 0.669 | 0.285 | 0.125 |
| ReconnectScoreNet | 0.252 | 0.537 | 0.343 | 0.670 | 0.292 | 0.125 |

The order of the procedures is the same as at day 400. ROC-AUC drops by less than 0.01
for the learned models and by 0.03 for the weighted sum. Precision and AP are higher
because the base rate is higher (15.5%).

### 4.3 Ablation

Each signal is dropped in turn. For the weighted sum its term is removed; logistic
regression and the network are refit on the other three signals. "Top partner changed"
is the share of 100 randomly sampled people (each with at least three dormant partners)
whose highest-ranked dormant partner changes.

| Dropped | Δ AP weighted sum | Δ AP logistic | Δ AP network | Top partner changed (weighted sum) | Top partner changed (network) |
|---|---|---|---|---|---|
| S | −0.037 | −0.025 | −0.019 | 33% | 24% |
| H | −0.049 | −0.030 | −0.029 | 51% | 50% |
| C | −0.013 | +0.001 | −0.002 | 14% | 19% |
| T | **+0.005** | −0.013 | −0.013 | 32% | 37% |

Closeness carries the most signal, then similarity. History adds almost nothing once the
others are present (C and H are correlated: a mutual thread is also a longer thread).
Removing Overdue *improves* the weighted sum, because its positive weight pushes the
longest-silent pairs up. The learned models use T with a negative weight (logistic
coefficient −0.33 vs +0.68 for S and +1.24 for H), which is why dropping T hurts them.

### 4.4 Error analysis

The planned check targets false positives with high overdue and near-zero similarity
(T ≥ 75 and S ≤ 20): pairs that silence alone would rank high.

| Procedure | False positives on test | With T ≥ 75 and S ≤ 20 | Share |
|---|---|---|---|
| Weighted sum (validation cutoff) | 626 | 66 | 10.5% |
| ReconnectScoreNet | 214 | 15 | 7.0% |

This pattern covers 26.5% of all test negatives, so both procedures already flag it far
less than its base rate. The similarity signal does hold these pairs down. The weighted
sum's larger problem is the volume of false positives: its cutoff must be low to reach a
useful recall, and the T term lifts long-dead ties above the cutoff. Rows are in
`reports/error_analysis_fp.csv`; the full write-up is planned for Phase 3.

### 4.5 Case study

The case-study sender is person 16: 188 emails sent before day 400, the closest to the
median (187) among people with at least 8 dormant partners. Person 16 has 35 dormant
partners at day 400, and 5 of them wrote again within 60 days. The full table, with
department, counts, days silent, the four signals, the score, the network probability
and the label for every partner, is `reports/case_study_sender.csv`.

Top five partners by Reconnect Score:

| Rank | Partner | Same dept. | Emails (16→p / p→16) | Days silent | S | H | C | T | Score | P(net) | Resumed |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 249 | yes | 4 / 8 | 159.4 | 60 | 90 | 80 | 77.2 | 74.9 | 0.216 | no |
| 2 | 502 | no | 1 / 2 | 364.6 | 80 | 35 | 80 | 100.0 | 72.8 | 0.117 | no |
| 3 | 724 | yes | 2 / 2 | 271.6 | 80 | 35 | 80 | 100.0 | 72.8 | 0.117 | no |
| 4 | 423 | yes | 2 / 5 | 175.7 | 60 | 60 | 80 | 95.2 | 71.0 | 0.142 | no |
| 5 | 832 | no | 4 / 5 | 133.7 | 60 | 90 | 80 | 48.5 | 69.2 | 0.253 | **yes** |

Partners 502 and 724 show the Overdue problem directly: a year-long silence after three
or four emails puts them second and third under the weighted sum, while the network
ranks them tied at 16th of 35.

**Hand check.** Three rows (ranks 1, 18 and 35) were recomputed from the raw gzip file by
a separate routine that uses only the Python standard library. All S, H, C, T and score
values match the pipeline exactly (`reports/case_study_check.json`).

**Stability.** At day 460, four of person 16's top five at day 400 are still dormant, and
all four are still in the top five. Across all 707 people whose top partner at day 400 is
still dormant at day 460, 66% keep the same top partner.

## 5. Interpretation so far

- The four signals carry real but modest information: the best ROC-AUC is about 0.68.
- The network only ties logistic regression. As the proposal anticipated, four tabular
  signals with a roughly monotone relationship to the label need no deeper model.
- The hand-weighted score trails the learned models, and the reason is specific:
  Overdue enters with the wrong sign. Phase 3 will report a refit of the weights (or a
  reversed T) as a clearly labeled post-hoc variant, alongside the original score, and
  the original score will stay in the table unchanged.

## 6. Departures from the Phase 1 proposal

| Proposal | Phase 2 | Reason |
|---|---|---|
| Check that node IDs align, then join | Recovered a crosswalk by graph isomorphism; 21 people unresolved | IDs do not align; the naive join is no better than chance (2.1) |
| t₀ ≈ 600, second t₀ = 660 | t₀ = 400, second t₀ = 460 | No email between day 525.5 and 797.3 (2.2) |
| Trace of 803 days | Contiguous record of 525.5 days | Same gap |
| S defined from the sender's side | S = max of both directions | Pairs are unordered; both directions are kept in the data (`S_uv`, `S_vu`) |
| Closeness bins from training quantiles | Tertiles of counts above 2 on training pairs: 3–4, 5–7, >7 | Makes "quantiles" concrete; one or two emails stay at 20 |
| Random 70/15/15 split | Random split stratified on the label | Keeps the rare positive class balanced across folds |
| Threshold tuned on validation for the weighted sum | Also for logistic regression and the network (max F1) | Same rule for every procedure |
| 4–16–8–1 network | Same, with hidden biases initialized to 0.1 | With default initialization one of five seeds collapsed to a constant output (dead ReLUs) |

Pairs at the out-of-time decision time can also appear in the day-400 training fold.
This does not leak labels: a pair dormant at day 460 sent no email in (370, 460], so its
day-400 label was 0, and no day-460 label window overlaps training data.

## 7. Remaining work (Phase 3, due November 20)

- Full error analysis write-up with example pairs.
- Post-hoc variant of the Reconnect Score with refit or sign-corrected weights, reported
  separately from the pre-specified score.
- Methodology section, final metric tables, and the recorded walkthrough.

## Reproducing the results

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py   # skips files already in data/raw/
python src/run_all.py
```

All numbers in this report come from the files in `reports/`, produced by
`src/run_all.py` with seed 4740.
