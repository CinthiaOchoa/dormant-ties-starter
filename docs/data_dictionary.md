# Data Dictionary

Project: Ranking Dormant Ties in a Timestamped Email Network
Author: Cinthia Ochoa Torre — CSC 4740/6740, Team 3

All times are days since the first email in the temporal file. `t0` is the decision time
(400 for the main tables, 460 for the out-of-time tables).

## Raw inputs (`data/raw/`)

| File | Columns | Notes |
|---|---|---|
| `email-Eu-core-temporal.txt.gz` | `src`, `dst`, `ts` (seconds) | 332,334 emails, 986 people, no self-mail. Emails stop at day 525.5 and resume for 7 days at day 797.3. |
| `email-Eu-core.txt.gz` | `src`, `dst` | Static release: 25,571 directed edges, 642 of them self-loops. Used only to build the crosswalk. |
| `email-Eu-core-department-labels.txt.gz` | `node`, `department` | 1,005 people, 42 departments, numbered in the **static** ID space. |

## `data/processed/node_crosswalk.csv`

| Column | Type | Meaning |
|---|---|---|
| `temporal_id` | int | Person ID in the temporal file. |
| `static_id` | int | Matching ID in the static file and department labels. |
| `structural_twin` | bool | Color refinement could not tell this person apart from one to three others. |
| `department_resolved` | bool | False when the twins carry different departments, so the match is ambiguous. |
| `department` | int / empty | Department used by the pipeline; empty when unresolved. |

## `data/processed/pairs_main.csv` and `pairs_oot.csv`

One row per unordered pair `(u, v)`, `u < v`, that exchanged at least one email at or
before `t0` and has been silent for at least 90 days.

| Column | Type | Meaning |
|---|---|---|
| `u`, `v` | int | Temporal person IDs, `u < v`. |
| `first_email_day`, `last_email_day` | float | First and last email between the pair at or before `t0`. |
| `emails_before_t0` | int | Total emails between the pair at or before `t0`. |
| `u_to_v`, `v_to_u` | int | Emails in each direction at or before `t0`; zero when the direction is absent. |
| `future_emails` | int | Emails in either direction in `(t0, t0 + 60]`. Used only for the label. |
| `label` | 0/1 | 1 when `future_emails > 0`. |
| `days_silent` | float | `t0 - last_email_day`. |
| `dormant` | bool | `days_silent >= 90` (always true in these files). |
| `t0` | int | Decision time of the row. |
| `dept_u`, `dept_v` | int / empty | Departments from the crosswalk. |
| `dept_known` | bool | Both departments resolved. |
| `same_department` | bool | `dept_u == dept_v`. |
| `S_uv`, `S_vu` | 0–100 | Similarity from u's side and from v's side (see below). |
| `S` | 0–100 | `max(S_uv, S_vu)`. |
| `H` | 20/35/60/90 | Closeness bin of `emails_before_t0`: 1–2 → 20, 3–4 → 35, 5–7 → 60, more than 7 → 90. Bin edges are fit on training pairs at `t0 = 400` and frozen. |
| `C` | 40/80 | History: 80 if both directions are non-zero, else 40. |
| `T` | 0–100 | Overdue: `min(100, max(0, 100 (days_silent - 90) / 90))`. |
| `score` | 0–93.5 | Reconnect Score `0.35 S + 0.25 H + 0.20 C + 0.20 T`. |
| `fold` | str | `train` / `val` / `test` (main) or `oot`. |

**Similarity from x's side.** Take the emails x sent in `(t0 - 30, t0]`, group them by the
receiver's department, and keep the top five departments by count (ties go to the lower
department ID). Build the other person y's set: y's own department plus every department y
wrote to at or before `t0`. `S_xy = 100 * |overlap| / 5`. If x or y has no resolved
department, `S_xy = 0`.

## `data/processed/excluded_main.csv`

Same columns as the pair table up to `t0`, for pairs whose last email is fewer than 90 days
before `t0`. They are excluded by the dormancy gate and counted in `reports/dataset_summary.json`.

## `data/processed/predictions_main.csv` and `predictions_oot.csv`

| Column | Meaning |
|---|---|
| `u`, `v`, `fold`, `label` | As above. |
| `p_logistic` | Probability from the L2 logistic regression. |
| `p_network` | Probability from ReconnectScoreNet. |

## Reports (`reports/`)

| File | Content |
|---|---|
| `data_audit.json` | Raw counts, naive ID overlap, network-wide gap, emails per 30 days. |
| `crosswalk_audit.json` | Evidence for the crosswalk and the list of structural twins. |
| `dataset_summary.json` | Pair counts, rows dropped, class balance, folds, frozen bins. |
| `metrics_test.csv`, `metrics_oot.csv` | Metric tables. |
| `network_seeds.csv` | ReconnectScoreNet over five seeds. |
| `ablation.csv` | Drop-one-signal results. |
| `error_analysis_fp.csv` | High-overdue, low-similarity false positives. |
| `model_details.json` | Thresholds, logistic coefficients, training summary. |
| `case_study_sender.csv`, `case_study_check.json` | One-sender table, hand recomputation, ranking stability. |
| `figures/pr_curve_test.png` | Precision–recall curves on the test pairs. |
