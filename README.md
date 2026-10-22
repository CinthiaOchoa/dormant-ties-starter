# Ranking Dormant Ties in a Timestamped Email Network

CSC 4740/6740 Data Mining — Team 3 — Phase 2 (October 27, 2026)
Author: Cinthia Ochoa Torre

For pairs of people who emailed each other and then went silent for at least 90 days,
the project scores how likely the pair is to write again within 60 days. An
interpretable weighted score (the Reconnect Score) is compared with a majority baseline,
logistic regression, and a small neural network (ReconnectScoreNet) on the SNAP
`email-Eu-core` data.

The Phase 2 mid-project report is [`reports/phase2_report.pdf`](reports/phase2_report.pdf) (also `.docx`; source in `reports/phase2_report.md`).

## Two data issues to know before reading the code

1. **Node IDs differ between releases.** The department labels belong to the static
   release, which numbers people differently from the temporal edge list. A direct join
   is no better than a random relabeling. `src/crosswalk.py` recovers the mapping by
   graph isomorphism and verifies it on all 24,929 directed edges. 21 people whose
   structural twins have different departments keep an unresolved department.
2. **The trace stops at day 525.5** and resumes only for a 7-day burst at day 797. The
   decision times are therefore day 400 (main) and day 460 (out-of-time), not the 600
   and 660 of the Phase 1 proposal. `src/config.py` documents the rule.

## Setup (macOS / Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py   # skips files already in data/raw/
python src/run_all.py
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1`.
The full run takes under a minute on a laptop CPU.

## Pipeline

| Step | Script | Main outputs |
|---|---|---|
| Download | `src/download_data.py` | `data/raw/*.gz` |
| Audit | `src/audit_data.py` | `reports/data_audit.json` |
| ID crosswalk | `src/crosswalk.py` | `data/processed/node_crosswalk.csv`, `reports/crosswalk_audit.json` |
| Pairs and signals | `src/build_dataset.py` (uses `src/features.py`) | `data/processed/pairs_main.csv`, `pairs_oot.csv`, `reports/dataset_summary.json` |
| Models and evaluation | `src/train_evaluate.py` (uses `src/models.py`) | `reports/metrics_test.csv`, `metrics_oot.csv`, `ablation.csv`, `error_analysis_fp.csv`, `figures/pr_curve_test.png` |
| Case study | `src/case_study.py` | `reports/case_study_sender.csv`, `case_study_check.json` |

`src/run_all.py` runs the five steps after the download in order. All constants (decision
times, gate, horizon, weights, split, seed) live in `src/config.py`. Column definitions are
in [`docs/data_dictionary.md`](docs/data_dictionary.md).

## Sources

- https://snap.stanford.edu/data/email-Eu-core-temporal.html
- https://snap.stanford.edu/data/email-Eu-core.html

The raw archives and `data/processed/` are not tracked; both are rebuilt by the scripts.

## Phase 2 checklist

- [x] Download original archives and record counts
- [x] Inspect node identity mismatch and resolve the attribute mapping
- [x] Freeze decision time, 90-day dormancy gate, 60-day label horizon (t0 moved to 400; see report)
- [x] Build the four documented features
- [x] Fit baselines and the neural network
- [x] Hold out test pairs and report actual metrics
- [x] Record four ablations; document departures from the proposal
- [x] Case study with hand recomputation; out-of-time check
