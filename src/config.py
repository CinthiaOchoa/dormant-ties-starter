"""Shared constants for the dormant-ties pipeline.

Project: Ranking Dormant Ties in a Timestamped Email Network
Course:  CSC 4740/6740 Data Mining, Team 3
Author:  Cinthia Ochoa Torre
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
PROCESSED = ROOT / 'data' / 'processed'
REPORTS = ROOT / 'reports'
FIGURES = REPORTS / 'figures'

TEMPORAL_FILE = RAW / 'email-Eu-core-temporal.txt.gz'
STATIC_FILE = RAW / 'email-Eu-core.txt.gz'
DEPARTMENT_FILE = RAW / 'email-Eu-core-department-labels.txt.gz'
CROSSWALK_FILE = PROCESSED / 'node_crosswalk.csv'

# Time design (days since the first observed email).
# Phase 1 proposed t0 = 600, but the trace has no email between day 525.5 and
# day 797.3, so a label window after day 600 is empty. The decision times are
# fixed by a rule that does not look at labels: the latest multiple of 10 such
# that both label windows (main and out-of-time) end inside the contiguous
# record, i.e. T0 + 2 * HORIZON <= 525.5.
T0 = 400            # main decision time
T0_OOT = 460        # out-of-time robustness check; never used for selection
MAX_GAP_DAYS = 30   # a silence longer than this in the whole network ends the contiguous record
GATE = 90           # a pair is dormant when days since last email >= GATE
HORIZON = 60        # label window is (t0, t0 + HORIZON]
PROFILE_WINDOW = 30 # recent-activity window ending at t0
TOP_K = 5           # departments kept in a sender's recent profile

# Reconnect Score weights: s = 0.35 S + 0.25 H + 0.20 C + 0.20 T
WEIGHTS = {'S': 0.35, 'H': 0.25, 'C': 0.20, 'T': 0.20}
SIGNALS = ['S', 'H', 'C', 'T']
FIRST_PASS_CUTOFF = 48.0

# Closeness bin values; edges between the last three are fit on training pairs.
H_VALUES = (20, 35, 60, 90)
H_LOW_MAX = 2  # one or two emails always map to 20

# Signal values for History.
C_MUTUAL, C_ONE_WAY = 80, 40

SPLIT = (0.70, 0.15, 0.15)
SEED = 4740
