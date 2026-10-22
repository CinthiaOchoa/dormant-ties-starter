"""Audit the raw SNAP files before any feature is built.

Author: Cinthia Ochoa Torre

Reports row counts, self-mail, node counts, the naive ID overlap between the
temporal and static releases (which is NOT evidence of alignment; see
crosswalk.py), and the network-wide silence that ends the contiguous record.
"""
import json

import numpy as np
import pandas as pd

from config import DEPARTMENT_FILE, MAX_GAP_DAYS, REPORTS, STATIC_FILE, TEMPORAL_FILE

if __name__ == '__main__':
    REPORTS.mkdir(exist_ok=True)
    edges = pd.read_csv(TEMPORAL_FILE, sep=r'\s+', names=['src', 'dst', 'ts'], comment='#')
    static = pd.read_csv(STATIC_FILE, sep=r'\s+', names=['src', 'dst'], comment='#')
    depts = pd.read_csv(DEPARTMENT_FILE, sep=r'\s+', names=['node', 'department'], comment='#')

    edges['day'] = (edges.ts - edges.ts.min()) / 86400.0
    days = np.sort(edges.day.unique())
    gaps = np.diff(days)
    longest = int(np.argmax(gaps))
    monthly = np.histogram(edges.day, bins=np.arange(0, edges.day.max() + 30, 30))[0]

    report = {
        'temporal_rows': len(edges),
        'temporal_self_mail': int((edges.src == edges.dst).sum()),
        'temporal_nodes': len(set(edges.src) | set(edges.dst)),
        'static_rows': len(static),
        'static_self_loops': int((static.src == static.dst).sum()),
        'department_rows': len(depts),
        'department_count': int(depts.department.nunique()),
        'naive_numeric_id_intersection': len((set(edges.src) | set(edges.dst)) & set(depts.node)),
        'warning': 'Temporal and static node IDs differ; use the verified crosswalk from crosswalk.py.',
        'trace_days': round(float(edges.day.max()), 2),
        'longest_network_wide_gap': {'from_day': round(float(days[longest]), 2),
                                     'to_day': round(float(days[longest + 1]), 2),
                                     'days': round(float(gaps[longest]), 2)},
        'gaps_longer_than_max_gap_days': int((gaps > MAX_GAP_DAYS).sum()),
        'emails_per_30_days': monthly.tolist(),
    }
    (REPORTS / 'data_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
