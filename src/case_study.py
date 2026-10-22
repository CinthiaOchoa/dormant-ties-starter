"""One-sender case study, hand recomputation, and ranking stability at the second decision time.

Author: Cinthia Ochoa Torre

The case-study sender is the person whose emails sent before T0 are closest to
the median among people with at least MIN_PARTNERS dormant partners. Three rows
of that sender's table are then recomputed from the raw gzip file with plain
Python (no pandas, no code shared with features.py) and must match exactly.

Outputs
  reports/case_study_sender.csv   every dormant partner of the sender with signals, score, probability, label
  reports/case_study_check.json   hand recomputation of three rows and ranking stability at T0_OOT
"""
import csv
import gzip
import json

import numpy as np
import pandas as pd

from config import (CROSSWALK_FILE, GATE, HORIZON, PROCESSED, PROFILE_WINDOW,
                    REPORTS, SEED, T0, T0_OOT, TEMPORAL_FILE, TOP_K, WEIGHTS)

MIN_PARTNERS = 8
TOP_N = 5


def partner_view(pairs, sender):
    """Re-express each unordered pair from the sender's side."""
    rows = pairs.loc[(pairs.u == sender) | (pairs.v == sender)].copy()
    is_u = rows.u == sender
    rows['partner'] = np.where(is_u, rows.v, rows.u)
    rows['sent_by_sender'] = np.where(is_u, rows.u_to_v, rows.v_to_u)
    rows['sent_by_partner'] = np.where(is_u, rows.v_to_u, rows.u_to_v)
    rows['partner_department'] = np.where(is_u, rows.dept_v, rows.dept_u)
    rows['S_from_sender'] = np.where(is_u, rows.S_uv, rows.S_vu)
    return rows


def choose_sender(pairs, edges_sent):
    counts = pd.concat([pairs.u, pairs.v]).value_counts()
    eligible = counts[counts >= MIN_PARTNERS].index
    sent = edges_sent.reindex(eligible, fill_value=0)
    median = sent.median()
    return int((sent - median).abs().sort_values(kind='stable').index[0]), int(sent.median())


def hand_recompute(sender, partner, h_edges):
    """Recompute S, H, C, T, and s for one pair straight from the raw file."""
    dept = {}
    with open(CROSSWALK_FILE) as f:
        for row in csv.DictReader(f):
            if row['department_resolved'] == 'True':
                dept[int(row['temporal_id'])] = int(row['department'])

    events = []
    with gzip.open(TEMPORAL_FILE, 'rt') as f:
        for line in f:
            a, b, ts = (int(x) for x in line.split())
            if a != b:
                events.append((a, b, ts))
    start = min(ts for _, _, ts in events)

    sent = {sender: 0, partner: 0}
    last_day = None
    recent = {}
    contacts = {sender: set(), partner: set()}
    for a, b, ts in events:
        day = (ts - start) / 86400.0
        if day > T0:
            continue
        if {a, b} == {sender, partner}:
            sent[a] += 1
            last_day = day if last_day is None else max(last_day, day)
        if a in contacts and b in dept:
            contacts[a].add(dept[b])
        if day > T0 - PROFILE_WINDOW and b in dept and a in (sender, partner):
            recent.setdefault(a, {}).setdefault(dept[b], 0)
            recent[a][dept[b]] += 1

    def top(person):
        ranked = sorted(recent.get(person, {}).items(), key=lambda kv: (-kv[1], kv[0]))
        return {d for d, _ in ranked[:TOP_K]}

    def similarity(x, y):
        if x not in dept or y not in dept:
            return 0.0
        return 100.0 * len(({dept[y]} | contacts[y]) & top(x)) / TOP_K

    total = sent[sender] + sent[partner]
    days_silent = T0 - last_day
    s_value = max(similarity(sender, partner), similarity(partner, sender))
    h_value = 20 if total <= 2 else 35 if total <= h_edges[0] else 60 if total <= h_edges[1] else 90
    c_value = 80 if sent[sender] > 0 and sent[partner] > 0 else 40
    t_value = min(100.0, max(0.0, 100.0 * (days_silent - GATE) / GATE))
    score = WEIGHTS['S'] * s_value + WEIGHTS['H'] * h_value + WEIGHTS['C'] * c_value + WEIGHTS['T'] * t_value
    return {'partner': partner, 'emails_sender_to_partner': sent[sender],
            'emails_partner_to_sender': sent[partner], 'days_silent': round(days_silent, 4),
            'S': s_value, 'H': h_value, 'C': c_value, 'T': round(t_value, 4), 'score': round(score, 4)}


def main():
    pairs = pd.read_csv(PROCESSED / 'pairs_main.csv')
    pairs_oot = pd.read_csv(PROCESSED / 'pairs_oot.csv')
    preds = pd.read_csv(PROCESSED / 'predictions_main.csv')
    preds_oot = pd.read_csv(PROCESSED / 'predictions_oot.csv')
    pairs = pairs.merge(preds[['u', 'v', 'p_network']], on=['u', 'v'])
    pairs_oot = pairs_oot.merge(preds_oot[['u', 'v', 'p_network']], on=['u', 'v'])
    summary = json.loads((REPORTS / 'dataset_summary.json').read_text())
    h_edges = summary['closeness_edges']

    edges = pd.read_csv(TEMPORAL_FILE, sep=r'\s+', names=['src', 'dst', 'ts'])
    edges['day'] = (edges.ts - edges.ts.min()) / 86400.0
    sent_before = edges.loc[(edges.day <= T0) & (edges.src != edges.dst)].src.value_counts()
    sender, median_sent = choose_sender(pairs, sent_before)

    view = partner_view(pairs, sender).sort_values(['score', 'partner'], ascending=[False, True])
    view['rank'] = np.arange(1, len(view) + 1)
    table = view[['rank', 'partner', 'fold', 'dept_u', 'dept_v', 'partner_department', 'same_department',
                  'sent_by_sender', 'sent_by_partner', 'emails_before_t0', 'days_silent',
                  'S_from_sender', 'S', 'H', 'C', 'T', 'score', 'p_network', 'label']].copy()
    table = table.drop(columns=['dept_u', 'dept_v'])
    table.insert(1, 'sender', sender)
    table.to_csv(REPORTS / 'case_study_sender.csv', index=False, float_format='%.3f')

    # Hand-check the top, middle, and bottom rows.
    picks = sorted({0, len(view) // 2, len(view) - 1})
    checks = []
    for i in picks:
        row = view.iloc[i]
        hand = hand_recompute(sender, int(row.partner), h_edges)
        pipeline = {'S': float(row.S), 'H': float(row.H), 'C': float(row.C),
                    'T': round(float(row['T']), 4), 'score': round(float(row.score), 4)}
        match = all(abs(hand[k] - pipeline[k]) < 1e-3 for k in pipeline)
        checks.append({'rank': int(row['rank']), 'hand': hand, 'pipeline': pipeline, 'match': match})
    if not all(c['match'] for c in checks):
        raise SystemExit(f'Hand recomputation does not match the pipeline: {checks}')

    # Ranking stability: does the sender's top keep its place at the second decision time?
    view_oot = partner_view(pairs_oot, sender).sort_values(['score', 'partner'], ascending=[False, True])
    top_main = view.partner.head(TOP_N).tolist()
    top_oot = view_oot.partner.head(TOP_N).tolist()
    still_dormant = [p for p in top_main if p in set(view_oot.partner)]

    # Same question across all senders: is the top partner at T0 still the top partner at T0_OOT?
    def tops(frame):
        both = pd.concat([pd.DataFrame({'sender': frame.u, 'partner': frame.v, 'score': frame.score}),
                          pd.DataFrame({'sender': frame.v, 'partner': frame.u, 'score': frame.score})])
        return both.sort_values(['sender', 'score', 'partner'], ascending=[True, False, True])
    ranked_main, ranked_oot = tops(pairs), tops(pairs_oot)
    first_main = ranked_main.groupby('sender').partner.first()
    first_oot = ranked_oot.groupby('sender').partner.first()
    common = first_main.index.intersection(first_oot.index)
    oot_partners = ranked_oot.groupby('sender').partner.apply(set)
    alive = [s for s in common if first_main[s] in oot_partners[s]]
    same_top = float(np.mean([first_main[s] == first_oot[s] for s in alive])) if alive else None

    report = {
        'sender': sender, 'selection_rule': f'emails sent before t0 closest to the median ({median_sent}) '
                                            f'among people with >= {MIN_PARTNERS} dormant partners',
        'sender_emails_sent_before_t0': int(sent_before.get(sender, 0)),
        'dormant_partners': len(view), 'resumed_within_horizon': int(view.label.sum()),
        'hand_checks': checks,
        'stability': {
            f'top_{TOP_N}_at_t{T0}': top_main, f'top_{TOP_N}_at_t{T0_OOT}': top_oot,
            f'top_{TOP_N}_still_dormant_at_t{T0_OOT}': still_dormant,
            'overlap_of_top_lists': len(set(top_main) & set(top_oot)),
            'all_senders': {'senders_with_dormant_partners_at_both_times': len(common),
                            'senders_whose_t0_top_partner_is_still_dormant': len(alive),
                            'share_keeping_same_top_partner': same_top},
        },
        'note': f'Labels are emails in (t0, t0 + {HORIZON}]; seed {SEED}.',
    }
    (REPORTS / 'case_study_check.json').write_text(json.dumps(report, indent=2, default=int))
    print(table.head(15).to_string(index=False))
    print(json.dumps({k: v for k, v in report.items() if k != 'hand_checks'}, indent=2, default=int))
    print('Hand checks match:', [c['match'] for c in checks])


if __name__ == '__main__':
    main()
