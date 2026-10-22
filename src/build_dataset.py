"""Build the modeling tables at the main and out-of-time decision times.

Author: Cinthia Ochoa Torre

Outputs
  data/processed/pairs_main.csv     dormant pairs at T0 with signals, score, label, fold
  data/processed/pairs_oot.csv      dormant pairs at the out-of-time decision time T0_OOT
  data/processed/excluded_main.csv  pairs removed by the dormancy gate (still active)
  reports/dataset_summary.json      pair counts, class balance, rows dropped, frozen bins
"""
import json

import numpy as np
from sklearn.model_selection import train_test_split

from config import (GATE, HORIZON, MAX_GAP_DAYS, PROCESSED, PROFILE_WINDOW, REPORTS, SEED,
                    SIGNALS, SPLIT, T0, T0_OOT, TOP_K)
from features import (apply_closeness, build_pairs, fit_closeness_edges,
                      load_departments, load_events, reconnect_score)


def assign_folds(dormant):
    """Random 70/15/15 split of dormant pairs, stratified on the label."""
    idx = np.arange(len(dormant))
    train, rest = train_test_split(idx, train_size=SPLIT[0], random_state=SEED,
                                   stratify=dormant.label)
    val, test = train_test_split(rest, test_size=SPLIT[2] / (SPLIT[1] + SPLIT[2]),
                                 random_state=SEED, stratify=dormant.label.iloc[rest])
    fold = np.empty(len(dormant), dtype=object)
    fold[train], fold[val], fold[test] = 'train', 'val', 'test'
    return fold


def contiguous_record(edges):
    """End of the contiguous record: the last email before the first network-wide gap > MAX_GAP_DAYS."""
    days = np.sort(edges.day.unique())
    gaps = np.diff(days)
    big = np.flatnonzero(gaps > MAX_GAP_DAYS)
    if len(big) == 0:
        return float(days[-1]), None
    i = big[0]
    return float(days[i]), {'from_day': round(float(days[i]), 2), 'to_day': round(float(days[i + 1]), 2),
                            'emails_after_gap': int((edges.day > days[i]).sum())}


def finish(frame, h_edges):
    frame['H'] = apply_closeness(frame.emails_before_t0.to_numpy(), h_edges)
    frame['score'] = reconnect_score(frame)
    for k in SIGNALS:
        lo, hi = frame[k].min(), frame[k].max()
        assert 0 <= lo and hi <= 100, f'{k} outside 0-100: [{lo}, {hi}]'
    return frame


def describe(pairs, dormant):
    return {
        'pairs_with_history': len(pairs),
        'active_pairs_excluded_by_gate': int((~pairs.dormant).sum()),
        'dormant_pairs': len(dormant),
        'dormant_positives': int(dormant.label.sum()),
        'dormant_positive_rate': round(float(dormant.label.mean()), 4),
        'dormant_pairs_with_unknown_department': int((~dormant.dept_known).sum()),
        'dormant_pairs_mutual': int((dormant.C == 80).sum()),
        'dormant_pairs_one_way': int((dormant.C == 40).sum()),
    }


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    edges = load_events()
    dept = load_departments()
    record_end, gap = contiguous_record(edges)
    if T0_OOT + HORIZON > record_end:
        raise SystemExit(f'Label window ends at day {T0_OOT + HORIZON}, after the record ends at {record_end:.1f}.')

    pairs, _ = build_pairs(edges, dept, T0)
    dormant = pairs.loc[pairs.dormant].reset_index(drop=True)
    pairs.loc[~pairs.dormant].to_csv(PROCESSED / 'excluded_main.csv', index=False)
    dormant['fold'] = assign_folds(dormant)
    h_edges = fit_closeness_edges(dormant.loc[dormant.fold == 'train', 'emails_before_t0'])
    dormant = finish(dormant, h_edges)
    dormant.to_csv(PROCESSED / 'pairs_main.csv', index=False)

    pairs_oot, _ = build_pairs(edges, dept, T0_OOT)
    dormant_oot = finish(pairs_oot.loc[pairs_oot.dormant].reset_index(drop=True), h_edges)
    dormant_oot['fold'] = 'oot'
    dormant_oot.to_csv(PROCESSED / 'pairs_oot.csv', index=False)

    summary = {
        'raw_temporal_rows': int(edges.attrs['raw_rows']),
        'self_mail_dropped': int(edges.attrs['self_mail_dropped']),
        'people_in_temporal_file': int(len(set(edges.src) | set(edges.dst))),
        'people_with_resolved_department': len(dept),
        'trace_days': round(float(edges.day.max()), 2),
        'contiguous_record_end_day': round(record_end, 2),
        'network_wide_gap': gap,
        'design': {'t0': T0, 't0_out_of_time': T0_OOT, 'gate_days': GATE,
                   'label_horizon_days': HORIZON, 'profile_window_days': PROFILE_WINDOW,
                   'profile_top_k': TOP_K, 'split': SPLIT, 'seed': SEED},
        'closeness_bins': {'20': '1-2 emails', '35': f'3-{h_edges[0]} emails',
                           '60': f'{h_edges[0] + 1}-{h_edges[1]} emails',
                           '90': f'>{h_edges[1]} emails', 'fit_on': 'training pairs at t0'},
        'closeness_edges': list(h_edges),
        f't{T0}': describe(pairs, dormant),
        f't{T0}_folds': {f: {'pairs': int((dormant.fold == f).sum()),
                             'positives': int(dormant.loc[dormant.fold == f, 'label'].sum())}
                         for f in ('train', 'val', 'test')},
        f't{T0_OOT}': describe(pairs_oot, dormant_oot),
        f'signal_means_t{T0}': {k: round(float(dormant[k].mean()), 2) for k in SIGNALS + ['score']},
    }
    (REPORTS / 'dataset_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
