"""Pair table and the four Reconnect signals (S, H, C, T) at a decision time.

Author: Cinthia Ochoa Torre

Every signal uses only emails with day <= t0. The label uses only emails in
(t0, t0 + HORIZON]. Closeness (H) needs bin edges fit on training pairs, so it
is applied separately with apply_closeness once those edges are frozen.
"""
import numpy as np
import pandas as pd

from config import (C_MUTUAL, C_ONE_WAY, CROSSWALK_FILE, GATE, H_LOW_MAX,
                    H_VALUES, HORIZON, PROFILE_WINDOW, TEMPORAL_FILE, TOP_K,
                    WEIGHTS)


def load_events():
    """Temporal edge list with self-mail dropped and time in days from the first email."""
    edges = pd.read_csv(TEMPORAL_FILE, sep=r'\s+', names=['src', 'dst', 'ts'], comment='#')
    n_raw = len(edges)
    edges = edges.loc[edges.src != edges.dst].copy()
    edges['day'] = (edges.ts - edges.ts.min()) / 86400.0
    edges.attrs['raw_rows'] = n_raw
    edges.attrs['self_mail_dropped'] = n_raw - len(edges)
    return edges


def load_departments():
    """Temporal node ID -> department, only for nodes the crosswalk resolved."""
    cw = pd.read_csv(CROSSWALK_FILE)
    cw = cw.loc[cw.department_resolved]
    return dict(zip(cw.temporal_id.astype(int), cw.department.astype(int)))


def recent_profiles(observed, dept, t0):
    """Each sender's top-K receiver departments over the PROFILE_WINDOW days ending at t0.

    Ties in email count are broken by the lower department ID.
    """
    recent = observed.loc[observed.day > t0 - PROFILE_WINDOW, ['src', 'dst']].copy()
    recent['dept'] = recent.dst.map(dept)
    recent = recent.dropna(subset=['dept'])
    counts = recent.groupby(['src', 'dept']).size().rename('n').reset_index()
    counts = counts.sort_values(['src', 'n', 'dept'], ascending=[True, False, True])
    top = counts.groupby('src').head(TOP_K)
    return top.groupby('src').dept.apply(lambda d: set(d.astype(int))).to_dict()


def contact_departments(observed, dept):
    """Departments each person wrote to at or before t0."""
    sent = observed[['src', 'dst']].drop_duplicates().copy()
    sent['dept'] = sent.dst.map(dept)
    sent = sent.dropna(subset=['dept'])
    return sent.groupby('src').dept.apply(lambda d: set(d.astype(int))).to_dict()


def directed_similarity(sender, alter, dept, profiles, contacts):
    """S from the sender's side: overlap of the alter's department set with the sender's top-K."""
    if sender not in dept or alter not in dept:
        return 0.0
    alter_set = {dept[alter]} | contacts.get(alter, set())
    return 100.0 * len(alter_set & profiles.get(sender, set())) / TOP_K


def overdue(days_silent, gate=GATE):
    return np.clip(100.0 * (days_silent - gate) / gate, 0.0, 100.0)


def build_pairs(edges, dept, t0):
    """One row per unordered pair with at least one email at or before t0.

    Returns every pair; the `dormant` column marks those that pass the gate.
    """
    observed = edges.loc[edges.day <= t0].copy()
    future = edges.loc[(edges.day > t0) & (edges.day <= t0 + HORIZON)].copy()
    for table in (observed, future):
        table['u'] = np.minimum(table.src, table.dst)
        table['v'] = np.maximum(table.src, table.dst)

    pairs = observed.groupby(['u', 'v'], as_index=False).agg(
        first_email_day=('day', 'min'), last_email_day=('day', 'max'),
        emails_before_t0=('day', 'size'))
    directional = observed.groupby(['src', 'dst']).size()
    pairs['u_to_v'] = directional.reindex(list(zip(pairs.u, pairs.v)), fill_value=0).to_numpy()
    pairs['v_to_u'] = directional.reindex(list(zip(pairs.v, pairs.u)), fill_value=0).to_numpy()

    future_counts = future.groupby(['u', 'v']).size().rename('future_emails').reset_index()
    pairs = pairs.merge(future_counts, on=['u', 'v'], how='left')
    pairs['future_emails'] = pairs.future_emails.fillna(0).astype(int)
    pairs['label'] = (pairs.future_emails > 0).astype(int)

    pairs['days_silent'] = t0 - pairs.last_email_day
    pairs['dormant'] = pairs.days_silent >= GATE
    pairs['t0'] = t0

    profiles = recent_profiles(observed, dept, t0)
    contacts = contact_departments(observed, dept)
    pairs['dept_u'] = pairs.u.map(dept).astype('Int64')
    pairs['dept_v'] = pairs.v.map(dept).astype('Int64')
    pairs['dept_known'] = pairs.dept_u.notna() & pairs.dept_v.notna()
    pairs['same_department'] = (pairs.dept_u == pairs.dept_v).fillna(False).astype(bool)
    pairs['S_uv'] = [directed_similarity(u, v, dept, profiles, contacts) for u, v in zip(pairs.u, pairs.v)]
    pairs['S_vu'] = [directed_similarity(v, u, dept, profiles, contacts) for u, v in zip(pairs.u, pairs.v)]
    # The pair is symmetric, so S takes the better-matching direction.
    pairs['S'] = pairs[['S_uv', 'S_vu']].max(axis=1)
    pairs['C'] = np.where((pairs.u_to_v > 0) & (pairs.v_to_u > 0), C_MUTUAL, C_ONE_WAY)
    pairs['T'] = overdue(pairs.days_silent)
    return pairs, profiles


def fit_closeness_edges(train_counts):
    """Tertile edges of email counts above H_LOW_MAX among training pairs."""
    above = np.asarray(train_counts)[np.asarray(train_counts) > H_LOW_MAX]
    lo = max(H_LOW_MAX + 1, int(np.floor(np.quantile(above, 1 / 3))))
    hi = max(lo + 1, int(np.floor(np.quantile(above, 2 / 3))))
    return lo, hi


def apply_closeness(counts, edges):
    """Map email counts to H: <=2 -> 20, <=lo -> 35, <=hi -> 60, else 90."""
    lo, hi = edges
    return np.select([counts <= H_LOW_MAX, counts <= lo, counts <= hi],
                     list(H_VALUES[:3]), default=H_VALUES[3]).astype(float)


def reconnect_score(frame, drop=None):
    """Weighted sum of the four signals; `drop` removes one term for the ablation."""
    return sum(w * frame[k] for k, w in WEIGHTS.items() if k != drop)
