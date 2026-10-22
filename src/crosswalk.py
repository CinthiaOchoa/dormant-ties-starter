"""Recover the node-ID crosswalk between the temporal and static SNAP releases.

Author: Cinthia Ochoa Torre

SNAP numbers people differently in email-Eu-core (static, which carries the
department labels) and email-Eu-core-temporal. A direct join on the integer ID
is invalid: its edge overlap is no better than a random relabeling.

The static edge list equals the collapsed temporal edge list plus self-loops,
so the two graphs (self-loops removed) should be isomorphic. This script finds
the isomorphism with joint color refinement (1-dimensional Weisfeiler-Leman) on
the disjoint union of both directed graphs, then individualizes and refines
until every class is a single node per side. The mapping is accepted only if
every directed temporal edge maps onto a static edge.

Nodes that color refinement cannot tell apart (structurally equivalent twins)
are mapped arbitrarily inside their class. Their department is trusted only if
every static node in the class carries the same department; otherwise it is
marked unresolved and the Similarity signal treats it as missing.
"""
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from config import (CROSSWALK_FILE, DEPARTMENT_FILE, PROCESSED, REPORTS,
                    STATIC_FILE, TEMPORAL_FILE)


def read_edges():
    temporal = pd.read_csv(TEMPORAL_FILE, sep=r'\s+', names=['src', 'dst', 'ts'], comment='#')
    static = pd.read_csv(STATIC_FILE, sep=r'\s+', names=['src', 'dst'], comment='#')
    depts = pd.read_csv(DEPARTMENT_FILE, sep=r'\s+', names=['node', 'department'], comment='#')
    return temporal, static, depts


def refine(nodes, out_nb, in_nb, colors):
    """Color refinement on a directed graph until the partition is stable."""
    while True:
        signature = {n: (colors[n],
                         tuple(sorted(colors[x] for x in out_nb[n])),
                         tuple(sorted(colors[x] for x in in_nb[n]))) for n in nodes}
        index = {key: i for i, key in enumerate(sorted(set(signature.values()), key=repr))}
        new = {n: index[signature[n]] for n in nodes}
        if len(set(new.values())) == len(set(colors.values())):
            return new
        colors = new


def classes_of(colors):
    groups = defaultdict(lambda: {'t': [], 's': []})
    for (side, node), color in colors.items():
        groups[color][side].append(node)
    return groups


def build_crosswalk():
    temporal, static, depts = read_edges()
    t_edges = set(zip(temporal.src, temporal.dst)) - {(a, a) for a in temporal.src}
    s_edges = {(a, b) for a, b in zip(static.src, static.dst) if a != b}

    union_edges = [(('t', a), ('t', b)) for a, b in t_edges] + \
                  [(('s', a), ('s', b)) for a, b in s_edges]
    nodes = sorted({x for e in union_edges for x in e})
    out_nb, in_nb = defaultdict(list), defaultdict(list)
    for a, b in union_edges:
        out_nb[a].append(b)
        in_nb[b].append(a)

    colors = refine(nodes, out_nb, in_nb, {n: 0 for n in nodes})
    groups = classes_of(colors)
    if any(len(g['t']) != len(g['s']) for g in groups.values()):
        raise RuntimeError('Graphs are not isomorphic under color refinement.')
    twin_classes = [g for g in groups.values() if len(g['t']) > 1]

    # Individualize one pair at a time until the partition is discrete.
    while True:
        open_classes = [g for g in classes_of(colors).values() if len(g['t']) > 1]
        if not open_classes:
            break
        g = min(open_classes, key=lambda c: (len(c['t']), min(c['t'])))
        fresh = max(colors.values()) + 1
        colors = dict(colors)
        colors[('t', min(g['t']))] = fresh
        colors[('s', min(g['s']))] = fresh
        colors = refine(nodes, out_nb, in_nb, colors)

    mapping = {g['t'][0]: g['s'][0] for g in classes_of(colors).values()}
    preserved = sum((mapping[a], mapping[b]) in s_edges for a, b in t_edges)
    if preserved != len(t_edges) or len(t_edges) != len(s_edges):
        raise RuntimeError(f'Crosswalk preserves only {preserved}/{len(t_edges)} edges.')

    dept_of_static = dict(zip(depts.node, depts.department))
    unresolved = set()
    for g in twin_classes:
        if len({dept_of_static[s] for s in g['s']}) > 1:
            unresolved.update(g['t'])
    twin_nodes = {t for g in twin_classes for t in g['t']}

    rows = []
    for t_node in sorted(mapping):
        s_node = mapping[t_node]
        rows.append({'temporal_id': t_node, 'static_id': s_node,
                     'structural_twin': t_node in twin_nodes,
                     'department_resolved': t_node not in unresolved,
                     'department': dept_of_static[s_node] if t_node not in unresolved else np.nan})
    table = pd.DataFrame(rows)
    table['department'] = table['department'].astype('Int64')

    # Evidence that the naive join is wrong: identity edge overlap vs a random relabeling.
    rng = np.random.default_rng(0)
    perm = rng.permutation(max(static.src.max(), static.dst.max()) + 1)
    audit = {
        'temporal_directed_edges': len(t_edges),
        'static_directed_edges_without_self_loops': len(s_edges),
        'static_self_loops': int((static.src == static.dst).sum()),
        'temporal_nodes': len(mapping),
        'static_nodes_with_labels': int(len(depts)),
        'naive_identity_join_edge_overlap': len(t_edges & s_edges),
        'random_relabeling_edge_overlap': len({(perm[a], perm[b]) for a, b in t_edges} & s_edges),
        'crosswalk_edges_preserved': preserved,
        'uniquely_identified_nodes': len(mapping) - len(twin_nodes),
        'structural_twin_nodes': len(twin_nodes),
        'twin_nodes_with_consistent_department': len(twin_nodes - unresolved),
        'nodes_with_unresolved_department': len(unresolved),
        'nodes_with_department': int(table.department_resolved.sum()),
        'twin_classes': [{'temporal_ids': sorted(int(x) for x in g['t']),
                          'static_ids': sorted(int(x) for x in g['s']),
                          'static_departments': sorted(int(dept_of_static[x]) for x in g['s'])}
                         for g in sorted(twin_classes, key=lambda c: min(c['t']))],
    }
    return table, audit


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    table, audit = build_crosswalk()
    table.to_csv(CROSSWALK_FILE, index=False)
    (REPORTS / 'crosswalk_audit.json').write_text(json.dumps(audit, indent=2))
    print(f"Crosswalk: {audit['crosswalk_edges_preserved']}/{audit['temporal_directed_edges']} edges preserved; "
          f"{audit['nodes_with_department']} of {audit['temporal_nodes']} nodes have a department "
          f"({audit['nodes_with_unresolved_department']} unresolved twins).")
    print(f"Naive ID join overlap: {audit['naive_identity_join_edge_overlap']} "
          f"(random relabeling: {audit['random_relabeling_edge_overlap']}).")


if __name__ == '__main__':
    main()
