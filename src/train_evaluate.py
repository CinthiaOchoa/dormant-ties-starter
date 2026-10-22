"""Fit the four procedures, evaluate on the test pairs, and run the ablation and error analysis.

Author: Cinthia Ochoa Torre

Model choices (logistic penalty, decision thresholds, early stopping) use only
the validation pairs. The test pairs and the out-of-time pairs are scored once.

Outputs
  reports/metrics_test.csv           the four procedures on the test pairs at T0
  reports/metrics_oot.csv            the same fitted procedures on the out-of-time pairs
  reports/network_seeds.csv          ReconnectScoreNet test metrics over five seeds
  reports/ablation.csv               drop-one-signal change in average precision and top partner
  reports/error_analysis_fp.csv      high-overdue, low-similarity false positives on the test pairs
  reports/model_details.json         thresholds, penalty, coefficients, training curve summary
  reports/figures/pr_curve_test.png  precision-recall curves on the test pairs
  data/processed/predictions_*.csv   per-pair probabilities for the case study
"""
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from config import (FIGURES, FIRST_PASS_CUTOFF, PROCESSED, REPORTS, SEED,
                    SIGNALS, T0, T0_OOT, WEIGHTS)
from features import reconnect_score
from models import (fit_logistic, predict_network, synthetic_debug_matrix,
                    train_network)

ERROR_T_MIN, ERROR_S_MAX = 75.0, 20.0
N_SEEDS = 5
N_RANKING_SENDERS = 100


def best_f1_threshold(y, score):
    precision, recall, thresholds = precision_recall_curve(y, score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[np.argmax(f1[:-1])])


def metrics(y, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        'accuracy': accuracy_score(y, pred),
        'precision': precision_score(y, pred, zero_division=0),
        'recall': recall_score(y, pred, zero_division=0),
        'f1': f1_score(y, pred, zero_division=0),
        'roc_auc': roc_auc_score(y, prob) if len(np.unique(prob)) > 1 else 0.5,
        'average_precision': average_precision_score(y, prob),
        'brier': brier_score_loss(y, np.clip(prob, 0, 1)),
        'threshold': threshold,
        'predicted_positive': int(pred.sum()),
    }


def inputs(frame, signals=SIGNALS):
    return frame[signals].to_numpy(dtype=float) / 100.0


def fit_procedures(train, val, signals=SIGNALS, seed=SEED):
    """Fit logistic regression and the network on `signals`; thresholds come from validation."""
    x_tr, y_tr = inputs(train, signals), train.label.to_numpy()
    x_va, y_va = inputs(val, signals), val.label.to_numpy()
    logistic, c = fit_logistic(x_tr, y_tr, x_va, y_va)
    net, info = train_network(x_tr, y_tr, x_va, y_va, seed=seed)
    return {
        'logistic': logistic, 'logistic_C': c,
        'logistic_threshold': best_f1_threshold(y_va, logistic.predict_proba(x_va)[:, 1]),
        'net': net, 'net_info': info,
        'net_threshold': best_f1_threshold(y_va, predict_network(net, x_va)),
    }


def score_all(frame, fitted, train_rate, score_threshold, signals=SIGNALS):
    x = inputs(frame, signals)
    return {
        'majority': np.full(len(frame), train_rate),
        'weighted_sum': frame.score.to_numpy() / 100.0,
        'logistic': fitted['logistic'].predict_proba(x)[:, 1],
        'network': predict_network(fitted['net'], x),
    }


def metric_table(frame, probs, thresholds):
    y = frame.label.to_numpy()
    rows = {
        'Majority class': metrics(y, probs['majority'], thresholds['majority']),
        f'Weighted sum (s >= {FIRST_PASS_CUTOFF:g}, first pass)':
            metrics(y, probs['weighted_sum'], FIRST_PASS_CUTOFF / 100.0),
        'Weighted sum (validation-tuned cutoff)':
            metrics(y, probs['weighted_sum'], thresholds['weighted_sum']),
        'Logistic regression (L2)': metrics(y, probs['logistic'], thresholds['logistic']),
        'ReconnectScoreNet (4-16-8-1)': metrics(y, probs['network'], thresholds['network']),
    }
    table = pd.DataFrame(rows).T
    table.index.name = 'procedure'
    return table


def top_partner(frame, value, senders):
    """Highest-valued dormant partner per sender; ties go to the lower partner ID."""
    both = pd.concat([
        pd.DataFrame({'sender': frame.u, 'partner': frame.v, 'value': value}),
        pd.DataFrame({'sender': frame.v, 'partner': frame.u, 'value': value})])
    both = both.loc[both.sender.isin(senders)]
    both = both.sort_values(['sender', 'value', 'partner'], ascending=[True, False, True])
    return both.groupby('sender').partner.first()


def run_ablation(main, train, val, test, full, train_rate):
    """Drop each signal in turn; refit learned models on the remaining three."""
    degree = pd.concat([main.u, main.v]).value_counts()
    rng = np.random.default_rng(SEED)
    eligible = np.sort(degree[degree >= 3].index.to_numpy())
    senders = rng.choice(eligible, size=min(N_RANKING_SENDERS, len(eligible)), replace=False)

    y_test = test.label.to_numpy()
    base_ap = {
        'weighted_sum': average_precision_score(y_test, test.score),
        'logistic': average_precision_score(y_test, full['logistic'].predict_proba(inputs(test))[:, 1]),
        'network': average_precision_score(y_test, predict_network(full['net'], inputs(test))),
    }
    base_top = {
        'weighted_sum': top_partner(main, main.score.to_numpy(), senders),
        'network': top_partner(main, predict_network(full['net'], inputs(main)), senders),
    }
    rows = []
    for dropped in SIGNALS:
        kept = [k for k in SIGNALS if k != dropped]
        fitted = fit_procedures(train, val, kept)
        ap = {
            'weighted_sum': average_precision_score(y_test, reconnect_score(test, drop=dropped)),
            'logistic': average_precision_score(y_test, fitted['logistic'].predict_proba(inputs(test, kept))[:, 1]),
            'network': average_precision_score(y_test, predict_network(fitted['net'], inputs(test, kept))),
        }
        top = {
            'weighted_sum': top_partner(main, reconnect_score(main, drop=dropped).to_numpy(), senders),
            'network': top_partner(main, predict_network(fitted['net'], inputs(main, kept)), senders),
        }
        row = {'dropped_signal': dropped}
        for name in ('weighted_sum', 'logistic', 'network'):
            row[f'{name}_ap'] = ap[name]
            row[f'{name}_delta_ap'] = ap[name] - base_ap[name]
        for name in ('weighted_sum', 'network'):
            row[f'{name}_top_partner_changed'] = float((top[name] != base_top[name]).mean())
        rows.append(row)
    table = pd.DataFrame(rows)
    return table, base_ap, len(senders)


def error_analysis(test, probs, thresholds):
    rows = []
    for name in ('weighted_sum', 'network'):
        flagged = probs[name] >= thresholds[name]
        fp = test.loc[flagged & (test.label == 0)].copy()
        fp['procedure'] = name
        fp['probability'] = probs[name][flagged & (test.label.to_numpy() == 0)]
        rows.append(fp)
    fps = pd.concat(rows)
    pattern = (fps['T'] >= ERROR_T_MIN) & (fps['S'] <= ERROR_S_MAX)
    negatives = test.loc[test.label == 0]
    base_rate = float(((negatives['T'] >= ERROR_T_MIN) & (negatives['S'] <= ERROR_S_MAX)).mean())
    summary = {name: {'false_positives': int((fps.procedure == name).sum()),
                      'high_T_low_S': int((pattern & (fps.procedure == name)).sum()),
                      'share_high_T_low_S': float(pattern[fps.procedure == name].mean())
                      if (fps.procedure == name).any() else 0.0}
               for name in ('weighted_sum', 'network')}
    summary['share_high_T_low_S_among_all_test_negatives'] = base_rate
    summary['rule'] = f'T >= {ERROR_T_MIN:g} and S <= {ERROR_S_MAX:g}'
    cols = ['procedure', 'u', 'v', 'dept_u', 'dept_v', 'emails_before_t0', 'u_to_v', 'v_to_u',
            'days_silent', 'S', 'H', 'C', 'T', 'score', 'probability', 'label']
    return fps.loc[pattern, cols].sort_values(['procedure', 'probability'], ascending=[True, False]), summary


def plot_pr(test, probs, prevalence):
    # Reference palette, light mode, first three categorical slots (validated all-pairs).
    colors = {'weighted_sum': '#2a78d6', 'logistic': '#eb6834', 'network': '#1baf7a'}
    names = {'weighted_sum': 'Weighted sum', 'logistic': 'Logistic regression',
             'network': 'ReconnectScoreNet'}
    y = test.label.to_numpy()
    fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
    fig.patch.set_facecolor('#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    for key in ('weighted_sum', 'logistic', 'network'):
        precision, recall, _ = precision_recall_curve(y, probs[key])
        ap = average_precision_score(y, probs[key])
        ax.plot(recall, precision, color=colors[key], linewidth=2, label=f'{names[key]} (AP {ap:.3f})')
    ax.axhline(prevalence, color='#8a8984', linewidth=1.2, linestyle='--',
               label=f'Majority / prevalence ({prevalence:.3f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall', color='#52514e')
    ax.set_ylabel('Precision', color='#52514e')
    ax.set_title(f'Precision-recall on test pairs (t0 = day {T0})', color='#0b0b0b', loc='left')
    ax.grid(color='#e6e5e0', linewidth=0.8)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.tick_params(colors='#52514e')
    ax.legend(frameon=False, labelcolor='#0b0b0b', fontsize=9)
    fig.tight_layout()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / 'pr_curve_test.png')
    plt.close(fig)


def main():
    torch.set_num_threads(1)
    main_pairs = pd.read_csv(PROCESSED / 'pairs_main.csv')
    oot = pd.read_csv(PROCESSED / 'pairs_oot.csv')
    train, val, test = (main_pairs.loc[main_pairs.fold == f].reset_index(drop=True)
                        for f in ('train', 'val', 'test'))

    # Debug pass on the synthetic matrix; it never enters a reported number.
    x_syn, y_syn = synthetic_debug_matrix()
    _, syn_info = train_network(x_syn[:3200], y_syn[:3200], x_syn[3200:], y_syn[3200:], seed=0)

    train_rate = float(train.label.mean())
    full = fit_procedures(train, val)
    val_probs = score_all(val, full, train_rate, None)
    thresholds = {
        'majority': 0.5 if train_rate < 0.5 else 0.0,
        'weighted_sum': best_f1_threshold(val.label, val_probs['weighted_sum']),
        'logistic': full['logistic_threshold'],
        'network': full['net_threshold'],
    }

    test_probs = score_all(test, full, train_rate, None)
    oot_probs = score_all(oot, full, train_rate, None)
    table = metric_table(test, test_probs, thresholds)
    table_oot = metric_table(oot, oot_probs, thresholds)
    table.to_csv(REPORTS / 'metrics_test.csv', float_format='%.4f')
    table_oot.to_csv(REPORTS / 'metrics_oot.csv', float_format='%.4f')

    seed_rows = []
    for seed in range(N_SEEDS):
        net, _ = train_network(inputs(train), train.label.to_numpy(), inputs(val), val.label.to_numpy(), seed=seed)
        thr = best_f1_threshold(val.label, predict_network(net, inputs(val)))
        seed_rows.append({'seed': seed, **metrics(test.label.to_numpy(), predict_network(net, inputs(test)), thr)})
    seeds = pd.DataFrame(seed_rows)
    seeds.to_csv(REPORTS / 'network_seeds.csv', index=False, float_format='%.4f')

    ablation, base_ap, n_senders = run_ablation(main_pairs, train, val, test, full, train_rate)
    ablation.to_csv(REPORTS / 'ablation.csv', index=False, float_format='%.4f')

    fps, fp_summary = error_analysis(test, test_probs, thresholds)
    fps.to_csv(REPORTS / 'error_analysis_fp.csv', index=False, float_format='%.3f')

    plot_pr(test, test_probs, float(test.label.mean()))

    for frame, probs, name in ((main_pairs, score_all(main_pairs, full, train_rate, None), 'main'),
                               (oot, oot_probs, 'oot')):
        out = frame[['u', 'v', 'fold', 'label']].copy()
        out['p_logistic'] = probs['logistic']
        out['p_network'] = probs['network']
        out.to_csv(PROCESSED / f'predictions_{name}.csv', index=False)
    torch.save(full['net'].state_dict(), PROCESSED / 'reconnect_score_net.pt')

    history = full['net_info']['history']
    details = {
        't0': T0, 't0_out_of_time': T0_OOT,
        'train_positive_rate': train_rate,
        'thresholds_chosen_on_validation': thresholds,
        'weighted_sum_cutoff_on_score_scale': {'first_pass': FIRST_PASS_CUTOFF,
                                               'validation_tuned': 100 * thresholds['weighted_sum']},
        'weights': WEIGHTS,
        'logistic': {'C': full['logistic_C'],
                     'coefficients_per_signal_div_100': dict(zip(SIGNALS, full['logistic'].coef_[0].round(4).tolist())),
                     'intercept': float(full['logistic'].intercept_[0])},
        'network': {'best_epoch': full['net_info']['best_epoch'], 'epochs_run': len(history),
                    'best_val_loss': full['net_info']['best_val_loss'],
                    'final_train_loss': history[-1][1]},
        'network_seeds_test': {m: {'mean': float(seeds[m].mean()), 'std': float(seeds[m].std())}
                               for m in ('roc_auc', 'average_precision', 'f1', 'brier')},
        'ablation_baseline_test_ap': base_ap,
        'ablation_ranking_senders': n_senders,
        'error_analysis': fp_summary,
        'synthetic_debug': {'rows': len(y_syn), 'best_val_loss': syn_info['best_val_loss'],
                            'note': 'training-loop check only; not a reported result'},
    }
    (REPORTS / 'model_details.json').write_text(json.dumps(details, indent=2, default=float))

    pd.set_option('display.width', 200)
    print(f'Test pairs at t0 = {T0}:')
    print(table.round(3).to_string())
    print(f'\nOut-of-time pairs at t0 = {T0_OOT}:')
    print(table_oot.round(3).to_string())
    print('\nAblation:')
    print(ablation.round(3).to_string(index=False))
    print('\nError analysis:', json.dumps(fp_summary, indent=2))
    print('\nNetwork over seeds:', json.dumps(details['network_seeds_test'], indent=2))
    print('\nLogistic:', details['logistic'])


if __name__ == '__main__':
    main()
