# Metrics are computed from raw predictions, not from the archive's normalize-then-multiply path.
"""Why this module exists rather than reusing what produced the published tables.

The archived pipeline normalized each confusion matrix with `normalize='true'`, stored it to
four decimal places, and then had `results_selection.ipynb` multiply those fractions back by a
hard-coded support to recover counts. The round trip loses precision: 17 of the 120 cells in
the manuscript's Tables 2 and 3 disagree in the second decimal place, all by 0.015 percentage
points or less. The AS-TR row of the Task II fusion run, for instance, sums to 1.0001.

Here everything is derived from `y_true` and `y_pred` against an integer confusion matrix. The
predictions themselves are written to `predictions.csv` so that any metric can be recomputed
afterwards and so that paired tests are possible at all.

The Overall row is a micro-average. As Referee 2 pointed out, under that definition
sensitivity, specificity, accuracy and F1 are algebraically dependent -- see
`overall_metrics_are_dependent` -- so reporting all four is close to writing one number four
times.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def per_class_metrics(y_true, y_pred, class_names) -> pd.DataFrame:
    """Per-class Se/Sp/Ac/F1 with the underlying TP/FP/FN/TN, one class against the rest."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    total = cm.sum()
    rows = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        rows.append({
            "class_name": name,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "accuracy": (tp + tn) / total if total else np.nan,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
            "support": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return pd.DataFrame(rows)


def overall_metrics(y_true, y_pred, n_classes) -> dict:
    """Micro-average, matching the definition behind the manuscript's Overall row."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()
    tp = int(np.trace(cm))
    fn = fp = total - tp
    tn = total * n_classes - tp - fp - fn
    return {
        "class_name": "Overall",
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
        "support": int(total), "TP": tp, "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def macro_metrics(y_true, y_pred, class_names) -> dict:
    pc = per_class_metrics(y_true, y_pred, class_names)
    return {
        "macro_sensitivity": float(pc.sensitivity.mean()),
        "macro_specificity": float(pc.specificity.mean()),
        "macro_f1": float(pc.f1_score.mean()),
        "balanced_accuracy": float(pc.sensitivity.mean()),
    }


def metrics_table(y_true, y_pred, class_names) -> pd.DataFrame:
    """Per-class rows plus an Overall row, laid out like Tables 2 and 3 of the manuscript."""
    pc = per_class_metrics(y_true, y_pred, class_names)
    ov = pd.DataFrame([overall_metrics(y_true, y_pred, len(class_names))])
    return pd.concat([pc, ov], ignore_index=True)


def majority_baseline(y_true, class_names) -> dict:
    """Always answer the most common class. Patient-level numbers mean nothing without it."""
    y_true = np.asarray(y_true)
    maj = int(pd.Series(y_true).value_counts().idxmax())
    y_pred = np.full_like(y_true, maj)
    ov = overall_metrics(y_true, y_pred, len(class_names))
    plain = float((y_pred == y_true).mean())
    return {
        "majority_class": class_names[maj],
        # The same one-vs-rest micro scale as the manuscript's Overall row: what doing
        # nothing already scores on that scale.
        "accuracy_onevsrest": ov["accuracy"],
        # How many were actually right. Read this one.
        "accuracy_plain": plain,
        "micro_sensitivity": ov["sensitivity"],
        **macro_metrics(y_true, y_pred, class_names),
    }


def aggregate_to_patient(df_pred: pd.DataFrame, n_classes: int, how="mean_prob") -> pd.DataFrame:
    """Aggregate segment predictions into one prediction per patient.

    `df_pred` carries patient_id, y_true, y_pred and prob_0..prob_{n-1}. Clinical diagnosis is
    made per patient, so the patient-level table is reported alongside every segment-level one
    rather than offered on request.
    """
    prob_cols = [f"prob_{i}" for i in range(n_classes)]
    rows = []
    for pid, g in df_pred.groupby("patient_id"):
        if how == "mean_prob":
            pred = int(np.argmax(g[prob_cols].mean(axis=0).values))
        elif how == "majority_vote":
            pred = int(g.y_pred.value_counts().idxmax())
        else:
            raise ValueError(how)
        rows.append({"patient_id": pid, "y_true": int(g.y_true.iloc[0]),
                     "y_pred": pred, "n_segments": len(g)})
    return pd.DataFrame(rows)


def mcnemar_paired(y_true, pred_a, pred_b) -> dict:
    """Paired comparison of two configurations, run on patient-level predictions.

    Paired, as Referee 2 asked, so segments are never treated as independent samples. The
    sample is small, so this uses the exact binomial test rather than the chi-square
    approximation.
    """
    from scipy import stats
    y_true = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    a_ok, b_ok = a == y_true, b == y_true
    n01 = int((~a_ok & b_ok).sum())   # a wrong, b right
    n10 = int((a_ok & ~b_ok).sum())   # a right, b wrong
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n10, n, 0.5).pvalue)
    return {"n_a_only_correct": n10, "n_b_only_correct": n01,
            "n_discordant": n, "p_value": p,
            "accuracy_a": float(a_ok.mean()), "accuracy_b": float(b_ok.mean())}


def overall_metrics_are_dependent(n_classes: int, sensitivity: float) -> dict:
    """Demonstrate that under micro-averaging, Sp and Ac follow algebraically from Se.

    In a C-class micro-average, FP = FN = N(1-Se), hence
        Sp = 1 - (1-Se)/(C-1),   Ac = 1 - 2(1-Se)/C,   F1 = Se
    holds exactly.
    """
    c, se = n_classes, sensitivity
    return {"sensitivity": se,
            "specificity_implied": 1 - (1 - se) / (c - 1),
            "accuracy_implied": 1 - 2 * (1 - se) / c,
            "f1_implied": se}
