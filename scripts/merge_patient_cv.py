# Merges the per-seed patient-level CV runs and compares the configurations.
"""
Running `run_patient_cv.py` as one array task per seed scatters the results across
`out/patient_cv/{task}/{model}/seed{N}/`. This gathers them and produces the confidence
intervals across folds and the paired comparisons between configurations.

Intervals come from the spread across folds; segments are never treated as independent.
Configurations are compared with a paired McNemar test on patient-level predictions, and the
patient-level aggregate is the primary result.

Usage.
    python scripts/merge_patient_cv.py --task task1 --out out/patient_cv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.metrics import majority_baseline, mcnemar_paired, metrics_table  # noqa: E402

MODELS = ["1d", "2d", "fusion",
          "temporal_matched", "resnet1d_matched", "tcn_matched"]


def class_names_for(root: Path, task: str, model: str):
    """Read the class names from whichever seed directory has them, falling back to integers."""
    import json
    for d in sorted((root / task / model).glob("seed*")):
        f = d / "config.json"
        if f.exists():
            try:
                return json.loads(f.read_text())["class_names"]
            except Exception:
                pass
    return None


def collect(root: Path, task: str, model: str):
    base = root / task / model
    folds, pats = [], []
    for d in sorted(base.glob("seed*")):
        f = d / "fold_summary.csv"
        p = d / "all_patient_predictions.csv"
        if f.exists():
            folds.append(pd.read_csv(f))
        if p.exists():
            pats.append(pd.read_csv(p))
    if not folds:
        return None, None
    return pd.concat(folds, ignore_index=True), (
        pd.concat(pats, ignore_index=True) if pats else None)


def ci(series):
    m, s, n = series.mean(), series.std(), len(series)
    h = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return m, s, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--out", type=Path, default=Path("out/patient_cv"))
    a = ap.parse_args()

    summ, preds, missing = {}, {}, []
    for m in MODELS:
        f, p = collect(a.out, a.task, m)
        if f is None:
            missing.append(m)
            continue
        summ[m], preds[m] = f, p

    if missing:
        print(f"[incomplete] {', '.join(missing)}\n")
    if not summ:
        print("no results found.")
        return

    print(f"===== {a.task}: patient-level cross-validation =====")
    cols = ["segment_accuracy", "segment_macro_f1", "patient_accuracy",
            "patient_macro_f1", "majority_accuracy"]
    rows = []
    for m, f in summ.items():
        r = {"model": m, "n_runs": len(f)}
        for c in cols:
            mu, sd, lo, hi = ci(f[c])
            r[c] = f"{mu:.4f} +/- {sd:.4f}"
            r[c + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(r)
    tab = pd.DataFrame(rows)
    print(tab[["model", "n_runs", "patient_accuracy", "patient_accuracy_ci",
               "patient_macro_f1", "patient_macro_f1_ci"]].to_string(index=False))
    print()
    print(tab[["model", "segment_accuracy", "segment_macro_f1",
               "majority_accuracy"]].to_string(index=False))
    tab.to_csv(a.out / f"{a.task}_summary.csv", index=False)

    # paired comparison between configurations, at patient level
    print("\n=== paired McNemar between configurations (patient level, pooled) ===")
    avail = [m for m in MODELS if preds.get(m) is not None]
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            A, B = avail[i], avail[j]
            # Compare only on the (seed, fold, patient) triples both configurations have.
            # When one has fewer seeds, use the overlap rather than discarding everything.
            key = ["seed", "fold", "patient_id"]
            a_df = preds[A].drop_duplicates(key).set_index(key)
            b_df = preds[B].drop_duplicates(key).set_index(key)
            common = a_df.index.intersection(b_df.index)
            if len(common) < 10:
                print(f"  {A} vs {B}: only {len(common)} shared predictions, skipping")
                continue
            a_c, b_c = a_df.loc[common], b_df.loc[common]
            assert (a_c.y_true.values == b_c.y_true.values).all(), "labels do not line up"
            r = mcnemar_paired(a_c.y_true.values, a_c.y_pred.values, b_c.y_pred.values)
            n_seed = len(set(i[0] for i in common))
            print(f"  {A:16s} vs {B:16s}  acc {r['accuracy_a']:.4f} vs {r['accuracy_b']:.4f} | "
                  f"n={len(common)} ({n_seed} seeds) | discordant {r['n_discordant']:3d} "
                  f"({r['n_a_only_correct']}/{r['n_b_only_correct']}) | "
                  f"p={r['p_value']:.4f}" + ("  <== significant" if r["p_value"] < 0.05 else ""))

    # per-class table, patient level, pooled over folds
    for m in avail:
        p = preds[m]
        cn = class_names_for(a.out, a.task, m)
        n_cls = len(cn) if cn else int(max(p.y_true.max(), p.y_pred.max())) + 1
        names = cn if cn else [str(i) for i in range(n_cls)]
        print(f"\n--- {m}: pooled patient level ({len(p)} predictions) ---")
        t = metrics_table(p.y_true.values, p.y_pred.values, names)
        print(t[["class_name", "sensitivity", "specificity", "f1_score", "support"]]
              .to_string(index=False, float_format=lambda v: f"{v*100:.2f}"))
        b = majority_baseline(p.y_true.values, names)
        print(f"    majority baseline accuracy {b['accuracy_plain']:.4f} | "
              f"macro-F1 {b['macro_f1']:.4f}")

    print(f"\nwrote {a.out / (a.task + '_summary.csv')}")


if __name__ == "__main__":
    main()
