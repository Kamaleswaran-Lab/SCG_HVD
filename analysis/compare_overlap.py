# Puts the overlapping-window and non-overlapping-window runs side by side, at patient level.
"""
One of the two supplementary analyses Referee 2 asked for.

    "A supplementary analysis using nonoverlapping windows and, where feasible, group-wise
     splitting for classes with sufficient patient counts would strengthen the work."

`run_patient_cv.py` provides the group-wise splitting; this script compares two runs of it
that differ only in whether windows overlap.

One distinction has to be kept straight when reading the output. Keeping only every second
window halves the training set (8678 -> 4365 segments on Task I). So a drop in performance
would not by itself mean that overlap was producing the performance. Under a patient-level
split overlap never crosses from training into test, so there is no leakage path left, and
what remains is mostly the smaller sample. The segment counts are printed alongside so the
two readings can be told apart.

Usage.
    python analysis/compare_overlap.py --task task1 --out out/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.final_report import LABEL, collect  # noqa: E402
from scg_hvd.metrics import mcnemar_paired  # noqa: E402

MODELS = ["1d", "2d", "fusion"]


def ci(v):
    m, s, n = float(v.mean()), float(v.std()), len(v)
    h = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return m, s, m - h, m + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--overlap", type=Path, default=Path("out/patient_cv"))
    ap.add_argument("--nonoverlap", type=Path, default=Path("out/patient_cv_nonoverlap"))
    ap.add_argument("--out", type=Path, default=Path("out/final"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    rows = []
    for m in MODELS:
        for tag, root in (("overlapping", a.overlap), ("non-overlapping", a.nonoverlap)):
            r = collect(root, a.task, m)
            if r is None:
                continue
            f = r["folds"]
            pa, pf = ci(f.patient_accuracy), ci(f.patient_macro_f1)
            rows.append({
                "model": m, "windows": tag, "n_folds": len(f), "n_seeds": f.seed.nunique(),
                "test_segments": int(f.n_test_segments.sum()),
                "pat_acc": pa[0], "pat_acc_sd": pa[1], "pat_acc_lo": pa[2], "pat_acc_hi": pa[3],
                "pat_f1": pf[0], "pat_f1_sd": pf[1],
                "majority": float(f.majority_accuracy.mean()),
            })
    if not rows:
        print("no results found."); return
    t = pd.DataFrame(rows)
    t.to_csv(a.out / f"{a.task}_overlap_comparison.csv", index=False)

    print(f"===== {a.task}: overlapping vs non-overlapping windows (patient level) =====")
    print(f"{'configuration':22s} {'windows':16s} {'seed':>4s} {'test seg':>9s} "
          f"{'patient accuracy (95% CI)':>26s} {'macro-F1':>10s}")
    for _, r in t.iterrows():
        print(f"{LABEL[r.model]:22s} {r.windows:16s} {int(r.n_seeds):>4d} {r.test_segments:>9,} "
              f"{r.pat_acc:.4f} [{r.pat_acc_lo:.4f}, {r.pat_acc_hi:.4f}]  {r.pat_f1:>9.4f}")
    print(f"{'majority baseline':22s} {'':16s} {'':>4s} {'':>9s} {t.majority.mean():.4f}")

    print(f"\n===== effect of removing the overlap =====")
    for m in MODELS:
        sub = t[t.model == m]
        if len(sub) != 2:
            continue
        ov = sub[sub.windows == "overlapping"].iloc[0]
        no = sub[sub.windows == "non-overlapping"].iloc[0]
        d = no.pat_acc - ov.pat_acc
        overlap_ci = (max(ov.pat_acc_lo, no.pat_acc_lo) <= min(ov.pat_acc_hi, no.pat_acc_hi))
        both_above = no.pat_acc_lo > no.majority and ov.pat_acc_lo > ov.majority
        print(f"  {LABEL[m]:22s} {ov.pat_acc:.4f} -> {no.pat_acc:.4f} ({d:+.4f}) | "
              f"CI {'overlap' if overlap_ci else 'disjoint'} | "
              f"both above baseline: {'yes' if both_above else 'no'} | "
              f"segments {ov.test_segments:,} -> {no.test_segments:,}")

    print("\nRead this carefully. Dropping the overlap halves the training sample. Under a")
    print("patient-level split overlap never crosses into the test set, so no leakage path")
    print("remains and what is left mostly reflects the smaller sample. Where the two")
    print("intervals overlap there is no distinction worth arguing about either way.")

    # A paired test only means something on the same patients, so this is restricted to the
    # (seed, fold, patient) triples the two runs share.
    print(f"\n===== paired comparison on shared (seed, fold, patient) =====")
    key = ["seed", "fold", "patient_id"]
    for m in MODELS:
        ro = collect(a.overlap, a.task, m)
        rn = collect(a.nonoverlap, a.task, m)
        if not (ro and rn and ro["preds"] is not None and rn["preds"] is not None):
            continue
        da = ro["preds"].drop_duplicates(key).set_index(key)
        db = rn["preds"].drop_duplicates(key).set_index(key)
        common = da.index.intersection(db.index)
        if len(common) < 10:
            print(f"  {LABEL[m]}: only {len(common)} shared predictions, skipping"); continue
        ca, cb = da.loc[common], db.loc[common]
        r = mcnemar_paired(ca.y_true.values, ca.y_pred.values, cb.y_pred.values)
        flag = "  <== significant" if r["p_value"] < 0.05 else ""
        print(f"  {LABEL[m]:22s} overlapping {r['accuracy_a']:.4f} vs non-overlapping "
              f"{r['accuracy_b']:.4f} | n={len(common)} | p={r['p_value']:.4f}{flag}")

    print(f"\nwrote {a.out / (a.task + '_overlap_comparison.csv')}")


if __name__ == "__main__":
    main()
