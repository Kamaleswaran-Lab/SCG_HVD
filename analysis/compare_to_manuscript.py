# Checks the reproduction against all 120 cells of the manuscript's Tables 2 and 3.
"""
It serves two purposes.

1. Confirm that the ported pipeline lands where the manuscript did.
2. Produce the values that go into the revised tables.

The second matters more than it sounds. The archived pipeline stored each confusion matrix
with `normalize='true'` and recovered counts by multiplying back through a hard-coded support;
that round trip is why 17 of the 120 published cells disagree in the second decimal place.
Everything here is computed from raw predictions, so a new value differing from the published
one is expected rather than alarming.

Training randomness means cells will not match exactly in any case. So this reports the
distribution of the differences rather than a pass/fail: how many percentage points the
per-class differences stay within, and whether the Overall rows preserve the same ordering
(1D < 2D < fusion).

Usage.
    python analysis/compare_to_manuscript.py --out out/paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Transcribed from Manuscript_BPEX.tex, Table 2 (lines 395-425) and Table 3 (lines 440-470),
# as (Se, Sp, Ac, F1) percentages.
PAPER = {
    "task1": {
        "AR":  {"1d": (98.44, 99.50, 99.42, 96.18), "2d": (98.44, 99.88, 99.77, 98.44),
                "fusion": (98.44, 100.00, 99.88, 99.21)},
        "AS":  {"1d": (98.11, 99.15, 98.96, 97.21), "2d": (96.25, 99.86, 99.20, 97.78),
                "fusion": (97.50, 99.86, 99.43, 98.43)},
        "MR":  {"1d": (98.71, 99.53, 99.31, 98.71), "2d": (98.71, 99.37, 99.19, 98.50),
                "fusion": (99.57, 99.53, 99.54, 99.14)},
        "MS":  {"1d": (97.18, 99.75, 99.54, 97.18), "2d": (100.00, 99.87, 99.88, 99.30),
                "fusion": (98.59, 100.00, 99.88, 99.29)},
        "N":   {"1d": (98.24, 100.00, 99.31, 99.11), "2d": (100.00, 99.43, 99.66, 99.56),
                "fusion": (99.71, 99.24, 99.43, 99.27)},
        "Overall": {"1d": (98.27, 99.57, 99.31, 98.27), "2d": (98.85, 99.71, 99.54, 98.85),
                    "fusion": (99.08, 99.77, 99.63, 99.08)},
    },
    "task2": {
        "AS":    {"1d": (97.20, 99.02, 98.09, 98.12), "2d": (100.00, 98.04, 99.04, 99.07),
                  "fusion": (100.00, 97.06, 98.56, 98.62)},
        "AS-AR": {"1d": (100.00, 100.00, 100.00, 100.00), "2d": (100.00, 100.00, 100.00, 100.00),
                  "fusion": (100.00, 100.00, 100.00, 100.00)},
        "AS-MR": {"1d": (95.24, 98.41, 98.09, 90.92), "2d": (90.48, 100.00, 99.04, 95.00),
                  "fusion": (90.48, 100.00, 99.04, 95.00)},
        "AS-MS": {"1d": (94.74, 100.00, 99.52, 97.30), "2d": (100.00, 99.47, 99.52, 97.43),
                  "fusion": (100.00, 99.47, 99.52, 97.43)},
        "AS-TR": {"1d": (98.11, 98.72, 98.56, 97.19), "2d": (94.34, 98.72, 97.61, 95.24),
                  "fusion": (96.22, 100.00, 99.04, 98.07)},
        "Overall": {"1d": (97.13, 99.28, 98.85, 97.13), "2d": (97.61, 99.40, 99.04, 97.61),
                    "fusion": (98.08, 99.52, 99.23, 98.08)},
    },
}

METRICS = ["sensitivity", "specificity", "accuracy", "f1_score"]
LABELS = ["Se", "Sp", "Ac", "F1"]


def load_repro(root: Path, task: str, model: str):
    f = root / task / model / "metrics.csv"
    if not f.exists():
        return None
    d = pd.read_csv(f)
    return {r.class_name: tuple(float(r[m]) * 100 for m in METRICS) for _, r in d.iterrows()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/paper"))
    ap.add_argument("--report", type=Path, default=Path("out/paper/comparison.csv"))
    a = ap.parse_args()

    rows, missing = [], []
    for task, per_class in PAPER.items():
        for model in ("1d", "2d", "fusion"):
            rep = load_repro(a.out, task, model)
            if rep is None:
                missing.append(f"{task}/{model}")
                continue
            for cls, by_model in per_class.items():
                if cls not in rep:
                    continue
                paper_v, rep_v = by_model[model], rep[cls]
                for lab, pv, rv in zip(LABELS, paper_v, rep_v):
                    rows.append({"task": task, "model": model, "class": cls, "metric": lab,
                                 "paper": pv, "repro": round(rv, 2), "diff": round(rv - pv, 2)})

    if missing:
        print(f"[incomplete] {', '.join(missing)}\n")
    if not rows:
        print("no reproduction results yet.")
        return

    df = pd.DataFrame(rows)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.report, index=False)

    print(f"=== summary over {len(df)} cells ===")
    ad = df["diff"].abs()
    print(f"  |diff| median {ad.median():.2f}pp | mean {ad.mean():.2f}pp | max {ad.max():.2f}pp")
    for thr in (0.5, 1.0, 2.0, 5.0):
        print(f"  |diff| <= {thr:>4.1f}pp : {(ad <= thr).sum():3d} / {len(df)}  "
              f"({(ad<=thr).mean()*100:.0f}%)")

    print("\n=== Overall rows ===")
    ov = df[df["class"] == "Overall"].pivot_table(index=["task", "model"], columns="metric",
                                                  values=["paper", "repro"])
    print(ov.round(2).to_string())

    print("\n=== does the ordering hold? (Overall Se) ===")
    for task in PAPER:
        sub = df[(df.task == task) & (df["class"] == "Overall") & (df.metric == "Se")]
        if len(sub) < 3:
            print(f"  {task}: incomplete")
            continue
        p = {r.model: r.paper for _, r in sub.iterrows()}
        r_ = {r.model: r.repro for _, r in sub.iterrows()}
        order_p = sorted(p, key=p.get)
        order_r = sorted(r_, key=r_.get)
        ok = "same" if order_p == order_r else "DIFFERENT"
        print(f"  {task}: paper {' < '.join(order_p)} | repro {' < '.join(order_r)}  -> {ok}")

    print("\n=== cells differing by more than 3pp ===")
    big = df[df["diff"].abs() > 3].sort_values("diff", key=abs, ascending=False)
    print(big.to_string(index=False) if len(big) else "  none")

    print(f"\nfull table: {a.report}")


if __name__ == "__main__":
    main()
