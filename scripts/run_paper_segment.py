# Reproduces the six segment-level runs behind the paper and checks them cell by cell against
# the manuscript's Tables 2 and 3.
"""
What this reproduces is the experiment that produced the published numbers, not an estimate
of generalization. The split is at segment level, so 98.96% of Task I test segments have a
50%-overlapping neighbour in training or validation. The run prints that figure alongside the
results rather than leaving it implicit.

What to expect. Of the 120 published cells, 103 match exactly and 17 differ in the second
decimal place. The difference comes from the archive storing normalized confusion matrices and
multiplying back to recover counts; everything here is computed from raw predictions, so a new
value differing from the published one is correct rather than a failure. Training randomness
means cells will not reproduce exactly in any case. The point of the script is to record
(a) whether the pipeline lands in the same place and (b) which cells differ.

Usage.
    python scripts/run_paper_segment.py --task task1 --models 1d 2d fusion --out out/paper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.paths import data_root, localize  # noqa: E402

from scg_hvd.metrics import (majority_baseline, metrics_table,  # noqa: E402
                             overall_metrics_are_dependent)
from scg_hvd.splits import (check_patient_disjoint, quantify_overlap_leakage,  # noqa: E402
                            segment_split)
from scg_hvd.datasets import WAVEFORM_MODELS  # noqa: E402
from scg_hvd.train import TrainConfig, run_training  # noqa: E402

DATA = data_root(required=False)
IMAGE_DIRS = {"task1": DATA / "Task1_images", "task2": DATA / "Task2_images"}
EPOCHS = {"1d": 25, "2d": 25, "fusion": 25}   # as in the canonical bash_script.sh


def load_task(task):
    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    names = sorted(df.label.unique())
    label_map = {n: i for i, n in enumerate(names)}
    df["label_name"] = df["label"]
    df["label"] = df["label"].map(label_map)
    # Repoint the signal paths at the local copy.
    df["filepath"] = localize(df["filepath"], DATA)
    return df, names


def main():
    global DATA
    DATA = data_root()   # fail here rather than on a puzzling missing file
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1", choices=["task1", "task2"])
    ap.add_argument("--models", nargs="+", default=["1d", "2d", "fusion"])
    ap.add_argument("--out", type=Path, default=Path("out/paper"))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    df, class_names = load_task(a.task)
    split_df = segment_split(df)
    n_classes = len(class_names)

    root = a.out / a.task
    root.mkdir(parents=True, exist_ok=True)

    # --- record the split diagnostics before training anything ---
    diag = {
        "task": a.task, "n_segments": len(df), "n_patients": df.patient_id.nunique(),
        "classes": class_names,
        "n_test": int((split_df.split == "test").sum()),
        "patients_split_across": check_patient_disjoint(split_df)["violations"],
        "overlap_vs_train": quantify_overlap_leakage(split_df, against=("train",)),
        "overlap_vs_train_val": quantify_overlap_leakage(split_df, against=("train", "val")),
    }
    (root / "split_diagnostics.json").write_text(json.dumps(diag, indent=2, default=str))
    split_df[["segment_id", "patient_id", "label_name", "start_time_sec", "split"]] \
        .to_csv(root / "frozen_split.csv", index=False)

    print(f"===== {a.task}: segment-level split diagnostics =====")
    print(f"  {len(df)} segments, {df.patient_id.nunique()} patients, classes {class_names}")
    print(f"  {diag['n_test']} test segments | "
          f"{diag['patients_split_across']} patients span two splits")
    print(f"  overlapping neighbour, vs train {diag['overlap_vs_train']['fraction']:.4f}, "
          f"train+val {diag['overlap_vs_train_val']['fraction']:.4f}")
    print("  -> this split reproduces the paper; it does not estimate generalization.\n")

    summary = []
    for m in a.models:
        print(f"===== {a.task} / {m} =====", flush=True)
        cfg = TrainConfig(
            model=m, num_classes=n_classes,
            epochs=a.epochs or EPOCHS.get(m, 25),
            lr=1e-3, num_workers=a.num_workers, seed=a.seed,
            class_weight_scope="all",     # as the canonical code did, since this reproduces it
        )
        out_dir = root / m
        _, (y_true, y_pred, _), res = run_training(
            split_df, cfg, image_dir=None if m in WAVEFORM_MODELS else IMAGE_DIRS[a.task],
            out_dir=out_dir, class_names=class_names,
        )
        tab = metrics_table(y_true, y_pred, class_names)
        tab.to_csv(out_dir / "metrics.csv", index=False, float_format="%.6f")
        ov = tab[tab.class_name == "Overall"].iloc[0]
        print(tab[["class_name", "sensitivity", "specificity", "accuracy", "f1_score"]]
              .to_string(index=False, float_format=lambda v: f"{v*100:.2f}"))
        print(f"  best val acc {res['best_val_acc']:.4f} | {res['elapsed_sec']:.0f}s\n", flush=True)
        summary.append({"model": m, **{k: float(ov[k]) for k in
                        ["sensitivity", "specificity", "accuracy", "f1_score"]},
                        "best_val_acc": res["best_val_acc"],
                        "elapsed_sec": res["elapsed_sec"]})

    sm = pd.DataFrame(summary)
    sm.to_csv(root / "summary_overall.csv", index=False, float_format="%.6f")
    print("===== Overall summary (%) =====")
    print(sm.assign(**{c: (sm[c] * 100).round(2) for c in
                       ["sensitivity", "specificity", "accuracy", "f1_score"]})
          .to_string(index=False))

    base = majority_baseline(split_df[split_df.split == "test"].label.values, class_names)
    print(f"\nmajority-class baseline ({base['majority_class']}) on the test set")
    print(f"  plain accuracy      {base['accuracy_plain']*100:.2f}%")
    print(f"  one-vs-rest scale   {base['accuracy_onevsrest']*100:.2f}%  "
          f"<- same definition as the manuscript's Overall row")
    print(f"  macro-F1           {base['macro_f1']*100:.2f}%")

    # Show that the four Overall metrics all follow from Se under this definition.
    if len(sm):
        se = float(sm.iloc[-1].sensitivity)
        d = overall_metrics_are_dependent(n_classes, se)
        print(f"\nGiven Se {se*100:.2f}, the definition forces "
              f"Sp {d['specificity_implied']*100:.2f}, Ac {d['accuracy_implied']*100:.2f} and "
              f"F1 {d['f1_implied']*100:.2f}.")
    print(f"\noutput: {root}")


if __name__ == "__main__":
    main()
