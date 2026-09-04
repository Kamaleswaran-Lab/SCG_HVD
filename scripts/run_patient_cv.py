# Patient-level cross-validation: the generalization evaluation the revision rests on.
"""
Six things this run is built to get right.

1. Only the split differs from the published experiment. The model, the preprocessing and the
   optimizer take the same code path as `run_paper_segment.py`, so any difference in
   performance can be attributed to the split by construction rather than by argument.
2. The validation set never sees the test patients' labels. This is the defect in the original
   LOOCV harness (`exp_loocv/evaluate.py:122-131`) that we are not repeating.
3. Class weights come from the training split alone (`class_weight_scope="train"`). The
   reproduction runs keep the original behavior of using everything; new experiments do not.
4. Both segment- and patient-level results are written. Clinical diagnosis is per patient, so
   the patient table is the primary one.
5. The majority-class baseline goes out with every result. Without it the numbers cannot be
   read.
6. Raw predictions are saved. Paired McNemar tests and any later recomputation come from
   these.

Usage.
    python scripts/run_patient_cv.py --task task1 --model fusion --folds 5 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.paths import data_root, localize  # noqa: E402

from scg_hvd.metrics import (aggregate_to_patient, kappa, macro_metrics,  # noqa: E402
                             majority_baseline, metrics_table)
from scg_hvd.splits import (check_patient_disjoint, patient_cv_folds,  # noqa: E402
                            quantify_overlap_leakage)
from scg_hvd.datasets import WAVEFORM_MODELS  # noqa: E402
from scg_hvd.models import MODELS  # noqa: E402
from scg_hvd.train import TrainConfig, run_training  # noqa: E402

DATA = data_root(required=False)
IMAGE_DIRS = {"task1": DATA / "Task1_images", "task2": DATA / "Task2_images"}


def drop_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only non-overlapping windows.

    Windows are 10 s on a 5 s stride, so taking every second segment leaves start times exactly
    10.0 s apart within a patient and no overlap anywhere. Verified on Task I: 8678 -> 4365
    segments, zero 5 s neighbor pairs, and the per-class patient counts unchanged.
    """
    idx = df.segment_id.str.extract(r"_seg(\d+)$")[0].astype(int)
    return df[idx % 2 == 0].reset_index(drop=True)


def load_task(task, nonoverlap=False):
    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    if nonoverlap:
        n0 = len(df)
        df = drop_overlap(df)
        print(f"  [non-overlapping] segments {n0} -> {len(df)}", flush=True)
    names = sorted(df.label.unique())
    df["label_name"] = df["label"]
    df["label"] = df["label"].map({n: i for i, n in enumerate(names)})
    df["filepath"] = localize(df["filepath"], DATA)
    return df, names


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1", choices=["task1", "task2"])
    ap.add_argument("--model", default="fusion", choices=sorted(MODELS),
                    help="a name registered in scg_hvd.models.MODELS")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/patient_cv"))
    ap.add_argument("--nonoverlap", action="store_true",
                    help="use non-overlapping windows only")
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="keep the best weights per fold, for the interpretability analysis")
    ap.add_argument("--shuffle-labels", action="store_true",
                    help="permute labels within each patient group; a control, not an experiment")
    a = ap.parse_args()
    DATA = data_root()   # fail here rather than on a puzzling missing file

    df, class_names = load_task(a.task, nonoverlap=a.nonoverlap)
    n_classes = len(class_names)

    if a.shuffle_labels:
        # Permute the label assigned to each patient, keeping every patient's segments
        # consistent with one another. Shuffling per segment instead would leave a patient
        # holding several labels, which no real split does and which the model could exploit.
        # The point of the control is a model that has fitted something but nothing about the
        # classes, so that an attribution method run on it shows what that method produces in
        # the absence of class structure.
        rng = np.random.default_rng(12345)
        pt = df.groupby("patient_id").label.agg(lambda x: x.value_counts().idxmax())
        permuted = dict(zip(pt.index, rng.permutation(pt.values)))
        df["label"] = df.patient_id.map(permuted)
        moved = int((pt.values != np.array([permuted[i] for i in pt.index])).sum())
        print(f"  [control] labels permuted across patients: {moved} of {len(pt)} changed\n",
              flush=True)
    # With a single seed, put it in the output path so parallel array tasks do not collide.
    root = a.out / a.task / a.model
    if len(a.seeds) == 1:
        root = root / f"seed{a.seeds[0]}"
    root.mkdir(parents=True, exist_ok=True)

    tag_no = " | non-overlapping windows" if a.nonoverlap else ""
    print(f"===== {a.task} / {a.model} | {a.folds}-fold x {len(a.seeds)} seed{tag_no} =====")
    print(f"  {len(df)} segments, {df.patient_id.nunique()} patients, classes {class_names}\n",
          flush=True)

    seg_rows, pat_rows, fold_summ = [], [], []

    for seed in a.seeds:
        for k, split_df in patient_cv_folds(df, n_splits=a.folds, random_state=seed):
            tag = f"seed{seed}_fold{k}"
            v = check_patient_disjoint(split_df)["violations"]
            lk = quantify_overlap_leakage(split_df, against=("train", "val"))
            assert v == 0, f"{tag}: {v} patient(s) span two splits"
            assert lk["n_with_overlapping_neighbour"] == 0, f"{tag}: overlapping-window leakage"

            te = split_df[split_df.split == "test"]
            print(f"--- {tag}: {te.patient_id.nunique()} test patients, {len(te)} segments ---",
                  flush=True)

            cfg = TrainConfig(model=a.model, num_classes=n_classes, epochs=a.epochs,
                              lr=1e-3, num_workers=a.num_workers, seed=seed,
                              class_weight_scope="train",   # training split only, unlike the paper
                              save_checkpoint=a.save_checkpoint)
            out_dir = root / tag
            _, (y_true, y_pred, y_prob), res = run_training(
                split_df, cfg,
                image_dir=None if a.model in WAVEFORM_MODELS else IMAGE_DIRS[a.task],
                out_dir=out_dir, class_names=class_names, verbose=False)

            # segment level
            seg_tab = metrics_table(y_true, y_pred, class_names)
            seg_tab.to_csv(out_dir / "segment_metrics.csv", index=False, float_format="%.6f")
            seg_ov = seg_tab[seg_tab.class_name == "Overall"].iloc[0]
            seg_macro = macro_metrics(y_true, y_pred, class_names)

            # patient level
            pred = pd.read_csv(out_dir / "predictions.csv")
            pt = aggregate_to_patient(pred, n_classes)
            pt.to_csv(out_dir / "patient_predictions.csv", index=False)
            pat_tab = metrics_table(pt.y_true.values, pt.y_pred.values, class_names)
            pat_tab.to_csv(out_dir / "patient_metrics.csv", index=False, float_format="%.6f")
            pat_acc = float((pt.y_pred == pt.y_true).mean())
            pat_macro = macro_metrics(pt.y_true.values, pt.y_pred.values, class_names)
            base = majority_baseline(pt.y_true.values, class_names)

            print(f"    segment acc {float((y_pred==y_true).mean()):.4f} "
                  f"macro-F1 {seg_macro['macro_f1']:.4f}"
                  f" | patient acc {pat_acc:.4f} "
                  f"({int((pt.y_pred==pt.y_true).sum())}/{len(pt)})"
                  f" macro-F1 {pat_macro['macro_f1']:.4f} | baseline {base['accuracy_plain']:.4f}",
                  flush=True)

            seg_rows.append(pred.assign(seed=seed, fold=k))
            pat_rows.append(pt.assign(seed=seed, fold=k))
            fold_summ.append({
                "seed": seed, "fold": k,
                "n_test_patients": int(pt.shape[0]), "n_test_segments": int(len(te)),
                "segment_accuracy": float((y_pred == y_true).mean()),
                "segment_macro_f1": seg_macro["macro_f1"],
                "segment_overall_se": float(seg_ov.sensitivity),
                "patient_accuracy": pat_acc,
                "patient_macro_f1": pat_macro["macro_f1"],
                "patient_kappa": kappa(pt.y_true.values, pt.y_pred.values),
                "majority_accuracy": base["accuracy_plain"],
                "best_val_acc": res["best_val_acc"], "elapsed_sec": res["elapsed_sec"],
            })

    pd.concat(seg_rows).to_csv(root / "all_segment_predictions.csv", index=False)
    pd.concat(pat_rows).to_csv(root / "all_patient_predictions.csv", index=False)
    fs = pd.DataFrame(fold_summ)
    fs.to_csv(root / "fold_summary.csv", index=False, float_format="%.6f")

    print(f"\n===== {a.task} / {a.model} summary over {len(fs)} folds =====")
    for col in ["segment_accuracy", "segment_macro_f1", "patient_accuracy",
                "patient_macro_f1", "patient_kappa", "majority_accuracy"]:
        m, s = fs[col].mean(), fs[col].std()
        lo, hi = m - 1.96 * s / np.sqrt(len(fs)), m + 1.96 * s / np.sqrt(len(fs))
        print(f"  {col:20s} {m:.4f} +/- {s:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    # patient-level results pooled over every fold
    allp = pd.concat(pat_rows)
    pooled = metrics_table(allp.y_true.values, allp.y_pred.values, class_names)
    pooled.to_csv(root / "pooled_patient_metrics.csv", index=False, float_format="%.6f")
    print(f"\n  pooled patient level ({len(allp)} predictions)")
    print(pooled[["class_name", "sensitivity", "specificity", "accuracy", "f1_score", "support"]]
          .to_string(index=False, float_format=lambda v: f"{v*100:.2f}"))

    (root / "config.json").write_text(json.dumps(
        {"task": a.task, "model": a.model, "folds": a.folds, "seeds": a.seeds,
         "epochs": a.epochs, "class_names": class_names}, indent=2))
    print(f"\noutput: {root}")


if __name__ == "__main__":
    main()
