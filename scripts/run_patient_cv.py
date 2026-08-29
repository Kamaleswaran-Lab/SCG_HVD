# 환자 단위 교차검증. 리뷰어가 요구한 일반화 평가의 본체다.
"""
이 스크립트 하나가 아래 코멘트들을 함께 겨냥한다.

    R1-M1 / R2-M1   세그먼트 단위 분할 + 50% 겹침 -> 환자 단위로 바꾼다
    R1-M2           unseen patient 로의 일반화 -> group-wise 분할
    R1-m4 / R2-M3   환자 단위 집계 결과
    R1-m3 / R2-M5   다중 시드, fold 간 신뢰구간, 짝지은 검정

설계에서 지킨 것.

1. **분할만 다르고 나머지는 논문과 같다.** 모델·전처리·옵티마이저가 `run_paper_segment.py` 와
   동일한 코드 경로를 쓴다. 성능 차이가 분할에서 왔음을 코드 구조로 보인다.
2. **검증 세트가 테스트 환자의 라벨을 보지 않는다.** 아카이브 LOOCV 하니스의 결함
   (`exp_loocv/evaluate.py:122-131`)을 반복하지 않는다.
3. **클래스 가중치를 학습 분할에서만 계산한다** (`class_weight_scope="train"`). 논문 재현에서는
   정본대로 전체를 썼지만, 새 실험은 R1-m2 를 만족해야 한다.
4. **세그먼트·환자 두 수준을 모두 낸다.** 임상 진단은 환자 단위이므로 환자 표가 주 결과다.
5. **다수 클래스 기준선을 항상 함께 낸다.** 없으면 숫자를 해석할 수 없다.
6. **원본 예측을 저장한다.** 짝지은 McNemar 와 사후 재계산이 여기서 나온다.

사용법.
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

from scg_hvd.metrics import (aggregate_to_patient, macro_metrics,  # noqa: E402
                             majority_baseline, metrics_table)
from scg_hvd.splits import (check_patient_disjoint, patient_cv_folds,  # noqa: E402
                            quantify_overlap_leakage)
from scg_hvd.datasets import WAVEFORM_MODELS  # noqa: E402
from scg_hvd.train import TrainConfig, run_training  # noqa: E402

DATA = Path("/work/jkim1/SCG_HVD_data")
IMAGE_DIRS = {"task1": DATA / "Task1_images", "task2": DATA / "Task2_images"}


def load_task(task):
    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    names = sorted(df.label.unique())
    df["label_name"] = df["label"]
    df["label"] = df["label"].map({n: i for i, n in enumerate(names)})
    df["filepath"] = df["filepath"].str.replace(
        "/hpc/dctrl/jk622/exp/2025_BHI/data/Data/", str(DATA) + "/", regex=False)
    return df, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1", choices=["task1", "task2"])
    ap.add_argument("--model", default="fusion", choices=["1d", "2d", "fusion", "resnet1d", "tcn",
                                                          "temporal_matched", "resnet1d_matched",
                                                          "tcn_matched",
                                                          "2d_independent", "fusion_independent"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/patient_cv"))
    ap.add_argument("--save-checkpoint", action="store_true",
                    help="fold 마다 best 가중치를 저장한다. 해석성 분석(R2-m7)에 쓴다.")
    a = ap.parse_args()

    df, class_names = load_task(a.task)
    n_classes = len(class_names)
    # 시드 하나만 돌릴 때는 출력 경로에 시드를 넣는다. 배열 잡으로 병렬 실행해도 덮어쓰지 않는다.
    root = a.out / a.task / a.model
    if len(a.seeds) == 1:
        root = root / f"seed{a.seeds[0]}"
    root.mkdir(parents=True, exist_ok=True)

    print(f"===== {a.task} / {a.model} | {a.folds}-fold x {len(a.seeds)} seed =====")
    print(f"  세그먼트 {len(df)}, 환자 {df.patient_id.nunique()}, 클래스 {class_names}\n", flush=True)

    seg_rows, pat_rows, fold_summ = [], [], []

    for seed in a.seeds:
        for k, split_df in patient_cv_folds(df, n_splits=a.folds, random_state=seed):
            tag = f"seed{seed}_fold{k}"
            v = check_patient_disjoint(split_df)["violations"]
            lk = quantify_overlap_leakage(split_df, against=("train", "val"))
            assert v == 0, f"{tag}: 환자가 두 split 에 걸쳤다 ({v}명)"
            assert lk["n_with_overlapping_neighbour"] == 0, f"{tag}: 겹침 누수가 있다"

            te = split_df[split_df.split == "test"]
            print(f"--- {tag}: test 환자 {te.patient_id.nunique()}명 / 세그먼트 {len(te)} ---",
                  flush=True)

            cfg = TrainConfig(model=a.model, num_classes=n_classes, epochs=a.epochs,
                              lr=1e-3, num_workers=a.num_workers, seed=seed,
                              class_weight_scope="train",   # 새 실험은 학습 분할만 사용
                              save_checkpoint=a.save_checkpoint)
            out_dir = root / tag
            _, (y_true, y_pred, y_prob), res = run_training(
                split_df, cfg,
                image_dir=None if a.model in WAVEFORM_MODELS else IMAGE_DIRS[a.task],
                out_dir=out_dir, class_names=class_names, verbose=False)

            # 세그먼트 수준
            seg_tab = metrics_table(y_true, y_pred, class_names)
            seg_tab.to_csv(out_dir / "segment_metrics.csv", index=False, float_format="%.6f")
            seg_ov = seg_tab[seg_tab.class_name == "Overall"].iloc[0]
            seg_macro = macro_metrics(y_true, y_pred, class_names)

            # 환자 수준
            pred = pd.read_csv(out_dir / "predictions.csv")
            pt = aggregate_to_patient(pred, n_classes)
            pt.to_csv(out_dir / "patient_predictions.csv", index=False)
            pat_tab = metrics_table(pt.y_true.values, pt.y_pred.values, class_names)
            pat_tab.to_csv(out_dir / "patient_metrics.csv", index=False, float_format="%.6f")
            pat_acc = float((pt.y_pred == pt.y_true).mean())
            pat_macro = macro_metrics(pt.y_true.values, pt.y_pred.values, class_names)
            base = majority_baseline(pt.y_true.values, class_names)

            print(f"    세그먼트 acc {float((y_pred==y_true).mean()):.4f} macro-F1 {seg_macro['macro_f1']:.4f}"
                  f" | 환자 acc {pat_acc:.4f} ({int((pt.y_pred==pt.y_true).sum())}/{len(pt)})"
                  f" macro-F1 {pat_macro['macro_f1']:.4f} | 기준선 {base['accuracy_plain']:.4f}",
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
                "majority_accuracy": base["accuracy_plain"],
                "best_val_acc": res["best_val_acc"], "elapsed_sec": res["elapsed_sec"],
            })

    pd.concat(seg_rows).to_csv(root / "all_segment_predictions.csv", index=False)
    pd.concat(pat_rows).to_csv(root / "all_patient_predictions.csv", index=False)
    fs = pd.DataFrame(fold_summ)
    fs.to_csv(root / "fold_summary.csv", index=False, float_format="%.6f")

    print(f"\n===== {a.task} / {a.model} 요약 ({len(fs)} fold) =====")
    for col in ["segment_accuracy", "segment_macro_f1", "patient_accuracy",
                "patient_macro_f1", "majority_accuracy"]:
        m, s = fs[col].mean(), fs[col].std()
        lo, hi = m - 1.96 * s / np.sqrt(len(fs)), m + 1.96 * s / np.sqrt(len(fs))
        print(f"  {col:20s} {m:.4f} ± {s:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")

    # 전체 fold 를 모은 환자 단위 결과 (pooled)
    allp = pd.concat(pat_rows)
    pooled = metrics_table(allp.y_true.values, allp.y_pred.values, class_names)
    pooled.to_csv(root / "pooled_patient_metrics.csv", index=False, float_format="%.6f")
    print(f"\n  pooled 환자 단위 ({len(allp)} 예측)")
    print(pooled[["class_name", "sensitivity", "specificity", "accuracy", "f1_score", "support"]]
          .to_string(index=False, float_format=lambda v: f"{v*100:.2f}"))

    (root / "config.json").write_text(json.dumps(
        {"task": a.task, "model": a.model, "folds": a.folds, "seeds": a.seeds,
         "epochs": a.epochs, "class_names": class_names}, indent=2))
    print(f"\n결과: {root}")


if __name__ == "__main__":
    main()
