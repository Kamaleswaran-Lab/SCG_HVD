# 논문의 세그먼트 단위 6개 실행을 재현하고 원고 Table 2/3 과 셀 단위로 대조한다.
"""
이 스크립트가 재현하는 것은 **원고에 실린 숫자를 만든 실험**이며, 일반화 성능 추정이 아니다.
세그먼트 단위 분할이므로 Task I 테스트 세그먼트의 98.96% 가 학습·검증 세트에 50% 겹치는
이웃 창을 갖는다. 이 사실 자체를 함께 출력한다.

기대 결과. 감사에 따르면 원고 120셀 중 103셀은 정확히 일치하고 17셀은 소수 둘째 자리에서
어긋난다. 어긋남의 원인은 아카이브가 혼동행렬을 정규화해 저장한 뒤 되곱아 복원한 손실이며,
여기서는 원본 예측에서 직접 계산하므로 **새 값이 원고와 다른 것이 정상**이다.
다만 학습에는 난수가 개입하므로 셀이 정확히 재현되지는 않는다. 이 스크립트의 목적은
(a) 파이프라인이 같은 수준의 결과를 내는지, (b) 어느 셀이 원고와 다른지를 기록하는 것이다.

사용법.
    python scripts/run_paper_segment.py --task task1 --models 1d 2d fusion --out out/paper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.metrics import (majority_baseline, metrics_table,  # noqa: E402
                             overall_metrics_are_dependent)
from scg_hvd.splits import (check_patient_disjoint, quantify_overlap_leakage,  # noqa: E402
                            segment_split)
from scg_hvd.datasets import WAVEFORM_MODELS  # noqa: E402
from scg_hvd.train import TrainConfig, run_training  # noqa: E402

DATA = Path("/work/jkim1/SCG_HVD_data")
IMAGE_DIRS = {"task1": DATA / "Task1_images", "task2": DATA / "Task2_images"}
EPOCHS = {"1d": 25, "2d": 25, "fusion": 25}   # 정본 bash_script.sh 와 동일


def load_task(task):
    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    names = sorted(df.label.unique())
    label_map = {n: i for i, n in enumerate(names)}
    df["label_name"] = df["label"]
    df["label"] = df["label"].map(label_map)
    # 신호 경로를 로컬 사본으로 바꾼다.
    df["filepath"] = df["filepath"].str.replace(
        "/hpc/dctrl/jk622/exp/2025_BHI/data/Data/", str(DATA) + "/", regex=False)
    return df, names


def main():
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

    # --- 분할 진단을 먼저 기록한다 ---
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

    print(f"===== {a.task}: 세그먼트 단위 분할 진단 =====")
    print(f"  세그먼트 {len(df)}, 환자 {df.patient_id.nunique()}, 클래스 {class_names}")
    print(f"  test {diag['n_test']}개 | 두 split 에 걸친 환자 {diag['patients_split_across']}명")
    print(f"  겹침 누수 train {diag['overlap_vs_train']['fraction']:.4f}, "
          f"train+val {diag['overlap_vs_train_val']['fraction']:.4f}")
    print("  -> 이 분할은 논문 재현용이며 일반화 성능이 아니다.\n")

    summary = []
    for m in a.models:
        print(f"===== {a.task} / {m} =====", flush=True)
        cfg = TrainConfig(
            model=m, num_classes=n_classes,
            epochs=a.epochs or EPOCHS.get(m, 25),
            lr=1e-3, num_workers=a.num_workers, seed=a.seed,
            class_weight_scope="all",     # 논문 재현이므로 정본과 동일하게 전체 사용
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
    print("===== Overall 요약 (%) =====")
    print(sm.assign(**{c: (sm[c] * 100).round(2) for c in
                       ["sensitivity", "specificity", "accuracy", "f1_score"]})
          .to_string(index=False))

    base = majority_baseline(split_df[split_df.split == "test"].label.values, class_names)
    print(f"\n다수 클래스 기준선({base['majority_class']}), 테스트셋 기준")
    print(f"  실제 정확도        {base['accuracy_plain']*100:.2f}%")
    print(f"  one-vs-rest 척도   {base['accuracy_onevsrest']*100:.2f}%  <- 원고 Overall 행과 같은 정의")
    print(f"  macro-F1           {base['macro_f1']*100:.2f}%")

    # R2-M5: Overall 네 지표가 Se 로부터 결정됨을 같이 보인다.
    if len(sm):
        se = float(sm.iloc[-1].sensitivity)
        d = overall_metrics_are_dependent(n_classes, se)
        print(f"\nR2-M5 확인: Se {se*100:.2f} 이면 Sp {d['specificity_implied']*100:.2f}, "
              f"Ac {d['accuracy_implied']*100:.2f}, F1 {d['f1_implied']*100:.2f} 로 결정된다.")
    print(f"\n결과: {root}")


if __name__ == "__main__":
    main()
