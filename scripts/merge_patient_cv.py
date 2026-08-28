# 시드별로 나눠 돌린 환자 단위 CV 결과를 하나로 합치고 구성 간 비교까지 낸다.
"""
`run_patient_cv.py` 를 시드마다 별도 배열 잡으로 돌리면 결과가
`out/patient_cv/{task}/{model}/seed{N}/` 로 흩어진다. 이 스크립트가 그걸 모아
fold 간 신뢰구간과 구성 간 짝지은 검정을 낸다.

리뷰어 요구와의 대응.
    R1-m3 / R2-M5  fold 간 분산 기반 95% CI. 세그먼트를 독립으로 보지 않는다.
    R2-M5          구성 간 비교는 환자 단위 짝지은 McNemar 로 한다.
    R1-m4 / R2-M3  환자 단위 집계가 주 결과다.

사용법.
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

MODELS = ["1d", "2d", "fusion"]


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
        print(f"[미완] {', '.join(missing)}\n")
    if not summ:
        print("결과 없음.")
        return

    print(f"===== {a.task}: 환자 단위 교차검증 =====")
    cols = ["segment_accuracy", "segment_macro_f1", "patient_accuracy",
            "patient_macro_f1", "majority_accuracy"]
    rows = []
    for m, f in summ.items():
        r = {"model": m, "n_runs": len(f)}
        for c in cols:
            mu, sd, lo, hi = ci(f[c])
            r[c] = f"{mu:.4f} ± {sd:.4f}"
            r[c + "_ci"] = f"[{lo:.4f}, {hi:.4f}]"
        rows.append(r)
    tab = pd.DataFrame(rows)
    print(tab[["model", "n_runs", "patient_accuracy", "patient_accuracy_ci",
               "patient_macro_f1", "patient_macro_f1_ci"]].to_string(index=False))
    print()
    print(tab[["model", "segment_accuracy", "segment_macro_f1",
               "majority_accuracy"]].to_string(index=False))
    tab.to_csv(a.out / f"{a.task}_summary.csv", index=False)

    # 구성 간 짝지은 비교 (환자 단위)
    print("\n=== 구성 간 짝지은 McNemar (환자 단위, 시드·fold 통합) ===")
    avail = [m for m in MODELS if preds.get(m) is not None]
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            A, B = avail[i], avail[j]
            a_df = preds[A].sort_values(["seed", "fold", "patient_id"]).reset_index(drop=True)
            b_df = preds[B].sort_values(["seed", "fold", "patient_id"]).reset_index(drop=True)
            if len(a_df) != len(b_df) or not (a_df.patient_id.values == b_df.patient_id.values).all():
                print(f"  {A} vs {B}: 예측 짝이 맞지 않아 건너뛴다")
                continue
            r = mcnemar_paired(a_df.y_true.values, a_df.y_pred.values, b_df.y_pred.values)
            print(f"  {A:7s} vs {B:7s}  acc {r['accuracy_a']:.4f} vs {r['accuracy_b']:.4f} | "
                  f"불일치 {r['n_discordant']:3d} ({r['n_a_only_correct']}/{r['n_b_only_correct']}) | "
                  f"p={r['p_value']:.4f}" + ("  <== 유의" if r["p_value"] < 0.05 else ""))

    # pooled 환자 단위 클래스별 표
    for m in avail:
        p = preds[m]
        names = sorted(pd.read_json(a.out / a.task / m / "seed0" / "config.json",
                                    typ="series")["class_names"]) if False else None
        print(f"\n--- {m}: pooled 환자 단위 ({len(p)} 예측) ---")
        n_cls = int(max(p.y_true.max(), p.y_pred.max())) + 1
        t = metrics_table(p.y_true.values, p.y_pred.values, [str(i) for i in range(n_cls)])
        print(t[["class_name", "sensitivity", "specificity", "f1_score", "support"]]
              .to_string(index=False, float_format=lambda v: f"{v*100:.2f}"))
        b = majority_baseline(p.y_true.values, [str(i) for i in range(n_cls)])
        print(f"    다수 클래스 기준선 정확도 {b['accuracy_plain']:.4f} | macro-F1 {b['macro_f1']:.4f}")

    print(f"\n결과: {a.out / (a.task + '_summary.csv')}")


if __name__ == "__main__":
    main()
