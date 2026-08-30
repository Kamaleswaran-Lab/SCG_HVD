# R2-M1 보조 분석. 겹치는 창과 겹치지 않는 창의 환자 단위 결과를 나란히 낸다.
"""
Referee 2 major 1 이 요구한 두 분석 중 하나다.

    "A supplementary analysis using nonoverlapping windows and, where feasible, group-wise
     splitting for classes with sufficient patient counts would strengthen the work."

group-wise 분할은 `run_patient_cv.py` 가 제공하고, 이 스크립트는 그 위에서 겹침 유무만 바꾼
두 실행을 비교한다.

**해석에서 반드시 구분해야 하는 것.** 겹치지 않는 창만 쓰면 학습 세그먼트가 절반으로 준다
(Task I 8678 -> 4365). 따라서 성능이 떨어져도 그것이 곧 "겹침이 성능을 만들었다" 는 뜻은
아니다. 환자 단위 분할에서는 겹침이 train 과 test 를 넘나들지 않으므로 누수 경로가 없고,
남는 차이는 주로 표본 감소다. 두 해석을 구분해 적기 위해 세그먼트 수를 함께 보고한다.

사용법.
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
        print("결과 없음."); return
    t = pd.DataFrame(rows)
    t.to_csv(a.out / f"{a.task}_overlap_comparison.csv", index=False)

    print(f"===== {a.task}: 창 겹침 유무 비교 (환자 단위) =====")
    print(f"{'구성':22s} {'창':16s} {'seed':>4s} {'test seg':>9s} "
          f"{'환자 정확도 (95% CI)':>26s} {'macro-F1':>10s}")
    for _, r in t.iterrows():
        print(f"{LABEL[r.model]:22s} {r.windows:16s} {int(r.n_seeds):>4d} {r.test_segments:>9,} "
              f"{r.pat_acc:.4f} [{r.pat_acc_lo:.4f}, {r.pat_acc_hi:.4f}]  {r.pat_f1:>9.4f}")
    print(f"{'다수 클래스 기준선':22s} {'':16s} {'':>4s} {'':>9s} {t.majority.mean():.4f}")

    print(f"\n===== 겹침 제거의 영향 =====")
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
              f"CI {'겹침' if overlap_ci else '분리'} | "
              f"둘 다 기준선 위: {'예' if both_above else '아니오'} | "
              f"세그먼트 {ov.test_segments:,} -> {no.test_segments:,}")

    print("\n해석 주의. 겹치지 않는 창만 쓰면 학습 표본이 절반이 된다. 환자 단위 분할에서는")
    print("겹침이 train 과 test 를 넘나들지 않으므로 누수 경로가 없고, 남는 차이는 주로")
    print("표본 감소를 반영한다. 두 CI 가 겹치면 그 구분을 굳이 주장할 필요도 없다.")

    # 짝지은 비교는 같은 환자 집합에서만 의미가 있으므로 fold 구성이 같은 경우에 한한다.
    print(f"\n===== 같은 (seed, fold, patient) 에서의 짝지은 비교 =====")
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
            print(f"  {LABEL[m]}: 공통 예측 {len(common)}개 — 건너뜀"); continue
        ca, cb = da.loc[common], db.loc[common]
        r = mcnemar_paired(ca.y_true.values, ca.y_pred.values, cb.y_pred.values)
        flag = "  <== 유의" if r["p_value"] < 0.05 else ""
        print(f"  {LABEL[m]:22s} overlapping {r['accuracy_a']:.4f} vs non-overlapping "
              f"{r['accuracy_b']:.4f} | n={len(common)} | p={r['p_value']:.4f}{flag}")

    print(f"\n결과: {a.out / (a.task + '_overlap_comparison.csv')}")


if __name__ == "__main__":
    main()
