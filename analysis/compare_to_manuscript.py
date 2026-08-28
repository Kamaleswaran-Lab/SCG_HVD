# 재현 결과를 원고 Table 2/3 과 120셀 전부 대조한다.
"""
목적 두 가지다.

1. 이관된 파이프라인이 원고와 같은 수준을 내는지 확인한다.
2. **리비전 표에 넣을 새 값을 만든다.**

두 번째가 중요하다. 아카이브는 혼동행렬을 `normalize='true'` 로 저장한 뒤 하드코딩된
support 로 되곱아 지표를 복원했고, 그 왕복 손실로 원고 120셀 중 17셀이 소수 둘째 자리에서
어긋난다. 여기서는 원본 예측에서 직접 계산하므로 **새 값이 원고와 다른 것이 정상**이다.

학습 난수 때문에 셀이 정확히 일치하지는 않는다. 따라서 "일치/불일치" 판정이 아니라
차이의 분포를 본다. 클래스별 차이가 몇 퍼센트포인트 안에 들어오는지, Overall 이 같은
서열(1D < 2D < fusion)을 유지하는지가 판단 기준이다.

사용법.
    python analysis/compare_to_manuscript.py --out out/paper
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# 원고 Manuscript_BPEX.tex Table 2 (:395-425) 및 Table 3 (:440-470) 에서 옮긴 값.
# (Se, Sp, Ac, F1) 퍼센트.
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
        print(f"[미완] {', '.join(missing)}\n")
    if not rows:
        print("재현 결과가 아직 없다.")
        return

    df = pd.DataFrame(rows)
    a.report.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.report, index=False)

    print(f"=== 대조 요약 ({len(df)}셀) ===")
    ad = df["diff"].abs()
    print(f"  |차이| 중앙값 {ad.median():.2f}pp | 평균 {ad.mean():.2f}pp | 최대 {ad.max():.2f}pp")
    for thr in (0.5, 1.0, 2.0, 5.0):
        print(f"  |차이| <= {thr:>4.1f}pp : {(ad <= thr).sum():3d} / {len(df)}  ({(ad<=thr).mean()*100:.0f}%)")

    print("\n=== Overall 행 대조 ===")
    ov = df[df["class"] == "Overall"].pivot_table(index=["task", "model"], columns="metric",
                                                  values=["paper", "repro"])
    print(ov.round(2).to_string())

    print("\n=== 서열 유지 확인 (Overall Se) ===")
    for task in PAPER:
        sub = df[(df.task == task) & (df["class"] == "Overall") & (df.metric == "Se")]
        if len(sub) < 3:
            print(f"  {task}: 미완")
            continue
        p = {r.model: r.paper for _, r in sub.iterrows()}
        r_ = {r.model: r.repro for _, r in sub.iterrows()}
        order_p = sorted(p, key=p.get)
        order_r = sorted(r_, key=r_.get)
        ok = "일치" if order_p == order_r else "불일치"
        print(f"  {task}: 원고 {' < '.join(order_p)} | 재현 {' < '.join(order_r)}  -> {ok}")

    print("\n=== 차이가 큰 셀 (|diff| > 3pp) ===")
    big = df[df["diff"].abs() > 3].sort_values("diff", key=abs, ascending=False)
    print(big.to_string(index=False) if len(big) else "  없음")

    print(f"\n전체 표: {a.report}")


if __name__ == "__main__":
    main()
