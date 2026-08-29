# R2-M6 코호트 표와 R2-m3 환자당 세그먼트 통계를 만든다.
"""
Referee 2 major 6 은 원 데이터셋 논문을 참조하라고 넘기지 말고 코호트 표를 직접 실으라고 했다.

    "The manuscript should include a cohort table by diagnostic class and dataset. The reference
     standard for each valve diagnosis, disease severity, presence of multiple lesions, and timing
     relative to intervention should be clearly stated."

Referee 2 minor 3 은 별도로 이것을 요구했다.

    "Report the total recording duration and number of segments per patient, including the range
     and median. It is unclear why patients contribute substantially different numbers of windows
     and whether this creates patient-level weighting imbalance."

두 요구가 같은 원천(`df_metadata.csv` + 세그먼트 메타데이터)에서 나오므로 한 스크립트로 낸다.

정직하게 밝혀야 하는 한계 두 가지.
  (a) Dataset II 정상군 29명은 `df_metadata.csv` 에 없다. 나이·성별·병력을 모른다.
  (b) 판막 지표는 결측이 많다(대동맥판 면적 37/100, 승모판 압력차 3/100).
표에 결측 수를 함께 적어 리뷰어가 스스로 판단할 수 있게 한다.

없는 항목도 명시한다. rhythm, sensor position, recording posture 는 어느 원천에도 없다.

사용법.
    python analysis/cohort_table.py --out out/cohort
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/work/jkim1/SCG_HVD_data")

#: 리뷰어가 이름을 댄 항목 -> 우리가 가진 열. None 이면 어느 원천에도 없다.
REQUESTED = {
    "age": "Age",
    "sex": "Gender",
    "body habitus": "Height (cm) / Weight (kg)",
    "valve-disease severity": "Aortic/Mitral valve area, mean gradient, peak velocity",
    "rhythm": None,
    "comorbidities": "History of MI / CABG / PCI",
    "prior interventions": "History of CABG / PCI",
    "sensor position": None,
    "recording posture": None,
    "recording duration": "Duration",
    "heart rate": "ECG 에서 산출 (analysis/extract_heart_rate.py)",
    "reference standard": "Echo available / Date of echo",
}


def build(task: str, out: Path):
    seg = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    md = pd.read_csv(DATA / "meta" / "df_metadata.csv")
    md["duration_sec"] = pd.to_timedelta(md["Duration"]).dt.total_seconds()

    hr_csv = Path("out/hr") / f"{task}_patient_hr.csv"
    hr = pd.read_csv(hr_csv) if hr_csv.exists() else None

    # 환자당 세그먼트 (R2-m3)
    per_pat = (seg.groupby(["patient_id", "label"]).size()
               .reset_index(name="n_segments"))
    per_pat["dataset"] = np.where(per_pat.patient_id.str.startswith("sub"),
                                  "Dataset II", "Dataset I")
    if hr is not None:
        per_pat = per_pat.merge(hr[["patient_id", "hr_median"]], on="patient_id", how="left")
    per_pat = per_pat.merge(
        md[["Patient ID", "Age", "Gender", "Height (cm)", "Weight (kg)",
            "Ejection fraction (%)", "duration_sec"]],
        left_on="patient_id", right_on="Patient ID", how="left").drop(columns="Patient ID")
    per_pat.to_csv(out / f"{task}_per_patient.csv", index=False)

    print(f"===== {task}: 환자당 세그먼트 (R2-m3) =====")
    t = (per_pat.groupby("label")
         .agg(n_patients=("n_segments", "size"), total_segments=("n_segments", "sum"),
              seg_median=("n_segments", "median"), seg_min=("n_segments", "min"),
              seg_max=("n_segments", "max")).astype(int))
    print(t.to_string())
    imb = per_pat.n_segments.max() / per_pat.n_segments.min()
    print(f"\n  전체 환자당 세그먼트: 중앙값 {per_pat.n_segments.median():.0f}, "
          f"범위 {per_pat.n_segments.min()}–{per_pat.n_segments.max()}, "
          f"최대/최소 비 {imb:.1f}배")
    print(f"  -> 환자 단위 가중 불균형이 존재한다. 세그먼트 단위 지표는 세그먼트가 많은 환자에")
    print(f"     더 큰 가중을 준다. 환자 단위 집계를 함께 보고해야 하는 이유다.")

    # 코호트 표 (R2-M6)
    print(f"\n===== {task}: 코호트 표 (R2-M6) =====")
    d1 = per_pat[per_pat.dataset == "Dataset I"]
    rows = []
    for lab, g in per_pat.groupby("label"):
        g1 = g[g.dataset == "Dataset I"]
        r = {"class": lab, "n_patients": len(g),
             "Dataset I": int((g.dataset == "Dataset I").sum()),
             "Dataset II": int((g.dataset == "Dataset II").sum()),
             "segments": int(g.n_segments.sum())}
        if len(g1):
            r["age_mean"] = round(g1.Age.mean(), 1)
            r["age_sd"] = round(g1.Age.std(), 1) if len(g1) > 1 else np.nan
            r["male_n"] = int((g1.Gender == "M").sum())
            r["age_missing"] = int(g1.Age.isna().sum()) + int((g.dataset == "Dataset II").sum())
            r["EF_mean"] = round(g1["Ejection fraction (%)"].mean(), 1)
            r["duration_median_s"] = round(g1.duration_sec.median(), 0)
        if hr is not None:
            r["hr_median"] = round(g.hr_median.median(), 1)
        rows.append(r)
    tab = pd.DataFrame(rows)
    tab.to_csv(out / f"{task}_cohort_table.csv", index=False)
    print(tab.to_string(index=False))

    if (per_pat.dataset == "Dataset II").any():
        n2 = int((per_pat.dataset == "Dataset II").sum())
        print(f"\n  [한계] Dataset II 환자 {n2}명은 코호트 메타데이터가 없어 나이·성별·병력이 결측이다.")
        print(f"         심박수는 신호에서 산출했으므로 이 {n2}명도 포함된다.")
    return per_pat, tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/cohort"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    for task in ("task1", "task2"):
        build(task, a.out)
        print()

    print("===== 리뷰어가 이름을 댄 항목별 보유 현황 (R2-M6) =====")
    for k, v in REQUESTED.items():
        print(f"  {'O' if v else 'X'}  {k:24s} {v or '어느 원천에도 없음 — 한계로 명시할 것'}")
    print(f"\n결과: {a.out}")


if __name__ == "__main__":
    main()
