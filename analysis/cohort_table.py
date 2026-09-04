# Builds the cohort table and the per-patient segment statistics.
"""
Referee 2's sixth major comment says that pointing readers at the source dataset papers is not
adequate and the manuscript needs its own cohort table:

    "The manuscript should include a cohort table by diagnostic class and dataset. The reference
     standard for each valve diagnosis, disease severity, presence of multiple lesions, and timing
     relative to intervention should be clearly stated."

Their third minor comment asks for something the same sources can answer:

    "Report the total recording duration and number of segments per patient, including the range
     and median. It is unclear why patients contribute substantially different numbers of windows
     and whether this creates patient-level weighting imbalance."

Both draw on `df_metadata.csv` and the segment metadata, so one script produces both.

Two limitations that the table has to state rather than hide.
  (a) The 29 Dataset II controls are absent from `df_metadata.csv`. Their age, sex and history
      are unknown.
  (b) The valve measurements are sparse: aortic valve area for 37 of 100 patients, mitral mean
      gradient for 3.
Missing counts go into the table itself so a reader can judge for themselves.

What we do not have is listed too. Cardiac rhythm, sensor position and recording posture
appear in neither source.

Usage.
    python analysis/cohort_table.py --out out/cohort
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.paths import data_root  # noqa: E402

DATA = data_root(required=False)

#: Each item the reviewer named -> the column we hold for it. None means no source has it.
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
    "heart rate": "derived from ECG (analysis/extract_heart_rate.py)",
    "reference standard": "Echo available / Date of echo",
}


def build(task: str, out: Path):
    seg = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    md = pd.read_csv(DATA / "meta" / "df_metadata.csv")
    md["duration_sec"] = pd.to_timedelta(md["Duration"]).dt.total_seconds()

    hr_csv = Path("out/hr") / f"{task}_patient_hr.csv"
    hr = pd.read_csv(hr_csv) if hr_csv.exists() else None

    # segments per patient
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

    print(f"===== {task}: segments per patient =====")
    t = (per_pat.groupby("label")
         .agg(n_patients=("n_segments", "size"), total_segments=("n_segments", "sum"),
              seg_median=("n_segments", "median"), seg_min=("n_segments", "min"),
              seg_max=("n_segments", "max")).astype(int))
    print(t.to_string())
    imb = per_pat.n_segments.max() / per_pat.n_segments.min()
    print(f"\n  overall: median {per_pat.n_segments.median():.0f}, "
          f"range {per_pat.n_segments.min()}-{per_pat.n_segments.max()}, "
          f"max/min ratio {imb:.1f}x")
    print(f"  -> patients are weighted unequally. A segment-level metric counts a patient in")
    print(f"     proportion to how many windows they contributed, which is why the")
    print(f"     patient-level aggregate is reported alongside.")

    # cohort table
    print(f"\n===== {task}: cohort table =====")
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
        print(f"\n  [limitation] {n2} Dataset II participants carry no cohort metadata, so age,")
        print(f"               sex and history are missing for them. Heart rate is derived from")
        print(f"               the signal, so those {n2} are included in that column.")
    return per_pat, tab


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/cohort"))
    a = ap.parse_args()
    DATA = data_root()   # fail here rather than on a puzzling missing file
    a.out.mkdir(parents=True, exist_ok=True)

    for task in ("task1", "task2"):
        build(task, a.out)
        print()

    print("===== what we hold for each item the reviewer named =====")
    for k, v in REQUESTED.items():
        print(f"  {'yes' if v else ' no'}  {k:24s} "
              f"{v or 'absent from both sources -- state as a limitation'}")
    print(f"\noutput: {a.out}")


if __name__ == "__main__":
    main()
