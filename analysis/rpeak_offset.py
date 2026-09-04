# Measures how far our detector's R estimate sits from the actual QRS peak, per dataset.
"""Gate for the between-class timing comparison in the attention analysis.

The attention curves are aligned to R-peaks found by a Pan-Tompkins style detector. That
detector peaks on a squared-derivative envelope, which sits near the QRS centroid rather than
exactly on R, and the size of that displacement depends on QRS morphology. The two source
datasets do not supply the same ECG derivation -- Dataset I a limb lead, Dataset II a single
channel of its own -- so the displacement can differ between them systematically.

This matters because 29 of the 33 healthy participants come from Dataset II. A per-dataset
offset would move that class along the time axis relative to every other class, which is
exactly where the interesting observation sat.

The measurement: for each detected peak, find the largest absolute deflection of the
bandpass-filtered ECG within a short window around it, and record the signed difference. The
detector's error is then the distribution of those differences, and what we care about is
whether its center differs between datasets.

Decision rule, fixed before running (see BPEX_R1/interpretability-plan.md): if the median
offsets of the two datasets differ by less than 10 ms, the between-class timing comparison is
restorable. Otherwise it stays withdrawn.

Usage.
    python analysis/rpeak_offset.py --task task1 --out out/rpeak
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.extract_heart_rate import ecg_from_segment, detect_r_peaks  # noqa: E402
from scg_hvd.paths import data_root, localize  # noqa: E402

FS = 256
DATA = data_root(required=False)


def bandpassed(ecg: np.ndarray, fs=FS) -> np.ndarray:
    """The same 5-15 Hz band the detector works in, zero-phase so no delay is introduced."""
    nyq = fs / 2
    b, a = sps.butter(3, [5 / nyq, 15 / nyq], btype="band")
    return sps.filtfilt(b, a, ecg - np.mean(ecg))


def offsets(ecg: np.ndarray, peaks, search_ms=60, fs=FS):
    """Signed sample offsets from each detected peak to the nearest dominant QRS deflection.

    Positive means the detector fired late. The search window is deliberately narrow: widening
    it past half a QRS complex would start matching neighbouring waves.
    """
    f = bandpassed(ecg, fs)
    half = int(search_ms / 1000 * fs)
    out = []
    for p in peaks:
        lo, hi = max(0, p - half), min(len(f), p + half + 1)
        if hi - lo < 3:
            continue
        out.append(int(np.argmax(np.abs(f[lo:hi])) + lo - p))
    return np.array(out, dtype=float) / fs * 1000.0   # milliseconds


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--per-patient", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("out/rpeak"))
    a = ap.parse_args()
    DATA = data_root()
    a.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{a.task}.csv")
    df["filepath"] = localize(df.filepath, DATA)
    # sub* identifiers are the Dataset II participants; everything else is Dataset I.
    df["dataset"] = np.where(df.patient_id.str.startswith("sub"), "Dataset II", "Dataset I")

    rows = []
    for pid, g in df.groupby("patient_id"):
        for _, r in g.sample(min(a.per_patient, len(g)), random_state=0).iterrows():
            try:
                raw = np.load(r.filepath)
            except Exception:
                continue
            ecg = ecg_from_segment(raw)
            if ecg is None:
                continue
            pk, _ = detect_r_peaks(ecg)
            if len(pk) < 2:
                continue
            for d in offsets(ecg, pk):
                rows.append({"patient_id": pid, "dataset": r.dataset,
                             "label": r.label, "offset_ms": d})
    if not rows:
        print("no usable segments."); return
    t = pd.DataFrame(rows)
    t.to_csv(a.out / f"{a.task}_rpeak_offsets.csv", index=False)

    print(f"=== {a.task}: detector offset from the dominant QRS deflection ===")
    print("Positive means the detector fired late.\n")
    summ = t.groupby("dataset").offset_ms.agg(["count", "median", "mean", "std"]).round(2)
    print(summ.to_string())

    meds = summ["median"].to_dict()
    verdict = {"medians_ms": meds}
    if len(meds) == 2:
        (a_name, a_med), (b_name, b_med) = meds.items()
        gap = abs(a_med - b_med)
        ok = gap < 10.0
        verdict.update({"gap_ms": round(gap, 2), "threshold_ms": 10.0, "restorable": bool(ok)})
        print(f"\nbetween-dataset gap: {gap:.2f} ms")
        print("verdict:", "under the 10 ms threshold; the between-class timing comparison is "
                          "restorable." if ok else
                          "at or over the 10 ms threshold; the comparison stays withdrawn.")
    print("\nby class (median ms)")
    print(t.groupby("label").offset_ms.median().round(2).to_string())
    (a.out / f"{a.task}_rpeak_offset_verdict.json").write_text(json.dumps(verdict, indent=2))
    print(f"\noutput: {a.out}")


if __name__ == "__main__":
    main()
