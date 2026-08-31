# Derives heart rate from the ECG that is already sitting inside the segment files.
"""
Heart rate is the covariate Referee 2 pressed hardest on.

    "I think the results need to be adjusted first especially for heart rate. Any change in
     heart rate can significantly change the time-freq results and domains."

It is absent from the cohort metadata, but the segment `.npy` files carry ECG alongside the
accelerometer channels. The layout was confirmed against the original CSV headers, indexed
after dropping the time column:

    Dataset I  (12 ch)  0-2 SCG_x/y/z | 3-4 EMG_status (constant, padding) |
                        5-8 ECG_LA-RA / LL-LA / LL-RA / Vx-RL | 9-11 GCG_x/y/z
    Dataset II ( 7 ch)  0 EKG | 1-3 SCG_x/y/z | 4-6 GCG_x/y/z

The training code reads `x[:, 0:3]` and `x[:, 1:4]` respectively, so the ECG never reaches a
model. Using it here computes a covariate; it does not leak anything into training.

Detection is Pan-Tompkins: 5-15 Hz bandpass, differentiate, square, 150 ms moving integration,
then peak-picking with a refractory period. A 10 s segment holds few beats, so each estimate
carries a quality flag and the aggregation step can drop the poor ones rather than averaging
them in.

Incidentally, this is how we found that the manuscript called Dataset I 'single-lead ECG' when
it has four derivations.

Usage.
    python analysis/extract_heart_rate.py --task task1 --out out/hr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.paths import data_root, localise  # noqa: E402

DATA = data_root(required=False)
FS = 256

#: channel count -> candidate ECG columns, best first
ECG_CHANNELS = {12: [5, 6, 7, 8], 7: [0]}


def ecg_from_segment(x: np.ndarray) -> np.ndarray | None:
    """Pick the most usable ECG derivation, skipping any whose amplitude has flatlined."""
    idxs = ECG_CHANNELS.get(x.shape[1])
    if idxs is None:
        return None
    best, best_std = None, 0.0
    for i in idxs:
        v = x[:, i].astype(np.float64)
        s = float(np.std(v))
        if s > best_std:
            best, best_std = v, s
    return best if best_std > 1e-6 else None


def detect_r_peaks(ecg: np.ndarray, fs=FS):
    """Pan-Tompkins R-peak detection. Returns (peak indices, integrated signal)."""
    x = ecg - np.mean(ecg)
    nyq = fs / 2
    b, a = sps.butter(3, [5 / nyq, 15 / nyq], btype="band")
    f = sps.filtfilt(b, a, x)
    d = np.diff(f, prepend=f[0])
    sq = d ** 2
    win = max(1, int(0.150 * fs))
    integ = np.convolve(sq, np.ones(win) / win, mode="same")

    thr = np.mean(integ) + 0.5 * np.std(integ)
    peaks, _ = sps.find_peaks(
        integ,
        height=max(thr, 1e-12),
        distance=int(0.25 * fs),      # refractory period, i.e. at most 240 bpm
    )
    return peaks, integ


def segment_hr(x: np.ndarray, fs=FS) -> dict:
    """Heart rate and quality indicators for one segment."""
    ecg = ecg_from_segment(x)
    if ecg is None:
        return {"hr_bpm": np.nan, "n_beats": 0, "rr_cv": np.nan, "quality": "no_ecg"}

    peaks, _ = detect_r_peaks(ecg, fs)
    if len(peaks) < 3:
        return {"hr_bpm": np.nan, "n_beats": len(peaks), "rr_cv": np.nan,
                "quality": "too_few_beats"}

    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.25) & (rr < 2.0)]          # outside 30-240 bpm, treat as a detection error
    if len(rr) < 2:
        return {"hr_bpm": np.nan, "n_beats": len(peaks), "rr_cv": np.nan,
                "quality": "rr_out_of_range"}

    hr = 60.0 / float(np.median(rr))
    cv = float(np.std(rr) / np.mean(rr))
    q = "ok" if (40 <= hr <= 140 and cv < 0.35) else "suspect"
    return {"hr_bpm": hr, "n_beats": len(peaks), "rr_cv": cv, "quality": q}


def run(task: str, out_dir: Path, limit=None):
    meta = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    meta["filepath"] = localise(meta["filepath"], DATA)
    if limit:
        meta = meta.head(limit)

    rows = []
    for n, (_, r) in enumerate(meta.iterrows(), 1):
        x = np.load(r.filepath)
        res = segment_hr(x)
        rows.append({"segment_id": r.segment_id, "patient_id": r.patient_id,
                     "label": r.label, "n_channels": x.shape[1], **res})
        if n % 1000 == 0:
            print(f"  {n}/{len(meta)}", flush=True)

    seg = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    seg.to_csv(out_dir / f"{task}_segment_hr.csv", index=False, float_format="%.4f")

    print(f"\n=== {task}: {len(seg)} segments ===")
    print(seg.quality.value_counts().to_string())

    ok = seg[seg.quality == "ok"]
    pat = (ok.groupby(["patient_id", "label"])
             .agg(hr_median=("hr_bpm", "median"), hr_sd=("hr_bpm", "std"),
                  n_ok=("hr_bpm", "size"))
             .reset_index())
    pat.to_csv(out_dir / f"{task}_patient_hr.csv", index=False, float_format="%.4f")

    print(f"\n=== heart rate by class (per-patient median, usable segments only) ===")
    summ = (pat.groupby("label")
              .agg(n_patients=("hr_median", "size"), hr_mean=("hr_median", "mean"),
                   hr_sd=("hr_median", "std"), hr_min=("hr_median", "min"),
                   hr_max=("hr_median", "max")).round(1))
    print(summ.to_string())

    from scipy import stats
    groups = [g.hr_median.values for _, g in pat.groupby("label") if len(g) >= 2]
    if len(groups) >= 2:
        s, p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis across classes: H={s:.3f}, p={p:.3e}"
              f"  -> {'significant' if p < 0.05 else 'not significant'}")
    summ.to_csv(out_dir / f"{task}_hr_by_class.csv")
    return seg, pat


def main():
    global DATA
    DATA = data_root()   # fail here rather than on a puzzling missing file
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1", choices=["task1", "task2"])
    ap.add_argument("--out", type=Path, default=Path("out/hr"))
    ap.add_argument("--limit", type=int, default=None, help="cap the segment count, for a trial run")
    a = ap.parse_args()
    run(a.task, a.out, a.limit)


if __name__ == "__main__":
    main()
