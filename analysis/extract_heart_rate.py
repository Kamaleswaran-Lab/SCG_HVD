# R2-M2 1순위 대응. 세그먼트 파일 안의 ECG에서 심박수를 뽑는다.
"""
Referee 2 major 2 가 가장 강조한 공변량이 심박수다.

    "I think the results need to be adjusted first especially for heart rate. Any change in
     heart rate can significantly change the time-freq results and domains."

심박수는 코호트 메타데이터에 없지만 **세그먼트 .npy 안에 ECG 가 들어 있다.** 원본 CSV 헤더로
채널 구성을 확정했다(time 열을 뺀 뒤 인덱스).

    Dataset I  (12채널)  0-2 SCG_x/y/z | 3-4 EMG_status(상수, 패딩) |
                         5-8 ECG_LA-RA / LL-LA / LL-RA / Vx-RL | 9-11 GCG_x/y/z
    Dataset II ( 7채널)  0 EKG | 1-3 SCG_x/y/z | 4-6 GCG_x/y/z

학습 코드는 각각 `x[:, 0:3]`, `x[:, 1:4]` 로 SCG 만 쓴다. ECG 는 학습에 들어가지 않으므로
여기서 쓰는 것은 누수가 아니라 공변량 산출이다.

검출은 Pan-Tompkins 계열이다. 5-15 Hz 대역통과 -> 미분 -> 제곱 -> 150 ms 이동적분 -> 불응기를
둔 피크 검출. 세그먼트가 10초뿐이라 박동 수가 적으므로 품질 플래그를 함께 남기고,
집계할 때 품질이 나쁜 세그먼트를 제외할 수 있게 한다.

원고 정정 사항. `Manuscript_BPEX.tex:142` 는 Dataset I 을 "single-lead ECG" 라고 쓰지만
실제로는 4유도다.

사용법.
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

DATA = Path("/work/jkim1/SCG_HVD_data")
FS = 256

#: 채널 수 -> ECG 채널 인덱스(우선순위 순)
ECG_CHANNELS = {12: [5, 6, 7, 8], 7: [0]}


def ecg_from_segment(x: np.ndarray) -> np.ndarray | None:
    """세그먼트 배열에서 가장 쓸 만한 ECG 유도를 고른다. 진폭이 죽은 유도는 건너뛴다."""
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
    """Pan-Tompkins 계열 R-peak 검출. (피크 인덱스, 적분 신호) 를 돌려준다."""
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
        distance=int(0.25 * fs),      # 최대 240 bpm
    )
    return peaks, integ


def segment_hr(x: np.ndarray, fs=FS) -> dict:
    """세그먼트 하나의 심박수와 품질 지표."""
    ecg = ecg_from_segment(x)
    if ecg is None:
        return {"hr_bpm": np.nan, "n_beats": 0, "rr_cv": np.nan, "quality": "no_ecg"}

    peaks, _ = detect_r_peaks(ecg, fs)
    if len(peaks) < 3:
        return {"hr_bpm": np.nan, "n_beats": len(peaks), "rr_cv": np.nan,
                "quality": "too_few_beats"}

    rr = np.diff(peaks) / fs
    rr = rr[(rr > 0.25) & (rr < 2.0)]          # 30-240 bpm 밖은 검출 오류로 본다
    if len(rr) < 2:
        return {"hr_bpm": np.nan, "n_beats": len(peaks), "rr_cv": np.nan,
                "quality": "rr_out_of_range"}

    hr = 60.0 / float(np.median(rr))
    cv = float(np.std(rr) / np.mean(rr))
    q = "ok" if (40 <= hr <= 140 and cv < 0.35) else "suspect"
    return {"hr_bpm": hr, "n_beats": len(peaks), "rr_cv": cv, "quality": q}


def run(task: str, out_dir: Path, limit=None):
    meta = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    meta["filepath"] = meta["filepath"].str.replace(
        "/hpc/dctrl/jk622/exp/2025_BHI/data/Data/", str(DATA) + "/", regex=False)
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

    print(f"\n=== {task}: 세그먼트 {len(seg)} ===")
    print(seg.quality.value_counts().to_string())

    ok = seg[seg.quality == "ok"]
    pat = (ok.groupby(["patient_id", "label"])
             .agg(hr_median=("hr_bpm", "median"), hr_sd=("hr_bpm", "std"),
                  n_ok=("hr_bpm", "size"))
             .reset_index())
    pat.to_csv(out_dir / f"{task}_patient_hr.csv", index=False, float_format="%.4f")

    print(f"\n=== 클래스별 심박수 (환자 단위 중앙값, 품질 ok 세그먼트만) ===")
    summ = (pat.groupby("label")
              .agg(n_patients=("hr_median", "size"), hr_mean=("hr_median", "mean"),
                   hr_sd=("hr_median", "std"), hr_min=("hr_median", "min"),
                   hr_max=("hr_median", "max")).round(1))
    print(summ.to_string())

    from scipy import stats
    groups = [g.hr_median.values for _, g in pat.groupby("label") if len(g) >= 2]
    if len(groups) >= 2:
        s, p = stats.kruskal(*groups)
        print(f"\nKruskal-Wallis (클래스 간 심박수 차이): H={s:.3f}, p={p:.3e}"
              f"  -> {'유의함' if p < 0.05 else '유의하지 않음'}")
    summ.to_csv(out_dir / f"{task}_hr_by_class.csv")
    return seg, pat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1", choices=["task1", "task2"])
    ap.add_argument("--out", type=Path, default=Path("out/hr"))
    ap.add_argument("--limit", type=int, default=None, help="시험 실행용 세그먼트 수 제한")
    a = ap.parse_args()
    run(a.task, a.out, a.limit)


if __name__ == "__main__":
    main()
