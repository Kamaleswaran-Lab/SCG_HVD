# 건강군의 늦은 attention 피크가 생리 신호인지 Dataset II 장비 특성인지 가른다.
"""Splits the healthy class by source dataset and compares the beat-aligned attention.

Section 4.5 reports that the healthy class carries a second attention peak near 0.4 s while the
lesion classes return to baseline. Twenty-nine of the 33 healthy patients come from Dataset II,
recorded with a different accelerometer, and Section 5.5 names that as the strongest confound in
the cohort. So the finding has a mundane alternative reading: an artefact of the second device.

Dataset I contributes four healthy patients (CP- prefix) against Dataset II's twenty-nine (sub-).
Four is a small sample and the comparison cannot be conclusive either way, but if the late peak
appears in both subsets the device reading is much harder to sustain, and if it appears in only
one the finding should not be reported as a property of the healthy class.

Usage.
    python analysis/attention_dataset_check.py --ckpt-dir out/patient_cv/task1/1d --out out/interp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.interpretability import (AXIS_ORDER, FS, AttentionRecorder,  # noqa: E402
                                       beat_aligned, load_task, r_peaks)
from analysis.extract_heart_rate import ecg_from_segment  # noqa: E402
from scg_hvd.channels import select_scg_channels  # noqa: E402
from scg_hvd.models import build_model  # noqa: E402
from scg_hvd.paths import data_root  # noqa: E402

SOURCE = {"CP": "Dataset I", "UP": "Dataset I", "sub": "Dataset II"}


def source_of(patient_id: str) -> str:
    for pref, name in SOURCE.items():
        if patient_id.startswith(pref):
            return name
    return "unknown"


def main():
    data_root()
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--ckpt-dir", type=Path, default=Path("out/patient_cv/task1/1d"))
    ap.add_argument("--n-per-patient", type=int, default=12)
    ap.add_argument("--out", type=Path, default=Path("out/interp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df, class_names = load_task(a.task)
    healthy = class_names.index("N")
    sub = df[df.label == healthy].copy()
    sub["source"] = sub.patient_id.map(source_of)
    print("healthy patients by source:")
    print(sub.groupby("source").patient_id.nunique().to_string())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model("1d", len(class_names)).to(device).eval()
    # One checkpoint per seed, matching how the figure in Section 4.5 was produced.
    ckpts = []
    for d in sorted(a.ckpt_dir.glob("seed*")):
        found = sorted(d.rglob("model.pt"))
        if found:
            ckpts.append(found[0])
    if not ckpts:
        raise SystemExit(f"no model.pt under {a.ckpt_dir}")
    print(f"using {len(ckpts)} checkpoints, one per seed")

    t = np.arange(int(-0.2 * FS), int(0.6 * FS)) / FS
    rows = []
    for ck in ckpts:
        model.load_state_dict(torch.load(ck, map_location=device))
        rec = AttentionRecorder(model.trunk)
        per_source = {s: [] for s in sub.source.unique()}
        for pid, g in sub.groupby("patient_id"):
            take = g.sample(min(a.n_per_patient, len(g)), random_state=42)
            for _, r in take.iterrows():
                raw = np.load(r.filepath).astype(np.float32)
                ecg = ecg_from_segment(raw)
                if ecg is None:
                    continue
                pk = r_peaks(ecg)
                if len(pk) < 3:
                    continue
                x = torch.from_numpy(select_scg_channels(raw).T.copy()).unsqueeze(0).to(device)
                with torch.no_grad():
                    model.extract_features(x)
                al = beat_aligned(rec.weights["branch_z"][0].numpy(), pk)
                if al is not None:
                    per_source[r.source].append(al)
        rec.close()
        for s, lst in per_source.items():
            if not lst:
                continue
            m = np.mean(lst, axis=0)
            pre = m[t < 0].mean()
            late = m[(t > 0.3) & (t < 0.5)].max()
            rows.append({"ckpt": ck.parent.name, "source": s, "n_segments": len(lst),
                         "pre_R_mean": pre, "late_peak": late, "ratio": late / pre,
                         "late_peak_t": float(t[(t > 0.3) & (t < 0.5)][
                             np.argmax(m[(t > 0.3) & (t < 0.5)])])})
            for ti, v in zip(t, m):
                pass
    d = pd.DataFrame(rows)
    d.to_csv(a.out / f"{a.task}_healthy_attention_by_source.csv", index=False)
    print("\n=== late attention peak (z axis), healthy class split by source dataset ===")
    print(d.groupby("source")[["n_segments", "ratio", "late_peak_t"]]
          .agg(["mean", "min", "max"]).round(3).to_string())
    print(f"\nwrote {a.out / (a.task + '_healthy_attention_by_source.csv')}")


if __name__ == "__main__":
    main()
