# Asks the signal whether the two datasets' axis labels point at the same physical axes.
"""
Referee 2 minor 2.

    "Please clarify whether the three accelerometer axes were consistently aligned across subjects
     and datasets. Variation in sensor orientation could substantially affect axis-specific SCG
     morphology."

Both datasets name their channels SCG_x / SCG_y / SCG_z, but they were recorded in different
studies with different accelerometers, and a shared label is no guarantee of a shared physical
axis. The source documentation does not settle it, so we check what the signal can tell us and
stop there.

What is checkable. Chest-wall SCG has different vibration characteristics along each axis; the
dorsoventral axis usually carries the largest amplitude. So if two datasets agree on

  (a) the relative ordering of per-axis variance,
  (b) the distribution of band-limited energy per axis, and
  (c) the correlation structure between axes,

the labeling is plausibly consistent, and if the ordering is swapped that points at an axis
permutation.

What is not checkable. This is a necessary condition, not a sufficient one. Similar statistics
are compatible with a rotation, and per-subject variation in sensor placement angle is not
separable in either dataset regardless. The strongest conclusion available is 'no evidence of
a permutation' or 'a permutation is suspected'. It never licenses 'the axes are aligned', and
the manuscript states the limitation alongside the result.

Usage.
    python analysis/axis_alignment.py --out out/axis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.paths import data_root, localize  # noqa: E402

from scg_hvd.channels import select_scg_channels  # noqa: E402

DATA = data_root(required=False)
FS = 256
AXES = ["x", "y", "z"]


def band_energy(sig, fs=FS, bands=((1, 5), (5, 15), (15, 30))):
    f, p = sps.welch(sig, fs=fs, nperseg=min(512, len(sig)))
    tot = np.trapz(p, f) + 1e-12
    return [float(np.trapz(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) / tot)
            for lo, hi in bands]


def per_segment_features(x):
    """Per-axis features from a (T,3) SCG array.

    Absolute amplitude scale differs between devices, so everything here is a ratio.
    """
    out = {}
    v = x.var(axis=0)
    frac = v / (v.sum() + 1e-12)
    for i, ax in enumerate(AXES):
        out[f"var_frac_{ax}"] = float(frac[i])
        b = band_energy(x[:, i])
        for (lo, hi), val in zip(((1, 5), (5, 15), (15, 30)), b):
            out[f"band{lo}_{hi}_{ax}"] = val
    c = np.corrcoef(x.T)
    out["corr_xy"], out["corr_xz"], out["corr_yz"] = float(c[0, 1]), float(c[0, 2]), float(c[1, 2])
    out["dominant_axis"] = AXES[int(np.argmax(v))]
    return out


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--per-patient", type=int, default=10, help="segments sampled per patient")
    ap.add_argument("--out", type=Path, default=Path("out/axis"))
    a = ap.parse_args()
    DATA = data_root()   # fail here rather than on a puzzling missing file
    a.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{a.task}.csv")
    df["filepath"] = localize(df.filepath, DATA)
    df["dataset"] = np.where(df.patient_id.str.startswith("sub"), "Dataset II", "Dataset I")

    rows = []
    for pid, g in df.groupby("patient_id"):
        sub = g.sample(min(a.per_patient, len(g)), random_state=42)
        for _, r in sub.iterrows():
            x = select_scg_channels(np.load(r.filepath).astype(np.float64))
            rows.append({"patient_id": pid, "dataset": r.dataset, "label": r.label,
                         **per_segment_features(x)})
    d = pd.DataFrame(rows)
    d.to_csv(a.out / f"{a.task}_axis_features.csv", index=False)

    print(f"=== {a.task}: per-axis variance share, averaged within each dataset ===")
    vf = d.groupby("dataset")[[f"var_frac_{ax}" for ax in AXES]].mean().round(3)
    print(vf.to_string())
    print("\n  Matching orderings across datasets suggest the labeling is consistent.")
    for ds, row in vf.iterrows():
        order = [AXES[i] for i in np.argsort(-row.values)]
        print(f"    {ds}: {' > '.join(order)}")

    print(f"\n=== which axis dominates ===")
    print(pd.crosstab(d.dataset, d.dominant_axis, normalize="index").round(3).to_string())

    print(f"\n=== band-limited energy share, averaged within each dataset ===")
    cols = [f"band{lo}_{hi}_{ax}" for ax in AXES for lo, hi in ((1, 5), (5, 15), (15, 30))]
    print(d.groupby("dataset")[cols].mean().round(3).T.to_string())

    print(f"\n=== between-axis correlation ===")
    print(d.groupby("dataset")[["corr_xy", "corr_xz", "corr_yz"]].mean().round(3).to_string())

    # verdict
    if d.dataset.nunique() < 2:
        print("\nNo verdict: only one dataset is present.")
        return
    o = {ds: [AXES[i] for i in np.argsort(-vf.loc[ds].values)] for ds in vf.index}
    same = len(set(tuple(v) for v in o.values())) == 1
    print("\n" + "=" * 66)
    if same:
        print("Verdict: the variance ordering agrees across datasets. No evidence of a\n"
              "         permutation.")
    else:
        print("Verdict: the variance ordering differs between datasets. A permutation or an\n"
              "         orientation difference is suspected.")
    print("Note: this is a necessary condition only. A matching ordering is still compatible")
    print("      with a rotation, and per-subject placement angle is not separable here.")
    print(f"\noutput: {a.out}")


if __name__ == "__main__":
    main()
