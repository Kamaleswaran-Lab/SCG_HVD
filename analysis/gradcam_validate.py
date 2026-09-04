# Decides whether the Grad-CAM attribution says anything about the model, before we report it.
"""The first Grad-CAM run produced a striking class pattern that we could not defend, and it
was withdrawn from the manuscript. Three things were wrong with the measurement rather than
with the result: the map was seven rows standing for 128 wavelet scales, the bands were cut by
row index so their widths in frequency differed by two orders of magnitude, and there was no
null to compare the numbers against.

The band definition and the choice of layer are fixed elsewhere (`band_fractions`,
`pick_target_layer`). What remains is the null, and that is what this script supplies.

Two controls, both of which should produce a flat or arbitrary attribution if the pattern we
saw belongs to the model:

    random      the same architecture with untrained weights
    shuffled    the same architecture trained on permuted labels, if a checkpoint is supplied

Decision rule, fixed before running (BPEX_R1/interpretability-plan.md):

  1. the trained model's band ratios must fall outside the range the controls produce, and
  2. the direction of the class pattern must agree across map resolutions.

Failing either, the attribution is a property of the rendering or of the attribution method
and we do not report it. That conclusion is worth as much as a positive one and is cheaper to
defend.

Usage.
    python analysis/gradcam_validate.py --ckpt out/interp_train/task1/fusion/seed0/model.pt \\
        --shuffled-ckpt out/interp_shuffled/task1/fusion/seed0/model.pt --out out/gradcam_val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.interpretability import (GradCAM2D,  # noqa: E402
                                       band_fractions, load_task, pick_target_layer)
from scg_hvd.channels import select_scg_channels  # noqa: E402
from scg_hvd.models import build_model  # noqa: E402
from scg_hvd.paths import data_root  # noqa: E402

RESOLUTIONS = (7, 14, 28)
DATA = data_root(required=False)


def class_maps(model, df, class_names, image_dir, layer_size, n_per_class, device):
    """Class-averaged Grad-CAM maps at one attribution resolution."""
    from PIL import Image
    from scg_hvd.datasets import build_image_transform

    target = pick_target_layer(model.trunk_2d.shared_backbone, want=layer_size)
    if target is None:
        return None
    engine = GradCAM2D(model, target, axis=0)   # images are passed as (x, y, z)
    tf = build_image_transform()
    image_dir = Path(image_dir)
    out = {}
    for ci, cls in enumerate(class_names):
        sub = df[df.label == ci]
        if sub.empty:
            continue
        sub = sub.sample(min(n_per_class, len(sub)), random_state=42)
        acc = []
        for _, r in sub.iterrows():
            stem = Path(r.filepath).stem
            try:
                ims = [tf(Image.open(image_dir / cls / f"{stem}_{ax}.png").convert("RGB"))
                       .unsqueeze(0).to(device) for ax in ("x", "y", "z")]
                sig = select_scg_channels(np.load(r.filepath).astype(np.float32))
                x1 = torch.from_numpy(sig.T.copy()).unsqueeze(0).to(device)
            except Exception:
                continue
            acc.append(engine((x1, *ims), class_idx=ci)[0])
        if acc:
            out[cls] = np.mean(acc, axis=0)
    return out or None


def main():
    global DATA
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--shuffled-ckpt", type=Path, default=None)
    ap.add_argument("--n-per-class", type=int, default=30)
    ap.add_argument("--out", type=Path, default=Path("out/gradcam_val"))
    a = ap.parse_args()
    DATA = data_root()
    a.out.mkdir(parents=True, exist_ok=True)

    df, class_names = load_task(a.task)
    image_dir = DATA / ("Task1_images" if a.task == "task1" else "Task2_images")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    variants = {"trained": a.ckpt, "random": None}
    if a.shuffled_ckpt:
        variants["shuffled"] = a.shuffled_ckpt

    rows = []
    for vname, ckpt in variants.items():
        model = build_model("fusion", len(class_names)).to(device).eval()
        if ckpt is not None:
            model.load_state_dict(torch.load(ckpt, map_location=device))
        for res in RESOLUTIONS:
            maps = class_maps(model, df, class_names, image_dir, res, a.n_per_class, device)
            if maps is None:
                print(f"  [{vname} @ {res}x{res}] no usable samples"); continue
            for cls, m in maps.items():
                for band, (frac, expect, ratio) in band_fractions(m).items():
                    rows.append({"variant": vname, "resolution": res, "class": cls,
                                 "band": band, "frac": round(frac, 4),
                                 "uniform": round(expect, 4), "ratio": round(ratio, 3)})
            print(f"  [{vname} @ {res}x{res}] {len(maps)} classes")

    if not rows:
        print("nothing computed."); return
    t = pd.DataFrame(rows)
    t.to_csv(a.out / f"{a.task}_gradcam_bands.csv", index=False)

    print(f"\n=== band ratio by variant and resolution (1.0 = the band's own share) ===")
    piv = t.pivot_table(index=["band", "class"], columns=["variant", "resolution"],
                        values="ratio")
    print(piv.round(2).to_string())

    # --- decision rule -------------------------------------------------------------
    verdict = {"bands": list(band_fractions(np.ones((7, 7)))), "resolutions": list(RESOLUTIONS)}
    ctrl = [v for v in variants if v != "trained"]
    tr = t[t.variant == "trained"]
    co = t[t.variant.isin(ctrl)]

    # 1. does the trained model leave the range the controls occupy?
    sep = {}
    for band in tr.band.unique():
        a_r = tr[tr.band == band].ratio
        b_r = co[co.band == band].ratio
        if b_r.empty:
            continue
        sep[band] = {"trained_range": [round(a_r.min(), 2), round(a_r.max(), 2)],
                     "control_range": [round(b_r.min(), 2), round(b_r.max(), 2)],
                     "separated": bool(a_r.min() > b_r.max() or a_r.max() < b_r.min())}
    verdict["separation_from_controls"] = sep

    # 2. does the class ordering hold across resolutions?
    agree = {}
    for band in tr.band.unique():
        sub = tr[tr.band == band].pivot_table(index="class", columns="resolution", values="ratio")
        if sub.shape[1] < 2:
            continue
        corr = sub.corr(method="spearman").values
        off = corr[np.triu_indices_from(corr, k=1)]
        agree[band] = {"min_spearman": round(float(np.nanmin(off)), 3),
                       "consistent": bool(np.nanmin(off) > 0.5)}
    verdict["consistency_across_resolutions"] = agree

    ok = (any(v["separated"] for v in sep.values())
          and all(v["consistent"] for v in agree.values()) and bool(agree))
    verdict["reportable"] = bool(ok)
    (a.out / f"{a.task}_gradcam_verdict.json").write_text(json.dumps(verdict, indent=2))

    print("\n=== verdict ===")
    for band, v in sep.items():
        print(f"  {band:12s} trained {v['trained_range']} vs control {v['control_range']}"
              f"  -> {'separated' if v['separated'] else 'overlapping'}")
    for band, v in agree.items():
        print(f"  {band:12s} across resolutions, min Spearman {v['min_spearman']:+.2f}"
              f"  -> {'consistent' if v['consistent'] else 'not consistent'}")
    print("\n" + ("Reportable: the attribution separates from the controls and holds across "
                  "resolutions." if ok else
                  "Not reportable: it does not clear both conditions, so we do not present it."))
    print(f"output: {a.out}")


if __name__ == "__main__":
    main()
