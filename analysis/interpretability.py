# Shows where the model looks: attention weights in the 1D branch, Grad-CAM in the 2D one.
"""
Referee 2 minor 7.

    "Clarify whether attention maps or other interpretability analyses were performed. Attention
     pooling is included in the architecture, but the manuscript does not show which portions of
     the SCG contribute to classification."

The thing the reviewer points at is the attention pooling in the 1D branch, and they are
right that it is the natural place to look: its weights are a softmax over the time axis, so
they answer "which parts of the waveform contribute" directly. The archived Grad-CAM script
(`exp/Task1/3_hvdnet_fusion_model_LSTM_gradcam/gradcam_analysis.py`) works on the 2D
scalograms and so does not answer that question. Both are produced here.

1. Attention weights, distributed over time, per axis and per class. A segment is ten seconds
   long, so raw window position says little; to relate the weights to the cardiac cycle they
   are aligned to ECG R-peaks, which the segment files carry (see
   `analysis/extract_heart_rate.py`). That is what makes a statement like "concentrated in
   early systole" possible at all.
2. Grad-CAM on the last convolution of the shared EfficientNet backbone, showing which
   time-frequency regions of the scalogram contribute.

Read the output narrowly. A high attention weight is not evidence that the interval is
causally important. The manuscript says where the model attends and stops there; it does not
go on to claim those intervals carry the pathology. Referee 2 flagged exactly that overreach
elsewhere in the review.

Usage.
    python analysis/interpretability.py --ckpt <fold directory> --task task1 --out out/interp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.paths import data_root, localize  # noqa: E402

from scg_hvd.datasets import select_scg_channels  # noqa: E402
from scg_hvd.models import build_model  # noqa: E402

DATA = data_root(required=False)
FS = 256
AXIS_ORDER = ["z", "x", "y"]   # the order SCGTrunk1D concatenates them in


# ---------------------------------------------------------------- attention

class AttentionRecorder:
    """Intercept the softmax weights inside SCGBranch.attn."""

    def __init__(self, trunk):
        self.weights = {}
        self._handles = []
        for name in ("branch_z", "branch_x", "branch_y"):
            branch = getattr(trunk, name)
            self._handles.append(
                branch.attn.attn.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inp, out):
            # out: (B, T, 1), the tensor SelfAttention.forward applies its softmax to
            self.weights[name] = torch.softmax(out, dim=1).detach().squeeze(-1).cpu()
        return hook

    def close(self):
        for h in self._handles:
            h.remove()


def r_peaks(sig_1d, fs=FS):
    """R-peaks for aligning attention to the cardiac cycle, detected as in extract_heart_rate."""
    from scipy import signal as sps
    x = sig_1d - np.mean(sig_1d)
    nyq = fs / 2
    b, a = sps.butter(3, [5 / nyq, 15 / nyq], btype="band")
    f = sps.filtfilt(b, a, x)
    window = np.ones(int(.15 * fs)) / int(.15 * fs)
    integ = np.convolve(np.diff(f, prepend=f[0]) ** 2, window, "same")
    pk, _ = sps.find_peaks(integ, height=np.mean(integ) + .5 * np.std(integ),
                           distance=int(.25 * fs))
    return pk


def beat_aligned(weights, peaks, win=(-0.2, 0.6), fs=FS):
    """Cut the attention trace around each R-peak and average, so position can be stated in
    terms of the cardiac cycle rather than of the window."""
    lo, hi = int(win[0] * fs), int(win[1] * fs)
    segs = [weights[p + lo:p + hi] for p in peaks
            if p + lo >= 0 and p + hi <= len(weights) and len(weights[p + lo:p + hi]) == hi - lo]
    return np.mean(segs, axis=0) if segs else None


# ---------------------------------------------------------------- Grad-CAM

class GradCAM2D:
    """Grad-CAM on the last convolution of the shared EfficientNet backbone."""

    def __init__(self, model, target_layer):
        self.model, self.act, self.grad = model, None, None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "act", o))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grad", go[0]))

    def __call__(self, inputs, class_idx=None):
        # cuDNN refuses to run RNN backward on a module in eval mode, and the temporal branch
        # has an LSTM. Falling back to the native implementation for this pass keeps the model
        # in eval mode, which is the model whose attributions we want.
        with torch.backends.cudnn.flags(enabled=False):
            return self._forward_backward(inputs, class_idx)

    def _forward_backward(self, inputs, class_idx):
        self.model.zero_grad()
        out = self.model(*inputs)
        # The index has to live on the same device as `out`; a CPU index against a CUDA
        # logit tensor fails inside gather.
        idx = (out.argmax(1) if class_idx is None
               else torch.as_tensor([class_idx], device=out.device))
        out.gather(1, idx.view(-1, 1).to(out.device)).sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1))
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy()


# ---------------------------------------------------------------- main

def load_task(task):
    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    names = sorted(df.label.unique())
    df["label_name"] = df["label"]
    df["label"] = df["label"].map({n: i for i, n in enumerate(names)})
    df["filepath"] = localize(df["filepath"], DATA)
    return df, names


def plot_beat_aligned(df, out_png, title):
    """Plot the R-peak-aligned attention per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    classes = sorted(df["class"].unique())
    fig, axes = plt.subplots(1, len(AXIS_ORDER), figsize=(4.2 * len(AXIS_ORDER), 3.4),
                             sharey=True)
    if len(AXIS_ORDER) == 1:
        axes = [axes]
    cmap = plt.get_cmap("tab10")
    for ax, axis in zip(axes, AXIS_ORDER):
        for i, cls in enumerate(classes):
            sub = df[(df["class"] == cls) & (df.axis == axis)]
            if sub.empty:
                continue
            ax.plot(sub.t_from_R_sec, sub.attention * 1000, lw=1.4,
                    color=cmap(i % 10), label=cls)
        ax.axvline(0, color="k", lw=0.8, ls="--", alpha=0.6)
        ax.set_title(f"SCG {axis}-axis")
        ax.set_xlabel("time from R-peak (s)")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel(r"mean attention weight ($\times 10^{-3}$)")
    axes[-1].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    fig.savefig(str(out_png).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out_png


def run_gradcam(model, df, class_names, image_dir, out, n_per_class=12, device="cpu"):
    """Class-averaged Grad-CAM maps on the fusion model's shared EfficientNet backbone.

    The scalogram's vertical axis is CWT scale (so, frequency) and its horizontal axis is
    time, which means averaging the maps within a class shows which frequency band at which
    point in the window carries the contribution.

    One caveat that goes on the axis label: the scale grid is linear in scale (1..128) and
    pseudo-frequency is inversely proportional to scale, so the vertical axis is not uniform
    in frequency.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from scg_hvd.datasets import build_image_transform

    # Locate the last convolution in the shared backbone.
    backbone = model.trunk_2d.shared_backbone
    target = None
    for mod in backbone.modules():
        if isinstance(mod, torch.nn.Conv2d):
            target = mod
    if target is None:
        print("  [Grad-CAM] no convolutional layer found"); return None

    cam_engine = GradCAM2D(model, target)
    tf = build_image_transform()
    image_dir = Path(image_dir)
    maps = {c: [] for c in class_names}

    for ci_, cls in enumerate(class_names):
        sub = df[df.label == ci_].sample(min(n_per_class, int((df.label == ci_).sum())),
                                         random_state=42)
        for _, r in sub.iterrows():
            stem = Path(r.filepath).stem
            base = image_dir / r.label_name
            try:
                ims = [tf(Image.open(base / f"{stem}_{ax}.png").convert("RGB"))
                       .unsqueeze(0).to(device)
                       for ax in ("x", "y", "z")]
                sig = np.load(r.filepath).astype(np.float32)
                sig = select_scg_channels(sig)
                x1 = torch.from_numpy(sig.T.copy()).unsqueeze(0).to(device)
            except Exception:
                continue
            cam = cam_engine((x1, *ims), class_idx=ci_)
            maps[cls].append(cam[0])

    avail = {c: np.mean(v, axis=0) for c, v in maps.items() if v}
    if not avail:
        print("  [Grad-CAM] no usable samples"); return None

    fig, axes = plt.subplots(1, len(avail), figsize=(2.5 * len(avail) + 1.2, 3.0))
    if len(avail) == 1:
        axes = [axes]
    vmax = max(m.max() for m in avail.values())
    for ax, (cls, m) in zip(axes, avail.items()):
        im = ax.imshow(m, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax)
        ax.set_title(cls, fontsize=10)
        ax.set_xlabel("time within window")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_ylabel("CWT scale\n(low freq. at top)")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="Grad-CAM (normalized)")
    fig.suptitle("Grad-CAM on the shared image backbone, averaged within class", fontsize=11)
    png = out / "task1_fusion_gradcam.png"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(str(png).replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)

    rows = []
    for cls, m in avail.items():
        h = m.shape[0]
        rows.append({"class": cls,
                     "upper_third_frac": round(float(m[2 * h // 3:].sum() / m.sum()), 4),
                     "mid_third_frac": round(float(m[h // 3:2 * h // 3].sum() / m.sum()), 4),
                     "lower_third_frac": round(float(m[:h // 3].sum() / m.sum()), 4),
                     "n_samples": len(maps[cls])})
    sf = pd.DataFrame(rows)
    sf.to_csv(out / "task1_fusion_gradcam_bands.csv", index=False)
    print("\n  Grad-CAM contribution by scale band (0.333 each if uniform)")
    print(sf.to_string(index=False))
    print(f"  wrote {png}")
    return png


def main():
    global DATA
    DATA = data_root()   # fail here rather than on a puzzling missing file
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--model", default="1d", choices=["1d", "fusion"])
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="trained weights (.pt). Without them the run only checks the wiring.")
    ap.add_argument("--ckpt-dir", type=Path, default=None,
                    help="parent of the fold directories; the first model.pt found is used.")
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("out/interp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df, class_names = load_task(a.task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.model, len(class_names)).to(device).eval()
    if a.ckpt_dir and a.ckpt_dir.exists():
        found = sorted(a.ckpt_dir.rglob("model.pt"))
        if found:
            a.ckpt = found[0]
            print(f"  using {a.ckpt} out of {len(found)} checkpoints found")
    if a.ckpt and a.ckpt.exists():
        model.load_state_dict(torch.load(a.ckpt, map_location=device))
        print(f"loaded {a.ckpt}")
    else:
        print("[warning] running without trained weights. This checks the wiring only; do not\n"
              "          interpret the output.")

    trunk = model.trunk if a.model == "1d" else model.trunk_1d
    rec = AttentionRecorder(trunk)

    rows, aligned = [], {c: {ax: [] for ax in AXIS_ORDER} for c in class_names}
    for cls_i, cls in enumerate(class_names):
        sub = df[df.label == cls_i].sample(min(a.n_per_class, (df.label == cls_i).sum()),
                                           random_state=42)
        for _, r in sub.iterrows():
            raw = np.load(r.filepath).astype(np.float32)
            scg = select_scg_channels(raw)
            x = torch.from_numpy(scg.T.copy()).unsqueeze(0).to(device)
            with torch.no_grad():
                model.extract_features(x) if a.model == "1d" else trunk(x)
            # Detect R-peaks from the ECG so the weights can be aligned to the cardiac cycle
            ecg_idx = 5 if raw.shape[1] == 12 else 0
            pk = r_peaks(raw[:, ecg_idx].astype(np.float64))
            for ax, key in zip(AXIS_ORDER, ("branch_z", "branch_x", "branch_y")):
                w = rec.weights[key][0].numpy()
                rows.append({"class": cls, "axis": ax,
                             "entropy": float(-(w * np.log(w + 1e-12)).sum()),
                             "peak_pos_sec": float(np.argmax(w) / FS),
                             "top10pct_mass": float(np.sort(w)[-len(w) // 10:].sum())})
                if len(pk) >= 3:
                    al = beat_aligned(w, pk)
                    if al is not None:
                        aligned[cls][ax].append(al)
    rec.close()

    d = pd.DataFrame(rows)
    d.to_csv(a.out / f"{a.task}_{a.model}_attention_stats.csv", index=False)
    print(f"\n=== attention summary over {len(d)//3} segments ===")
    print(d.groupby(["class", "axis"])[["entropy", "top10pct_mass", "peak_pos_sec"]]
          .mean().round(3).to_string())
    print("\n  Lower entropy means the weight is concentrated. A uniform distribution has "
          f"entropy {np.log(2560):.3f}.")

    ba = []
    t = np.arange(int(-0.2 * FS), int(0.6 * FS)) / FS
    for cls, per_ax in aligned.items():
        for ax, lst in per_ax.items():
            if lst:
                m = np.mean(lst, axis=0)
                for ti, v in zip(t, m):
                    ba.append({"class": cls, "axis": ax, "t_from_R_sec": round(float(ti), 4),
                               "attention": float(v)})
    if ba:
        bdf = pd.DataFrame(ba)
        bdf.to_csv(a.out / f"{a.task}_{a.model}_beat_aligned_attention.csv", index=False)
        print(f"\n  wrote R-peak-aligned mean attention ({len(aligned)} classes x 3 axes)")
        png = plot_beat_aligned(
            bdf, a.out / f"{a.task}_{a.model}_attention.png",
            "Attention over the cardiac cycle, averaged within class "
            f"({a.model} encoder, R-peak aligned)")
        print(f"  wrote {png}")

        # Systolic share (0 to 0.35 s after R), so "where it looks" is a number too
        sys_mask = (bdf.t_from_R_sec >= 0.0) & (bdf.t_from_R_sec <= 0.35)
        rows = []
        for (cls, ax_), g in bdf.groupby(["class", "axis"]):
            tot = g.attention.sum()
            sysfrac = g[sys_mask.loc[g.index]].attention.sum() / tot if tot else float("nan")
            rows.append({"class": cls, "axis": ax_, "systolic_fraction": round(sysfrac, 4)})
        sf = pd.DataFrame(rows)
        sf.to_csv(a.out / f"{a.task}_{a.model}_systolic_fraction.csv", index=False)
        print("\n  attention mass in systole (R+0 to R+0.35 s)")
        print(sf.pivot(index="class", columns="axis", values="systolic_fraction").to_string())
        print(f"  (uniform over the window would give 0.35/{0.8:.1f} = {0.35/0.8:.3f})")
    if a.model == "fusion":
        print("\n=== Grad-CAM (2D branch) ===")
        img_dir = DATA / ("Task1_images" if a.task == "task1" else "Task2_images")
        try:
            run_gradcam(model, df, class_names, img_dir, a.out,
                        n_per_class=a.n_per_class // 4 or 5, device=device)
        except Exception as e:
            print(f"  [Grad-CAM failed] {type(e).__name__}: {e}")

    print(f"\noutput: {a.out}")


if __name__ == "__main__":
    main()
