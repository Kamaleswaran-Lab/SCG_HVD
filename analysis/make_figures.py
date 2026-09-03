# Builds the figures for the response letter, numbered R1, R2, R3 to keep them clear of the
# manuscript's own figure numbers.
"""
What it makes.

  FigR1  Patient-level cross-validation: accuracy per configuration with 95% CIs across
         folds, and both baselines drawn in.
  FigR2  Overlapping versus non-overlapping windows: removing the overlap does not cost
         performance.
  FigR3  Swapping the image backbone: the gain from adding the temporal branch recurs
         whichever backbone is underneath.
  FigR4  Where the temporal branch attends within the cardiac cycle, per class and axis.
  FigR5  The same attention as a curve against time from the R-peak.

Rules the figures follow.
  - Always draw the baselines. A five-class accuracy is uninterpretable without them.
  - Error bars are 95% CIs across folds. Never across segments, which are not independent.
  - Colour encodes role, not identity: baselines grey and dashed, only the proposed
    configuration emphasised.
  - Markers and line styles carry the distinction too, so the figures survive a monochrome
    printer.

Usage.
    python analysis/make_figures.py --out out/figures
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _use_times():
    """Match the Times of the manuscript's other figures, which were drawn in MATLAB.

    Times New Roman is not installed here. TeX Gyre Termes ships with TeX Live and is
    metrically compatible with it, so the glyphs sit at the same widths. If neither is
    present matplotlib falls back through the list and the figure still renders.
    """
    from matplotlib import font_manager as fm

    for otf in glob.glob(str(Path.home() / "texlive" / "*" / "texmf-dist" / "fonts"
                             / "opentype" / "public" / "tex-gyre" / "texgyretermes-*.otf")):
        try:
            fm.fontManager.addfont(otf)
        except Exception:
            pass
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "TeX Gyre Termes",
                       "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


_use_times()

from analysis.final_report import collect  # noqa: E402

# One accent for the condition under test, one neutral for what it is compared against, and a
# grey for reference lines. Colour carries the comparison, not emphasis: the fused model is not
# painted differently from the encoders it is being compared with.
C_A = "#4c4c4c"        # first condition
C_B = "#0072B2"        # second condition
C_BASE = "#9a9a9a"     # baselines and reference lines
SHORT = {
    "1d": "Temporal\n(1D)", "2d": "Image\n(2D)", "fusion": "Dual-domain\n(1D+2D)",
    "temporal_matched": "Temporal\nmatched", "resnet1d_matched": "1D ResNet\nmatched",
    "tcn_matched": "TCN\nmatched",
}


def ci(v):
    m, s, n = float(v.mean()), float(v.std()), len(v)
    return m, (1.96 * s / np.sqrt(n) if n > 1 else 0.0)


def covariate_baseline():
    p = Path("out/covariates/covariate_baseline_multiclass.json")
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return (d.get("age+sex+hr") or d.get("age+sex", {})).get("accuracy")


def fig_patient_cv(cv_root: Path, out: Path, models=("1d", "2d", "fusion")):
    """FigR1: patient-level accuracy per configuration against the two baselines.

    The table beside this figure already gives the numbers, so the figure is here for the one
    thing a table shows badly: how far the intervals overlap each other and how far all three
    sit above the baselines. Nothing is annotated that the table already states.
    """
    got = [(m, collect(cv_root, "task1", m)) for m in models]
    got = [(m, r) for m, r in got if r]
    if not got:
        return None
    fig, ax = plt.subplots(figsize=(0.95 * len(got) + 2.5, 2.9))
    maj = []
    for i, (m, r) in enumerate(got):
        mu, h = ci(r["folds"].patient_accuracy)
        maj.append(r["folds"].majority_accuracy.mean())
        ax.errorbar(i, mu, yerr=h, fmt="o", ms=6, color=C_B, ecolor=C_B,
                    capsize=4, lw=1.3, zorder=4)

    mb = float(np.mean(maj))
    cb = covariate_baseline()
    for y, ls, lab in ((mb, "--", "majority class"), (cb, ":", "covariates only")):
        if y is None:
            continue
        ax.axhline(y, color=C_BASE, ls=ls, lw=1.1, zorder=2)
        ax.text(len(got) - 0.42, y, f" {lab}", ha="left", va="center",
                fontsize=8, color="#555")

    ax.set_xticks(range(len(got)))
    ax.set_xticklabels([SHORT.get(m, m) for m, _ in got], fontsize=9)
    ax.set_xlim(-0.5, len(got) - 0.45)
    ax.set_ylabel("patient-level accuracy")
    ax.set_ylim(0.30, 0.78)
    ax.grid(axis="y", alpha=.2, lw=.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out

def fig_overlap(ov_root: Path, no_root: Path, out: Path):
    """FigR2: patient-level accuracy with and without window overlap.

    Paired within configuration, so the eye compares the two conditions rather than the three
    models. The point the analysis makes is that no configuration falls; the figure shows that
    and leaves the reader to judge the intervals.
    """
    models = ["1d", "2d", "fusion"]
    data = []
    for m in models:
        a, b = collect(ov_root, "task1", m), collect(no_root, "task1", m)
        if a and b:
            data.append((m, ci(a["folds"].patient_accuracy), ci(b["folds"].patient_accuracy),
                         a["folds"].majority_accuracy.mean()))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(1.05 * len(data) + 2.6, 2.9))
    d = 0.15
    for i, (m, (m1, h1), (m2, h2), _) in enumerate(data):
        ax.errorbar(i - d, m1, yerr=h1, fmt="o", ms=6, color=C_A, ecolor=C_A,
                    capsize=4, lw=1.3, zorder=4,
                    label="overlapping windows" if i == 0 else None)
        ax.errorbar(i + d, m2, yerr=h2, fmt="s", ms=6, color=C_B, ecolor=C_B,
                    capsize=4, lw=1.3, zorder=4,
                    label="every second window" if i == 0 else None)
        ax.plot([i - d, i + d], [m1, m2], color="#bbb", lw=1.0, zorder=3)

    mb = float(np.mean([x[3] for x in data]))
    ax.axhline(mb, color=C_BASE, ls="--", lw=1.1, zorder=2)
    ax.text(len(data) - 0.42, mb, " majority class", ha="left", va="center",
            fontsize=8, color="#555")

    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([SHORT.get(m, m) for m, *_ in data], fontsize=9)
    ax.set_xlim(-0.5, len(data) - 0.45)
    ax.set_ylabel("patient-level accuracy")
    ax.set_ylim(0.30, 0.80)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", handletextpad=.5)
    ax.grid(axis="y", alpha=.2, lw=.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out

def fig_backbone(cv_root: Path, bb_root: Path, out: Path):
    """FigR3: the image-to-fusion step, repeated with the image backbone varied.

    Drawn as a paired shift within each backbone, because the claim is about the direction of
    the step and not about which backbone is best. The step costs the same 564,111 parameters
    in every case, so the comparison is of the branch and not of capacity.
    """
    pairs = [("EfficientNet-B0",
              collect(cv_root, "task1", "2d"), collect(cv_root, "task1", "fusion"))]
    for key, name in (("resnet18", "ResNet-18"), ("mobilenet", "MobileNetV3"),
                      ("densenet", "DenseNet")):
        a, b = collect(bb_root, "task1", f"2d_{key}"), collect(bb_root, "task1", f"fusion_{key}")
        if a and b:
            pairs.append((name, a, b))
    # Drop pairs that have not accumulated enough folds. One or two folds cannot establish a
    # direction, and a plot that shows them as if they could is worse than no plot.
    MIN_FOLDS = 10
    dropped = [(n, len(a["folds"]), len(b["folds"])) for n, a, b in pairs
               if a and b and min(len(a["folds"]), len(b["folds"])) < MIN_FOLDS]
    pairs = [(n, a, b) for n, a, b in pairs
             if a and b and min(len(a["folds"]), len(b["folds"])) >= MIN_FOLDS]
    for n, na, nb in dropped:
        print(f"     [skipped] {n}: {na}/{nb} folds, below the {MIN_FOLDS} needed to plot")
    if not pairs:
        return None

    fig, ax = plt.subplots(figsize=(1.05 * len(pairs) + 2.6, 2.9))
    d = 0.15
    for i, (name, a, b) in enumerate(pairs):
        m1, h1 = ci(a["folds"].patient_accuracy)
        m2, h2 = ci(b["folds"].patient_accuracy)
        ax.errorbar(i - d, m1, yerr=h1, fmt="o", ms=6, color=C_A, ecolor=C_A,
                    capsize=4, lw=1.3, zorder=4, label="image encoder" if i == 0 else None)
        ax.errorbar(i + d, m2, yerr=h2, fmt="s", ms=6, color=C_B, ecolor=C_B,
                    capsize=4, lw=1.3, zorder=4, label="dual-domain" if i == 0 else None)
        ax.plot([i - d, i + d], [m1, m2], color="#bbb", lw=1.0, zorder=3)

    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([n for n, _, _ in pairs], fontsize=9)
    ax.set_xlim(-0.5, len(pairs) - 0.45)
    ax.set_xlabel("image-branch backbone")
    ax.set_ylabel("patient-level accuracy")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left", handletextpad=.5)
    ax.grid(axis="y", alpha=.2, lw=.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out

def fig_attention(interp_root: Path, out: Path, task="task1"):
    """FigR4: where the temporal branch attends, relative to the R-peak.

    One number per class and axis: the share of the attention mass falling in the 0 to 0.35 s
    window after the R-peak. A model attending uniformly over the window would put 0.437 there,
    which is drawn as the reference line.

    This comes from a single trained fold, so it describes that model rather than establishing
    a property of the architecture. The figure and the caption both say so.
    """
    frames = {}
    for m in ("1d", "fusion"):
        f = interp_root / f"{task}_{m}_systolic_fraction.csv"
        if f.exists():
            frames[m] = pd.read_csv(f).pivot(index="class", columns="axis",
                                             values="systolic_fraction")
    if not frames:
        return None

    UNIFORM = 0.35 / 0.8
    fig, axes = plt.subplots(1, len(frames), figsize=(4.2 * len(frames) + 0.6, 3.5),
                             sharey=True)
    if len(frames) == 1:
        axes = [axes]
    marks = {"x": ("o", C_A), "y": ("s", "#b07a3c"), "z": ("^", C_B)}
    for ax, (m, t) in zip(axes, frames.items()):
        xs = np.arange(len(t.index))
        for axis_name in ("x", "y", "z"):
            if axis_name not in t:
                continue
            mk, col = marks[axis_name]
            ax.plot(xs, t[axis_name].values, mk, ms=7, color=col, ls="-", lw=1.0,
                    alpha=.85, label=f"SCG {axis_name}", zorder=3)
        ax.axhline(UNIFORM, color=C_BASE, ls="--", lw=1.3, zorder=2)
        ax.set_xticks(xs)
        ax.set_xticklabels(t.index, fontsize=9)
        ax.set_title(SHORT.get(m, m).replace("\n", " "), fontsize=10)
        ax.grid(axis="y", alpha=.25, lw=.5, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("attention mass in systole\n(R+0 to R+0.35 s)")
    axes[-1].text(len(frames[list(frames)[-1]].index) - 0.6, UNIFORM + .006,
                  f"uniform ({UNIFORM:.3f})", ha="right", fontsize=8, color="#555")
    # The legend sits under the axes: inside the panel it lands on the AR points.
    axes[0].legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center",
                   bbox_to_anchor=(0.5, -0.12))
    for ax in axes:
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.02)
    fig.suptitle("Where the temporal branch attends within the cardiac cycle\n"
                 "(Task I, one trained fold; descriptive, not a repeated measurement)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def fig_attention_curve(interp_root: Path, out: Path, task="task1", model="fusion",
                        title=True):
    """FigR5: attention over the cardiac cycle, averaged within class, across seeds.

    One trained model describes that model. Three, trained on different seeds and therefore on
    different fold partitions, say something about the architecture. The solid line is the mean
    over seeds and the band is their full range, so how much of the shape is stable and how much
    is the particular model is visible rather than asserted.
    """
    frames = []
    for d in sorted(interp_root.glob("seed*")):
        f = d / f"{task}_{model}_beat_aligned_attention.csv"
        if f.exists():
            g = pd.read_csv(f); g["seed"] = d.name
            frames.append(g)
    if not frames:
        return None
    d = pd.concat(frames)
    n_seeds = d.seed.nunique()

    axes_order = [a for a in ("z", "x", "y") if a in set(d.axis)]
    classes = sorted(d["class"].unique())
    # Colour is tied to the class, not to its position in the list, so adding or dropping a
    # class never repaints the others. The four lesion hues pass the colour-vision checks as an
    # adjacent set; the healthy class is the reference and is drawn black and dashed. Line style
    # varies too, so the panel survives greyscale printing.
    colors = {"AR": "#0072B2", "AS": "#D55E00", "MR": "#009E73", "MS": "#8E44AD",
              "N": "#111111"}
    styles = {"AR": "-", "AS": (0, (5, 1.5)), "MR": (0, (1, 1.2)), "MS": (0, (4, 1.2, 1, 1.2)),
              "N": (0, (7, 2))}
    fallback = plt.get_cmap("tab10")
    for i, c in enumerate(classes):
        colors.setdefault(c, fallback(i % 10))
        styles.setdefault(c, "-")

    # Sized close to the text width it is printed at, so the labels are not shrunk to
    # illegibility by \includegraphics scaling.
    fig, axs = plt.subplots(1, len(axes_order), figsize=(2.35 * len(axes_order) + 0.5, 2.6),
                            sharey=True)
    if len(axes_order) == 1:
        axs = [axs]
    for ax, axis_name in zip(axs, axes_order):
        sub = d[d.axis == axis_name]
        for cls in classes:
            g = sub[sub["class"] == cls]
            if g.empty:
                continue
            piv = g.pivot_table(index="t_from_R_sec", columns="seed", values="attention")
            t = piv.index.values
            ax.fill_between(t, piv.min(axis=1), piv.max(axis=1),
                            color=colors[cls], alpha=.15, lw=0, zorder=2)
            ax.plot(t, piv.mean(axis=1), lw=1.7, color=colors[cls], label=cls,
                    ls=styles[cls], zorder=4 if cls == "N" else 3)
        ax.axvline(0, color="#444", lw=1.0, zorder=2)
        ax.axvspan(0, 0.35, color="#999", alpha=.10, zorder=0)
        # Weights are a softmax over the window's 2560 samples, so uniform is 1/2560.
        ax.axhline(1.0 / 2560, color="#c0392b", ls=":", lw=1.2, zorder=2)
        ax.set_title(f"SCG {axis_name}", fontsize=10)
        ax.set_xlabel("time from R-peak (s)")
        ax.grid(alpha=.2, lw=.5, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("mean attention weight")
    axs[0].annotate("uniform", xy=(-0.19, 1.0 / 2560), xytext=(0, 3),
                    textcoords="offset points", fontsize=7.5, color="#c0392b", va="bottom")
    axs[-1].legend(frameon=False, fontsize=8.5, loc="upper right", ncol=2)
    # The manuscript carries this in the LaTeX caption, so the in-figure title is optional.
    if title:
        fig.suptitle("Attention against time from the R-peak, averaged within class\n"
                     f"(Task I, temporal branch of the fused model; line is the mean over "
                     f"{n_seeds} seeds, band their range)", fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cv = Path("out/patient_cv")
    no = Path("out/patient_cv_nonoverlap")
    bb = Path("out/backbone_ablation")
    interp = Path("out/interp")
    made = []
    for fn, args, name in (
        (fig_patient_cv, (cv, a.out / "FigR1_patient_cv"), "FigR1 patient-level CV"),
        (fig_overlap, (cv, no, a.out / "FigR2_overlap"), "FigR2 window overlap"),
        (fig_backbone, (cv, bb, a.out / "FigR3_backbone"), "FigR3 backbone ablation"),
        (fig_attention, (interp, a.out / "FigR4_attention"), "FigR4 attention in systole"),
        (fig_attention_curve, (interp, a.out / "FigR5_attention_curve"),
         "FigR5 attention over the cardiac cycle"),
    ):
        r = fn(*args)
        print(f"  {'ok  ' if r else 'skip'}  {name}"
              + (f" -> {r}.pdf" if r else "  (not enough data)"))
        if r:
            made.append(r)
    print(f"\nwrote {len(made)} figure(s) to {a.out}")


if __name__ == "__main__":
    main()
