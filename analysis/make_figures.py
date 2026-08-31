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
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analysis.final_report import collect  # noqa: E402

# Colour by role: the proposed configuration saturated, everything else neutral.
C_PROPOSED = "#1f5fa9"
C_OTHER = "#7f8c9a"
C_BASE = "#999999"
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
    """FigR1: patient-level accuracy per configuration, with both baselines drawn in."""
    got = [(m, collect(cv_root, "task1", m)) for m in models]
    got = [(m, r) for m, r in got if r]
    if not got:
        return None
    fig, ax = plt.subplots(figsize=(1.55 * len(got) + 2.6, 3.6))
    xs, maj = np.arange(len(got)), []
    for i, (m, r) in enumerate(got):
        mu, h = ci(r["folds"].patient_accuracy)
        maj.append(r["folds"].majority_accuracy.mean())
        col = C_PROPOSED if m == "fusion" else C_OTHER
        ax.bar(i, mu, 0.6, color=col, alpha=.85, zorder=3)
        ax.errorbar(i, mu, yerr=h, fmt="none", ecolor="black", capsize=4, lw=1.2, zorder=4)
        ax.text(i, mu + h + .012, f"{mu:.3f}", ha="center", fontsize=9, zorder=5)

    mb = float(np.mean(maj))
    ax.axhline(mb, color=C_BASE, ls="--", lw=1.3, zorder=2)
    ax.text(len(got) - .45, mb + .006, f"majority class ({mb:.3f})",
            ha="right", fontsize=8, color="#555")
    cb = covariate_baseline()
    if cb:
        ax.axhline(cb, color=C_BASE, ls=":", lw=1.3, zorder=2)
        ax.text(len(got) - .45, cb + .006, f"age + sex + HR only ({cb:.3f})",
                ha="right", fontsize=8, color="#555")

    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT.get(m, m) for m, _ in got], fontsize=9)
    ax.set_ylabel("patient-level accuracy")
    ax.set_ylim(0, max(0.85, max(ci(r["folds"].patient_accuracy)[0] for _, r in got) + .12))
    ax.set_title("Patient-level cross-validation, Task I\n"
                 "(5-fold $\\times$ 3 seeds; bars are 95% CI across folds)", fontsize=10)
    ax.grid(axis="y", alpha=.25, lw=.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def fig_overlap(ov_root: Path, no_root: Path, out: Path):
    """FigR2: with and without window overlap. Performance does not fall, contrary to the
    concern that prompted the analysis."""
    models = ["1d", "2d", "fusion"]
    data = []
    for m in models:
        a, b = collect(ov_root, "task1", m), collect(no_root, "task1", m)
        if a and b:
            data.append((m, ci(a["folds"].patient_accuracy), ci(b["folds"].patient_accuracy),
                         a["folds"].majority_accuracy.mean()))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(1.7 * len(data) + 2.4, 3.6))
    w, xs = 0.34, np.arange(len(data))
    for i, (m, (m1, h1), (m2, h2), _) in enumerate(data):
        ax.bar(i - w / 2, m1, w, yerr=h1, capsize=3.5, color=C_OTHER, alpha=.85,
               label="overlapping windows" if i == 0 else None, zorder=3,
               error_kw=dict(lw=1.1))
        ax.bar(i + w / 2, m2, w, yerr=h2, capsize=3.5, color=C_PROPOSED, alpha=.85,
               label="non-overlapping windows" if i == 0 else None, zorder=3,
               error_kw=dict(lw=1.1))
    mb = float(np.mean([d[3] for d in data]))
    ax.axhline(mb, color=C_BASE, ls="--", lw=1.3, zorder=2)
    ax.text(len(data) - .55, mb + .006, f"majority class ({mb:.3f})",
            ha="right", fontsize=8, color="#555")
    ax.set_xticks(xs)
    ax.set_xticklabels([SHORT.get(m, m) for m, *_ in data], fontsize=9)
    ax.set_ylabel("patient-level accuracy")
    ax.set_ylim(0, .88)
    ax.set_title("Removing window overlap does not reduce performance\n"
                 "(Task I, patient-level; training data halved in the right bars)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.grid(axis="y", alpha=.25, lw=.5, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return out


def fig_backbone(cv_root: Path, bb_root: Path, out: Path):
    """FigR3: does the image-to-fusion gain recur across backbones? The paper rests on this."""
    pairs = [("efficientnet_b0",
              collect(cv_root, "task1", "2d"), collect(cv_root, "task1", "fusion"))]
    for bb in ("resnet18", "mobilenet", "densenet"):
        a, b = collect(bb_root, "task1", f"2d_{bb}"), collect(bb_root, "task1", f"fusion_{bb}")
        if a and b:
            pairs.append((bb, a, b))
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

    fig, ax = plt.subplots(figsize=(1.9 * len(pairs) + 2.2, 3.8))
    tops, bots = [], []
    for i, (name, a, b) in enumerate(pairs):
        m1, h1 = ci(a["folds"].patient_accuracy)
        m2, h2 = ci(b["folds"].patient_accuracy)
        ax.errorbar([i - .13], [m1], yerr=[h1], fmt="o", ms=7, color=C_OTHER,
                    capsize=4, lw=1.2, label="2D only" if i == 0 else None, zorder=3)
        ax.errorbar([i + .13], [m2], yerr=[h2], fmt="s", ms=7, color=C_PROPOSED,
                    capsize=4, lw=1.2, label="1D + 2D" if i == 0 else None, zorder=3)
        ax.annotate("", xy=(i + .13, m2), xytext=(i - .13, m1),
                    arrowprops=dict(arrowstyle="->", color="#444", lw=1.1, alpha=.8), zorder=2)
        tops.append(max(m1, m2) + max(h1, h2))
        ax.text(i, max(m1, m2) + max(h1, h2) + .004, f"{m2 - m1:+.3f}",
                ha="center", va="bottom", fontsize=9,
                color=C_PROPOSED if m2 > m1 else "#c0392b", zorder=5)
    ax.set_xticks(range(len(pairs)))
    ax.set_xticklabels([f"{n.replace('_','-')}\n({min(len(a['folds']), len(b['folds']))} folds)"
                        for n, a, b in pairs], fontsize=9)
    ax.set_xlabel("image-branch backbone")
    ax.set_ylabel("patient-level accuracy")
    n_pos = sum(1 for _, a, b in pairs
                if ci(b["folds"].patient_accuracy)[0] > ci(a["folds"].patient_accuracy)[0])
    ax.set_title(f"Effect of adding the temporal branch, by image backbone "
                 f"({n_pos}/{len(pairs)} positive)\n"
                 "(Task I, patient-level; +564,111 parameters in every case)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.grid(axis="y", alpha=.25, lw=.5, zorder=0)
    ax.set_xlim(-.5, len(pairs) - .5)
    if tops:
        lo = min(ci(a["folds"].patient_accuracy)[0] - ci(a["folds"].patient_accuracy)[1]
                 for _, a, _ in pairs)
        ax.set_ylim(lo - .02, max(tops) + .035)
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
    marks = {"x": ("o", C_OTHER), "y": ("s", "#b07a3c"), "z": ("^", C_PROPOSED)}
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
    ):
        r = fn(*args)
        print(f"  {'ok  ' if r else 'skip'}  {name}"
              + (f" -> {r}.pdf" if r else "  (not enough data)"))
        if r:
            made.append(r)
    print(f"\nwrote {len(made)} figure(s) to {a.out}")


if __name__ == "__main__":
    main()
