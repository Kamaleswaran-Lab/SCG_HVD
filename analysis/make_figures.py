# 응답 letter 와 원고에 넣을 그림을 만든다. 저자 양식의 R1, R2 번호 체계에 맞춘다.
"""
만드는 것.

  FigR1  환자 단위 교차검증 — 구성별 정확도와 fold 간 95% CI, 두 기준선을 함께 표시
  FigR2  창 겹침 유무 비교 — 겹침을 없애도 성능이 떨어지지 않음을 보인다 (R2-M1)
  FigR3  2D 백본 교체 ablation — 백본이 달라도 2D->융합 이득이 반복됨을 보인다 (R1-M5, R2-M4)

그림 설계에서 지킨 것.
  - 기준선을 항상 같이 그린다. 없으면 절대 수치를 해석할 수 없다.
  - 오차 막대는 fold 간 95% CI 다. 세그먼트를 독립으로 본 CI 는 쓰지 않는다(R2-M5).
  - 색은 구성이 아니라 역할로 준다. 기준선은 회색 점선, 제안 구성만 강조한다.
  - 흑백 인쇄에서도 읽히도록 마커와 선 스타일을 함께 쓴다.

사용법.
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

from analysis.final_report import LABEL, collect  # noqa: E402

# 색은 역할로 준다. 제안 구성만 진하게, 나머지는 중립색.
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
    """FigR1 — 구성별 환자 단위 정확도. 두 기준선을 같이 그린다."""
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
    """FigR2 — 겹침 유무. 리뷰어 우려와 달리 성능이 떨어지지 않음을 보인다."""
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
    """FigR3 — 백본을 바꿔도 2D->융합 이득이 반복되는가. 논문 주장의 직접 증거."""
    pairs = [("efficientnet_b0", collect(cv_root, "task1", "2d"), collect(cv_root, "task1", "fusion"))]
    for bb in ("resnet18", "mobilenet", "densenet"):
        a, b = collect(bb_root, "task1", f"2d_{bb}"), collect(bb_root, "task1", f"fusion_{bb}")
        if a and b:
            pairs.append((bb, a, b))
    # fold 가 충분히 쌓이지 않은 쌍은 제외한다. fold 한두 개로 방향을 말하면 안 된다.
    MIN_FOLDS = 10
    dropped = [(n, len(a["folds"]), len(b["folds"])) for n, a, b in pairs
               if a and b and min(len(a["folds"]), len(b["folds"])) < MIN_FOLDS]
    pairs = [(n, a, b) for n, a, b in pairs
             if a and b and min(len(a["folds"]), len(b["folds"])) >= MIN_FOLDS]
    for n, na, nb in dropped:
        print(f"     [제외] {n}: fold {na}/{nb} — {MIN_FOLDS} 미만이라 그리지 않는다")
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("out/figures"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    cv, no, bb = Path("out/patient_cv"), Path("out/patient_cv_nonoverlap"), Path("out/backbone_ablation")
    made = []
    for fn, args, name in (
        (fig_patient_cv, (cv, a.out / "FigR1_patient_cv"), "FigR1 환자 단위 CV"),
        (fig_overlap, (cv, no, a.out / "FigR2_overlap"), "FigR2 겹침 유무"),
        (fig_backbone, (cv, bb, a.out / "FigR3_backbone"), "FigR3 백본 ablation"),
    ):
        r = fn(*args)
        print(f"  {'O' if r else 'X'}  {name}" + (f" -> {r}.pdf" if r else "  (데이터 부족)"))
        if r:
            made.append(r)
    print(f"\n{len(made)}개 생성: {a.out}")


if __name__ == "__main__":
    main()
