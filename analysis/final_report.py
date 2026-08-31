# Assembles the revision's results: every number that goes into the response letter or the
# manuscript tables comes out of one run of this.
"""
Collects what `run_patient_cv.py` left behind per seed and puts it in the shape the revision
needs.

What it produces.
  1. Patient- and segment-level performance per configuration, with 95% CIs across folds.
  2. The increment over two baselines: the majority class, and cohort covariates alone
     (age + sex + heart rate).
  3. Paired McNemar tests between configurations, at patient level.
  4. A LaTeX table for the manuscript.

Two baselines rather than one, because they answer different questions. The majority-class
baseline says what you get for learning nothing. The covariate baseline says what you get from
the cohort without looking at the SCG signal at all. A model has to clear the second to be
worth anything, and the gap above it is the part attributable to the signal.

Confidence intervals here come from the spread across folds and nothing else. An interval that
treats thousands of windows as independent samples would be far too narrow, and we do not
compute one anywhere.

Usage.
    python analysis/final_report.py --task task1 --out out/final
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.metrics import (  # noqa: E402
    macro_metrics, majority_baseline, mcnemar_paired, metrics_table,
)

ORDER = ["1d", "2d", "fusion", "temporal_matched", "resnet1d_matched", "tcn_matched"]
LABEL = {
    "1d": "Temporal encoder (1D)",
    "2d": "Image encoder (2D)",
    "fusion": "Dual-domain (1D + 2D)",
    "temporal_matched": "Temporal encoder, parameter-matched",
    "resnet1d_matched": "1D ResNet, parameter-matched",
    "tcn_matched": "TCN, parameter-matched",
}
PARAMS = {"1d": 526_740, "2d": 4_203_263, "fusion": 4_767_374,
          "temporal_matched": 4_835_640, "resnet1d_matched": 4_914_197,
          "tcn_matched": 5_063_173}


def rebuild_from_folds(seed_dir: Path):
    """Rebuild the summary from the per-fold directories when fold_summary.csv is missing.

    `fold_summary.csv` and `all_patient_predictions.csv` are written only after every fold has
    finished, so an array job that hits its wall clock loses the folds that did complete along
    with the ones that did not. Each fold leaves its own `predictions.csv` and
    `patient_predictions.csv` behind, which is enough to reconstruct the same table.
    """
    rows, pats = [], []
    for fd in sorted(seed_dir.glob("seed*_fold*")):
        pf, sf = fd / "patient_predictions.csv", fd / "predictions.csv"
        if not (pf.exists() and sf.exists()):
            continue
        m = re.search(r"seed(\d+)_fold(\d+)", fd.name)
        seed, fold = int(m.group(1)), int(m.group(2))
        seg = pd.read_csv(sf); pat = pd.read_csv(pf)
        n_cls = sum(c.startswith("prob_") for c in seg.columns)
        cn = [str(i) for i in range(n_cls)]
        rows.append({
            "seed": seed, "fold": fold,
            "n_test_patients": len(pat), "n_test_segments": len(seg),
            "segment_accuracy": float((seg.y_pred == seg.y_true).mean()),
            "segment_macro_f1": macro_metrics(seg.y_true.values, seg.y_pred.values, cn)["macro_f1"],
            "patient_accuracy": float((pat.y_pred == pat.y_true).mean()),
            "patient_macro_f1": macro_metrics(pat.y_true.values, pat.y_pred.values, cn)["macro_f1"],
            "majority_accuracy": majority_baseline(pat.y_true.values, cn)["accuracy_plain"],
        })
        pats.append(pat.assign(seed=seed, fold=fold))
    if not rows:
        return None, None
    return pd.DataFrame(rows), (pd.concat(pats, ignore_index=True) if pats else None)


def collect(root: Path, task: str, model: str):
    base = root / task / model
    folds, pats, names, rebuilt = [], [], None, []
    for d in sorted(base.glob("seed*")):
        if (d / "fold_summary.csv").exists():
            folds.append(pd.read_csv(d / "fold_summary.csv"))
            if (d / "all_patient_predictions.csv").exists():
                pats.append(pd.read_csv(d / "all_patient_predictions.csv"))
        else:
            # The job is still running or was cut short. Recover from the folds that landed.
            f, p = rebuild_from_folds(d)
            if f is not None:
                folds.append(f); rebuilt.append(f"{d.name}({len(f)} fold)")
                if p is not None:
                    pats.append(p)
        if names is None and (d / "config.json").exists():
            names = json.loads((d / "config.json").read_text()).get("class_names")
    if rebuilt:
        print(f"  [partial recovery] {model}: {', '.join(rebuilt)}")
    if not folds:
        return None
    return {"folds": pd.concat(folds, ignore_index=True),
            "preds": pd.concat(pats, ignore_index=True) if pats else None,
            "class_names": names}


def ci(v):
    m, s, n = float(v.mean()), float(v.std()), len(v)
    h = 1.96 * s / np.sqrt(n) if n > 1 else 0.0
    return m, s, m - h, m + h


def covariate_baseline(path=Path("out/covariates/covariate_baseline_multiclass.json")):
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    return d.get("age+sex+hr") or d.get("age+sex")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--cv", type=Path, default=Path("out/patient_cv"))
    ap.add_argument("--out", type=Path, default=Path("out/final"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    got, missing = {}, []
    for m in ORDER:
        r = collect(a.cv, a.task, m)
        (got.__setitem__(m, r) if r else missing.append(m))
    if missing:
        print(f"[incomplete] {', '.join(missing)}\n")
    if not got:
        print("no results found."); return

    names = next((v["class_names"] for v in got.values() if v["class_names"]), None)

    # ---- main table ----
    rows = []
    for m, r in got.items():
        f = r["folds"]
        pa = ci(f.patient_accuracy); pf = ci(f.patient_macro_f1)
        sa = ci(f.segment_accuracy); sf = ci(f.segment_macro_f1)
        rows.append({
            "model": m, "label": LABEL[m], "params": PARAMS.get(m),
            "n_folds": len(f), "n_seeds": f.seed.nunique(),
            "pat_acc": pa[0], "pat_acc_sd": pa[1], "pat_acc_lo": pa[2], "pat_acc_hi": pa[3],
            "pat_f1": pf[0], "pat_f1_sd": pf[1], "pat_f1_lo": pf[2], "pat_f1_hi": pf[3],
            "seg_acc": sa[0], "seg_acc_sd": sa[1], "seg_f1": sf[0], "seg_f1_sd": sf[1],
            "majority": float(f.majority_accuracy.mean()),
        })
    t = pd.DataFrame(rows)
    t.to_csv(a.out / f"{a.task}_main_results.csv", index=False)

    maj = float(t.majority.mean())
    cov = covariate_baseline()

    print(f"===== {a.task}: patient-level cross-validation =====")
    print(f"{'configuration':38s} {'seed':>4s} "
          f"{'patient accuracy (95% CI)':>28s} {'macro-F1 (95% CI)':>26s}")
    for _, r in t.iterrows():
        print(f"{r.label:38s} {int(r.n_seeds):>4d} "
              f"{r.pat_acc:.4f} [{r.pat_acc_lo:.4f}, {r.pat_acc_hi:.4f}]   "
              f"{r.pat_f1:.4f} [{r.pat_f1_lo:.4f}, {r.pat_f1_hi:.4f}]")
    print(f"{'majority-class baseline':38s} {'':>4s} {maj:.4f}")
    if cov and "accuracy" in cov:
        print(f"{'covariates only (age+sex+HR)':38s} {'':>4s} {cov['accuracy']:.4f}"
              f"{'':22s}{cov['macro_f1']:.4f}")

    print(f"\n===== increment over the baselines =====")
    for _, r in t.iterrows():
        d1 = r.pat_acc - maj
        s1 = "above" if r.pat_acc_lo > maj else ("below" if r.pat_acc_hi < maj else "overlaps")
        line = f"  {r.label:38s} vs majority {d1:+.4f} (CI {s1})"
        if cov and "accuracy" in cov:
            d2 = r.pat_acc - cov["accuracy"]
            s2 = ("above" if r.pat_acc_lo > cov["accuracy"]
                  else "below" if r.pat_acc_hi < cov["accuracy"] else "overlaps")
            line += f" | vs covariates {d2:+.4f} (CI {s2})"
        print(line)

    # ---- McNemar ----
    print(f"\n===== paired McNemar between configurations (patient level) =====")
    key = ["seed", "fold", "patient_id"]
    avail = [m for m in ORDER if m in got and got[m]["preds"] is not None]
    mc = []
    for i in range(len(avail)):
        for j in range(i + 1, len(avail)):
            A, B = avail[i], avail[j]
            da = got[A]["preds"].drop_duplicates(key).set_index(key)
            db = got[B]["preds"].drop_duplicates(key).set_index(key)
            common = da.index.intersection(db.index)
            if len(common) < 10:
                continue
            ca, cb = da.loc[common], db.loc[common]
            r = mcnemar_paired(ca.y_true.values, ca.y_pred.values, cb.y_pred.values)
            mc.append({"a": A, "b": B, "n": len(common),
                       "seeds": len(set(k[0] for k in common)), **r})
            flag = "  <== significant" if r["p_value"] < 0.05 else ""
            print(f"  {LABEL[A]:34s} vs {LABEL[B]:34s} "
                  f"{r['accuracy_a']:.3f}/{r['accuracy_b']:.3f} n={len(common):3d} "
                  f"p={r['p_value']:.4f}{flag}")
    if mc:
        pd.DataFrame(mc).to_csv(a.out / f"{a.task}_mcnemar.csv", index=False)

    # ---- per-class, pooled over folds ----
    for m in avail:
        p = got[m]["preds"]
        cn = got[m]["class_names"] or names or [str(i) for i in range(int(p.y_true.max()) + 1)]
        tab = metrics_table(p.y_true.values, p.y_pred.values, cn)
        tab.to_csv(a.out / f"{a.task}_{m}_pooled_patient.csv", index=False)

    # ---- LaTeX ----
    lines = ["\\begin{table}[ht]", "\\centering", "\\begin{threeparttable}",
             f"\\caption{{\\rev{{Patient-level cross-validation for "
             f"{'Task~I' if a.task == 'task1' else 'Task~II'}. "
             "Mean $\\pm$ SD over folds, with 95\\% confidence intervals computed across "
             "folds rather than across segments. Baselines are shown for reference.}}",
             f"\\label{{tbl:patientcv_{a.task}}}", "\\renewcommand{\\arraystretch}{1.2}",
             "\\setlength{\\tabcolsep}{5pt}", "\\footnotesize",
             "\\begin{tabular}{l|c|c|c}", "\\hline\\hline",
             "\\textbf{Configuration} & \\textbf{Params} & \\textbf{Patient accuracy} & "
             "\\textbf{Patient macro-F1} \\\\", "\\hline"]
    for _, r in t.iterrows():
        pp = f"{int(r.params):,}".replace(",", "{,}") if r.params == r.params else "--"
        lines.append(f"{r.label} & {pp} & "
                     f"{r.pat_acc*100:.1f} $\\pm$ {r.pat_acc_sd*100:.1f} & "
                     f"{r.pat_f1:.3f} $\\pm$ {r.pat_f1_sd:.3f} \\\\")
    lines.append("\\hline")
    lines.append(f"Majority-class baseline & -- & {maj*100:.1f} & -- \\\\")
    if cov and "accuracy" in cov:
        lines.append(f"Covariates only (age, sex, HR) & -- & {cov['accuracy']*100:.1f} & "
                     f"{cov['macro_f1']:.3f} \\\\")
    lines += ["\\hline\\hline", "\\end{tabular}", "\\begin{tablenotes}[flushleft]\\footnotesize",
              "\\item No patient contributes segments to more than one split in any fold, and "
              "no test "
              "segment has an overlapping neighbour in training or validation; both are "
              "asserted in "
              "the released code.",
              "\\end{tablenotes}", "\\end{threeparttable}", "\\end{table}"]
    (a.out / f"{a.task}_table.tex").write_text("\n".join(lines))
    print(f"\nLaTeX table: {a.out / (a.task + '_table.tex')}")
    print(f"output: {a.out}")


if __name__ == "__main__":
    main()
