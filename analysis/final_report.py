# 리비전 최종 결과 정리. 응답 letter 와 원고 표에 들어갈 숫자를 한 번에 낸다.
"""
`run_patient_cv.py` 가 시드별로 남긴 결과를 모아 리비전에 필요한 형태로 정리한다.

내는 것.
  1. 구성별 환자·세그먼트 단위 성능과 fold 간 95% CI
  2. **두 기준선 대비 증분** — 다수 클래스 기준선과 공변량 기준선(나이+성별+HR)
  3. 구성 간 짝지은 McNemar (환자 단위)
  4. 원고용 LaTeX 표

왜 기준선이 둘인가. 다수 클래스 기준선은 "아무것도 학습하지 않았을 때" 를 준다.
공변량 기준선은 R2-M2 가 요구한 것으로 "SCG 없이 코호트 정보만으로" 를 준다. 모델이
의미가 있으려면 후자를 넘어야 하며, 그 증분이 SCG 신호의 기여분이다.

주의. 세그먼트를 독립 표본으로 보고 만든 신뢰구간은 쓰지 않는다(R2-M5 가 명시적으로 금지).
여기서는 fold 간 분산만 쓴다.

사용법.
    python analysis/final_report.py --task task1 --out out/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.metrics import majority_baseline, mcnemar_paired, metrics_table  # noqa: E402

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


def collect(root: Path, task: str, model: str):
    base = root / task / model
    folds, pats, names = [], [], None
    for d in sorted(base.glob("seed*")):
        if (d / "fold_summary.csv").exists():
            folds.append(pd.read_csv(d / "fold_summary.csv"))
        if (d / "all_patient_predictions.csv").exists():
            pats.append(pd.read_csv(d / "all_patient_predictions.csv"))
        if names is None and (d / "config.json").exists():
            names = json.loads((d / "config.json").read_text()).get("class_names")
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
        print(f"[미완] {', '.join(missing)}\n")
    if not got:
        print("결과 없음."); return

    names = next((v["class_names"] for v in got.values() if v["class_names"]), None)

    # ---- 주 표 ----
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

    print(f"===== {a.task}: 환자 단위 교차검증 최종 =====")
    print(f"{'구성':38s} {'seed':>4s} {'환자 정확도 (95% CI)':>28s} {'macro-F1 (95% CI)':>26s}")
    for _, r in t.iterrows():
        print(f"{r.label:38s} {int(r.n_seeds):>4d} "
              f"{r.pat_acc:.4f} [{r.pat_acc_lo:.4f}, {r.pat_acc_hi:.4f}]   "
              f"{r.pat_f1:.4f} [{r.pat_f1_lo:.4f}, {r.pat_f1_hi:.4f}]")
    print(f"{'다수 클래스 기준선':38s} {'':>4s} {maj:.4f}")
    if cov and "accuracy" in cov:
        print(f"{'공변량 기준선 (나이+성별+HR)':38s} {'':>4s} {cov['accuracy']:.4f}"
              f"{'':22s}{cov['macro_f1']:.4f}")

    print(f"\n===== 기준선 대비 증분 =====")
    for _, r in t.iterrows():
        d1 = r.pat_acc - maj
        s1 = "위" if r.pat_acc_lo > maj else ("아래" if r.pat_acc_hi < maj else "겹침")
        line = f"  {r.label:38s} 다수클래스 {d1:+.4f} (CI {s1})"
        if cov and "accuracy" in cov:
            d2 = r.pat_acc - cov["accuracy"]
            s2 = "위" if r.pat_acc_lo > cov["accuracy"] else ("아래" if r.pat_acc_hi < cov["accuracy"] else "겹침")
            line += f" | 공변량 {d2:+.4f} (CI {s2})"
        print(line)

    # ---- McNemar ----
    print(f"\n===== 구성 간 짝지은 McNemar (환자 단위) =====")
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
            mc.append({"a": A, "b": B, "n": len(common), "seeds": len(set(k[0] for k in common)), **r})
            flag = "  <== 유의" if r["p_value"] < 0.05 else ""
            print(f"  {LABEL[A]:34s} vs {LABEL[B]:34s} "
                  f"{r['accuracy_a']:.3f}/{r['accuracy_b']:.3f} n={len(common):3d} p={r['p_value']:.4f}{flag}")
    if mc:
        pd.DataFrame(mc).to_csv(a.out / f"{a.task}_mcnemar.csv", index=False)

    # ---- 클래스별 pooled ----
    for m in avail:
        p = got[m]["preds"]
        cn = got[m]["class_names"] or names or [str(i) for i in range(int(p.y_true.max()) + 1)]
        tab = metrics_table(p.y_true.values, p.y_pred.values, cn)
        tab.to_csv(a.out / f"{a.task}_{m}_pooled_patient.csv", index=False)

    # ---- LaTeX ----
    lines = ["\\begin{table}[ht]", "\\centering", "\\begin{threeparttable}",
             f"\\caption{{\\rev{{Patient-level cross-validation for {'Task~I' if a.task=='task1' else 'Task~II'}. "
             "Mean $\\pm$ SD over folds, with 95\\% confidence intervals computed across folds rather "
             "than across segments. Baselines are shown for reference.}}}",
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
              "\\item No patient contributes segments to more than one split in any fold, and no test "
              "segment has an overlapping neighbour in training or validation; both are asserted in "
              "the released code.",
              "\\end{tablenotes}", "\\end{threeparttable}", "\\end{table}"]
    (a.out / f"{a.task}_table.tex").write_text("\n".join(lines))
    print(f"\nLaTeX 표: {a.out / (a.task + '_table.tex')}")
    print(f"결과: {a.out}")


if __name__ == "__main__":
    main()
