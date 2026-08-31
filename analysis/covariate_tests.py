# Tests covariate differences between classes and decides whether age matching is feasible.
"""
Referee 2's second major comment asks for this:

    "The authors need to also describe how they ensured that the observed changes in SCG are
     not related to co-variates such as HR, age, and sex. [...] Please report the co-variate
     differences in each HVD class and see if they are significant or not. If so, adjusting
     is required."

Three things come out of it.

1. Covariates by class: age, sex, body habitus, ejection fraction, valve measurements. This is
   the reporting the reviewer asked for, verbatim.
2. A feasibility verdict on age matching. If matching works, the matched sample is written
   out; if it does not, the numbers showing why are.
3. A covariates-only baseline: what a classifier trained on age and sex alone achieves. When
   the confound cannot be matched away, this is the only way left to quantify what the SCG
   model adds on top of it, and the model's patient-level performance has to be read against
   it.

One caveat that limits all of this. The Dataset II controls (`sub_*`) are absent from
`df_metadata.csv`, so their age is unknown. Every covariate analysis here is therefore
restricted to Dataset I patients, and the output says so.

Usage.
    python analysis/covariate_tests.py --out out/covariates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.paths import data_root  # noqa: E402

CONTINUOUS = [
    "Age", "Height (cm)", "Weight (kg)", "Ejection fraction (%)",
    "Aortic valve area (cm2)", "Aortic valve area Mean gradient(mmHg)",
    "Aortic valve area Peak velocity (m/sec)",
]


def load(meta_dir: Path):
    md = pd.read_csv(meta_dir / "df_metadata.csv")
    md = md[(md.Task1 != "Exclude") & md.Task1.notna()].copy()
    md["duration_sec"] = pd.to_timedelta(md["Duration"]).dt.total_seconds()
    return md


def covariate_table(md: pd.DataFrame) -> pd.DataFrame:
    """Covariates summarised by class, with the missing count for each."""
    rows = []
    for lab, g in md.groupby("Task1"):
        row = {"class": lab, "n_patients": len(g),
               "male_n": int((g.Gender == "M").sum()),
               "female_n": int((g.Gender == "F").sum())}
        for c in CONTINUOUS + ["duration_sec"]:
            if c not in g:
                continue
            v = g[c].dropna()
            row[f"{c} mean"] = round(v.mean(), 1) if len(v) else np.nan
            row[f"{c} sd"] = round(v.std(), 1) if len(v) > 1 else np.nan
            row[f"{c} missing"] = int(g[c].isna().sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("class")


def omnibus_tests(md: pd.DataFrame) -> pd.DataFrame:
    """Test for between-class differences: Kruskal-Wallis for continuous, chi-square for sex."""
    out = []
    for c in CONTINUOUS + ["duration_sec"]:
        if c not in md:
            continue
        groups = [g[c].dropna().values for _, g in md.groupby("Task1")]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            out.append({"covariate": c, "test": "Kruskal-Wallis", "stat": np.nan,
                        "p": np.nan, "note": "too much missing data to test"})
            continue
        s, p = stats.kruskal(*groups)
        out.append({"covariate": c, "test": "Kruskal-Wallis", "stat": round(s, 3),
                    "p": p, "note": ""})
    ct = pd.crosstab(md.Task1, md.Gender)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    out.append({"covariate": "Gender", "test": "chi-square", "stat": round(chi2, 3),
                "p": p, "note": ""})
    df = pd.DataFrame(out)
    df["significant"] = df.p < 0.05
    return df


def age_matching(md: pd.DataFrame, target="AS", caliper=3.0):
    """1:1 nearest-age matching of the target class against the rest.

    Returns both the verdict and the matched sample, so a caller can see why it failed.
    """
    a = md[md.Task1 == target]
    b = md[md.Task1 != target]
    lo, hi = max(a.Age.min(), b.Age.min()), min(a.Age.max(), b.Age.max())
    a_ov, b_ov = a[(a.Age >= lo) & (a.Age <= hi)], b[(b.Age >= lo) & (b.Age <= hi)]

    pool, pairs = b_ov.copy(), []
    for _, r in a_ov.sort_values("Age").iterrows():
        if pool.empty:
            break
        d = (pool.Age - r.Age).abs()
        if d.min() <= caliper:
            j = d.idxmin()
            pairs.append({"case_id": r["Patient ID"], "case_age": int(r.Age),
                          "ctrl_id": pool.loc[j, "Patient ID"],
                          "ctrl_age": int(pool.loc[j, "Age"]),
                          "ctrl_class": pool.loc[j, "Task1"]})
            pool = pool.drop(j)

    pairs_df = pd.DataFrame(pairs)
    res = {
        "target": target, "caliper_years": caliper,
        "n_target_total": len(a), "n_control_total": len(b),
        "target_age_range": [int(a.Age.min()), int(a.Age.max())],
        "control_age_range": [int(b.Age.min()), int(b.Age.max())],
        "overlap_range": [int(lo), int(hi)],
        "n_target_in_overlap": len(a_ov), "n_control_in_overlap": len(b_ov),
        "n_pairs": len(pairs_df),
        "n_matched_total": 2 * len(pairs_df),
    }
    if len(pairs_df):
        res["matched_age_p"] = float(
            stats.mannwhitneyu(pairs_df.case_age, pairs_df.ctrl_age).pvalue
        )
        res["control_class_mix"] = pairs_df.ctrl_class.value_counts().to_dict()
    # Feasibility: retraining on five classes needs a workable number of patients per class.
    res["feasible_for_multiclass"] = bool(len(pairs_df) >= 15)
    res["verdict"] = (
        "matched sample is large enough to retrain on" if res["feasible_for_multiclass"]
        else f"matching is not feasible: of {len(a)} {target} patients only {len(a_ov)} fall at "
             f"or below the oldest control ({int(b.Age.max())} years), leaving a matched sample "
             f"of {2*len(pairs_df)}."
    )
    return res, pairs_df


def covariate_only_baseline(md: pd.DataFrame, target="AS", n_splits=5, seed=42):
    """Classify the target from age and sex alone: the floor the SCG model has to clear."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    X = pd.DataFrame({"age": md.Age.values, "male": (md.Gender == "M").astype(int).values})
    y = (md.Task1 == target).astype(int).values

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs, ses, sps, aucs = [], [], [], []
    for tr, te in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000).fit(X.iloc[tr], y[tr])
        p = clf.predict(X.iloc[te])
        prob = clf.predict_proba(X.iloc[te])[:, 1]
        tp = int(((p == 1) & (y[te] == 1)).sum()); fn = int(((p == 0) & (y[te] == 1)).sum())
        tn = int(((p == 0) & (y[te] == 0)).sum()); fp = int(((p == 1) & (y[te] == 0)).sum())
        accs.append((tp + tn) / len(te))
        ses.append(tp / (tp + fn) if tp + fn else np.nan)
        sps.append(tn / (tn + fp) if tn + fp else np.nan)
        aucs.append(roc_auc_score(y[te], prob) if len(set(y[te])) > 1 else np.nan)

    # An age threshold on its own. Fitted on all the data, so read it as an optimistic ceiling
    # for illustration rather than as a held-out result.
    best = max(
        ({"threshold": int(t),
          "accuracy": float(((md.Age >= t).values == (y == 1)).mean()),
          "sensitivity": float(((md.Age >= t).values & (y == 1)).sum() / (y == 1).sum()),
          "specificity": float(((md.Age < t).values & (y == 0)).sum() / (y == 0).sum())}
         for t in range(int(md.Age.min()), int(md.Age.max()) + 1)),
        key=lambda d: d["accuracy"],
    )
    return {
        "target": target,
        "features": ["age", "sex"],
        "cv": f"stratified {n_splits}-fold (patient level)",
        "accuracy_mean": round(float(np.nanmean(accs)), 4),
        "accuracy_sd": round(float(np.nanstd(accs)), 4),
        "sensitivity_mean": round(float(np.nanmean(ses)), 4),
        "specificity_mean": round(float(np.nanmean(sps)), 4),
        "auc_mean": round(float(np.nanmean(aucs)), 4),
        "age_threshold_only": best,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta-dir", type=Path, default=None,
                    help="defaults to $SCG_HVD_DATA/meta")
    ap.add_argument("--out", type=Path, default=Path("out/covariates"))
    ap.add_argument("--target", default="AS")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    meta_dir = a.meta_dir or (data_root() / "meta")
    md = load(meta_dir)
    print(f"{len(md)} Dataset I Task1 patients. The Dataset II controls carry no metadata and "
          f"are excluded.\n")

    tab = covariate_table(md)
    tab.to_csv(a.out / "covariate_table.csv")
    print("=== covariates by class ===")
    print(tab[["n_patients", "male_n", "female_n", "Age mean", "Age sd",
               "Ejection fraction (%) mean"]].to_string(), "\n")

    tests = omnibus_tests(md)
    tests.to_csv(a.out / "covariate_tests.csv", index=False)
    print("=== between-class tests ===")
    print(tests[["covariate", "test", "stat", "p", "significant"]].to_string(index=False), "\n")

    res, pairs = age_matching(md, target=a.target)
    pairs.to_csv(a.out / "age_matched_pairs.csv", index=False)
    (a.out / "age_matching.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("=== age matching feasibility ===")
    print(f"  {a.target} age {res['target_age_range']}, controls {res['control_age_range']}")
    print(f"  overlap {res['overlap_range']}, {res['n_pairs']} pairs "
          f"({res['n_matched_total']} patients)")
    print(f"  verdict: {res['verdict']}\n")

    base = covariate_only_baseline(md, target=a.target)
    (a.out / "covariate_baseline.json").write_text(json.dumps(base, indent=2, ensure_ascii=False))
    print("=== covariates-only baseline (age+sex logistic, patient-level 5-fold) ===")
    print(f"  accuracy {base['accuracy_mean']:.4f} +/- {base['accuracy_sd']:.4f} | "
          f"sensitivity {base['sensitivity_mean']:.4f} | "
          f"specificity {base['specificity_mean']:.4f} | "
          f"AUC {base['auc_mean']:.4f}")
    t = base["age_threshold_only"]
    print(f"  age threshold alone (>= {t['threshold']}): accuracy {t['accuracy']:.4f}, "
          f"sensitivity {t['sensitivity']:.4f}, specificity {t['specificity']:.4f}")
    print(f"\n=> Report the SCG model's {a.target} performance next to this baseline.")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
