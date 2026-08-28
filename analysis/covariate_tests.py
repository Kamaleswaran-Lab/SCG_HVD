# R2-M2 대응. 클래스별 공변량 차이를 검정하고, 연령 매칭이 가능한지 판정한다.
"""
Referee 2 major comment 2 는 이렇게 요구한다.

    "The authors need to also describe how they ensured that the observed changes in SCG are
     not related to co-variates such as HR, age, and sex. [...] Please report the co-variate
     differences in each HVD class and see if they are significant or not. If so, adjusting
     is required."

이 스크립트가 내놓는 것은 세 가지다.

1. 클래스별 공변량 표 (나이·성별·체격·EF·판막 지표). 리뷰어가 요구한 보고 그 자체다.
2. 연령 매칭 가능성 판정. 매칭이 성립하면 매칭 표본을 내놓고, 성립하지 않으면 왜 그런지를
   숫자로 보인다.
3. **공변량 단독 기준선.** 나이·성별만으로 분류기를 학습해 얻는 성능이다. 매칭으로 보정할 수
   없을 때, "SCG 모델이 공변량 너머에 무엇을 더하는가"를 정량화하는 유일한 방법이다.
   SCG 모델의 환자 단위 성능을 이 기준선과 나란히 보고해야 한다.

주의. Dataset II 의 정상군(`sub_*`)은 `df_metadata.csv` 에 없어 나이를 모른다. 따라서 모든
공변량 분석은 Dataset I 환자로 한정되며, 그 한계를 결과에 명시한다.

사용법.
    python analysis/covariate_tests.py --meta-dir /work/jkim1/SCG_HVD_data/meta --out out/covariates
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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
    """클래스별 공변량 요약. 결측 수를 함께 보고한다."""
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
    """클래스 간 차이 검정. 연속형은 Kruskal-Wallis, 성별은 카이제곱."""
    out = []
    for c in CONTINUOUS + ["duration_sec"]:
        if c not in md:
            continue
        groups = [g[c].dropna().values for _, g in md.groupby("Task1")]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            out.append({"covariate": c, "test": "Kruskal-Wallis", "stat": np.nan,
                        "p": np.nan, "note": "결측이 많아 검정 불가"})
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
    """target 클래스와 나머지를 1:1 최근접 연령 매칭한다. 성립 여부와 표본을 함께 돌려준다."""
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
    # 판정. 5클래스 재학습에 쓰려면 클래스당 최소 몇 명은 있어야 한다.
    res["feasible_for_multiclass"] = bool(len(pairs_df) >= 15)
    res["verdict"] = (
        "매칭 표본으로 재학습 가능" if res["feasible_for_multiclass"]
        else f"매칭 불가. {target} 환자 {len(a)}명 중 대조군 연령 상한({int(b.Age.max())}세) "
             f"이하가 {len(a_ov)}명뿐이라 표본이 {2*len(pairs_df)}명에 그친다."
    )
    return res, pairs_df


def covariate_only_baseline(md: pd.DataFrame, target="AS", n_splits=5, seed=42):
    """나이·성별만으로 target 을 분류한다. SCG 모델이 넘어야 할 하한선이다."""
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

    # 나이 임계값 단독 (설명용, 전체 데이터에 적합시킨 낙관적 상한)
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
    ap.add_argument("--meta-dir", type=Path, default=Path("/work/jkim1/SCG_HVD_data/meta"))
    ap.add_argument("--out", type=Path, default=Path("out/covariates"))
    ap.add_argument("--target", default="AS")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    md = load(a.meta_dir)
    print(f"Dataset I Task1 환자 {len(md)}명. Dataset II 정상군은 메타데이터가 없어 제외된다.\n")

    tab = covariate_table(md)
    tab.to_csv(a.out / "covariate_table.csv")
    print("=== 클래스별 공변량 ===")
    print(tab[["n_patients", "male_n", "female_n", "Age mean", "Age sd",
               "Ejection fraction (%) mean"]].to_string(), "\n")

    tests = omnibus_tests(md)
    tests.to_csv(a.out / "covariate_tests.csv", index=False)
    print("=== 클래스 간 차이 검정 ===")
    print(tests[["covariate", "test", "stat", "p", "significant"]].to_string(index=False), "\n")

    res, pairs = age_matching(md, target=a.target)
    pairs.to_csv(a.out / "age_matched_pairs.csv", index=False)
    (a.out / "age_matching.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print("=== 연령 매칭 판정 ===")
    print(f"  {a.target} 나이 {res['target_age_range']}, 대조군 {res['control_age_range']}")
    print(f"  겹침 구간 {res['overlap_range']}, 매칭 {res['n_pairs']}쌍 ({res['n_matched_total']}명)")
    print(f"  판정: {res['verdict']}\n")

    base = covariate_only_baseline(md, target=a.target)
    (a.out / "covariate_baseline.json").write_text(json.dumps(base, indent=2, ensure_ascii=False))
    print("=== 공변량 단독 기준선 (나이+성별 로지스틱, 환자 단위 5-fold) ===")
    print(f"  정확도 {base['accuracy_mean']:.4f} ± {base['accuracy_sd']:.4f} | "
          f"민감도 {base['sensitivity_mean']:.4f} | 특이도 {base['specificity_mean']:.4f} | "
          f"AUC {base['auc_mean']:.4f}")
    t = base["age_threshold_only"]
    print(f"  나이 임계값 단독(>= {t['threshold']}세): 정확도 {t['accuracy']:.4f}, "
          f"민감도 {t['sensitivity']:.4f}, 특이도 {t['specificity']:.4f}")
    print(f"\n=> SCG 모델의 {a.target} 성능은 이 기준선과 나란히 보고해야 한다.")
    print(f"결과 저장: {a.out}")


if __name__ == "__main__":
    main()
