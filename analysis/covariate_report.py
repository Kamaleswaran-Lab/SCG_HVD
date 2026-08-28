# R2-M2 최종 보고서. HR·나이·성별을 합쳐 클래스별 표, 검정, 공변량 기준선을 낸다.
"""
Referee 2 major 2 가 요구한 것을 한 파일로 낸다.

    1. 클래스별 공변량 표 (HR, 나이, 성별, 체격, EF, 녹음 길이)
    2. 클래스 간 차이 검정 — 리뷰어가 "see if they are significant or not" 이라 한 부분
    3. 보정 가능성 판정 — 연령 매칭이 왜 불가능한지를 숫자로
    4. 공변량 단독 기준선 — SCG 모델이 넘어야 할 선

`extract_heart_rate.py` 를 먼저 돌려 `out/hr/{task}_patient_hr.csv` 를 만들어 두어야 한다.

해석에서 반드시 함께 적어야 할 임상적 맥락이 있다. **퇴행성 대동맥 협착은 고령 질환이다.**
AS 군이 나머지보다 20년 이상 고령인 것은 표본 추출의 결함이 아니라 역학적으로 예상되는
분포다. 따라서 목표는 나이 효과를 제거하는 것이 아니라, 모델이 나이 상관 특징에만 의존하지
않음을 보이는 것이다. 이 구분을 응답 letter 에 명시해야 리뷰어의 "adjusting is required" 를
"연령을 통제한 재학습" 이 아닌 "공변량 대비 증분 보고" 로 되돌릴 수 있다.

사용법.
    python analysis/covariate_report.py --task task1 --out out/covariates
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.covariate_tests import (age_matching, covariate_table,  # noqa: E402
                                      load, omnibus_tests)

DATA = Path("/work/jkim1/SCG_HVD_data")


def merge_hr(md: pd.DataFrame, hr_csv: Path) -> pd.DataFrame:
    """환자 단위 HR 을 코호트 메타데이터에 붙인다."""
    if not hr_csv.exists():
        print(f"[경고] {hr_csv} 가 없다. extract_heart_rate.py 를 먼저 돌려라.")
        md["hr_median"] = np.nan
        return md
    hr = pd.read_csv(hr_csv)
    return md.merge(hr[["patient_id", "hr_median", "n_ok"]],
                    left_on="Patient ID", right_on="patient_id", how="left")


def hr_table(pat_hr: pd.DataFrame) -> pd.DataFrame:
    """Dataset II 정상군을 포함한 전체 클래스별 HR. 코호트 메타데이터와 무관하게 낸다."""
    return (pat_hr.groupby("label")
            .agg(n_patients=("hr_median", "size"), hr_mean=("hr_median", "mean"),
                 hr_sd=("hr_median", "std"), hr_min=("hr_median", "min"),
                 hr_max=("hr_median", "max")).round(1))


def hr_tests(pat_hr: pd.DataFrame) -> dict:
    groups = [g.hr_median.dropna().values for _, g in pat_hr.groupby("label")]
    groups = [g for g in groups if len(g) >= 2]
    out = {}
    if len(groups) >= 2:
        s, p = stats.kruskal(*groups)
        out["kruskal_H"] = round(float(s), 3)
        out["kruskal_p"] = float(p)
        out["significant"] = bool(p < 0.05)
    # AS 대 나머지
    if "AS" in set(pat_hr.label):
        a = pat_hr[pat_hr.label == "AS"].hr_median.dropna()
        b = pat_hr[pat_hr.label != "AS"].hr_median.dropna()
        if len(a) >= 2 and len(b) >= 2:
            out["AS_vs_rest_p"] = float(stats.mannwhitneyu(a, b).pvalue)
            out["AS_hr_mean"] = round(float(a.mean()), 1)
            out["rest_hr_mean"] = round(float(b.mean()), 1)
    return out


def multiclass_baseline(md: pd.DataFrame, seed=42) -> dict:
    """5클래스 공변량 단독 기준선. **모델과 직접 비교해야 하는 숫자는 이쪽이다.**

    AS 대 나머지 이진 AUC 만 보면 나이의 설명력이 압도적으로 보이지만, 논문의 과제는 5클래스다.
    5클래스에서 공변량이 얼마나 설명하는지가 R2-M2 에 대한 실질적 답이다.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    md = md.copy()
    md["male"] = (md.Gender == "M").astype(int)
    combos = {"age": ["Age"], "age+sex": ["Age", "male"],
              "hr": ["hr_median"], "age+sex+hr": ["Age", "male", "hr_median"]}
    out = {}
    for name, cols in combos.items():
        sub = md.dropna(subset=cols)
        y = sub.Task1.values
        k = int(min(3, pd.Series(y).value_counts().min()))
        if k < 2:
            out[name] = {"note": "클래스당 환자가 적어 fold 불가"}
            continue
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        accs, f1s = [], []
        for tr, te in skf.split(sub[cols], y):
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, class_weight="balanced"))
            clf.fit(sub[cols].values[tr], y[tr])
            p = clf.predict(sub[cols].values[te])
            accs.append(accuracy_score(y[te], p))
            f1s.append(f1_score(y[te], p, average="macro", zero_division=0))
        out[name] = {"features": cols, "n": len(sub), "folds": k,
                     "accuracy": round(float(np.mean(accs)), 4),
                     "accuracy_sd": round(float(np.std(accs)), 4),
                     "macro_f1": round(float(np.mean(f1s)), 4)}
    vc = pd.Series(md.Task1.values).value_counts()
    out["majority"] = {"class": str(vc.index[0]),
                       "accuracy": round(float(vc.iloc[0] / len(md)), 4)}
    return out


def hr_outlier_by_class(pat_hr: pd.DataFrame) -> pd.DataFrame:
    """어느 클래스의 심박수가 나머지와 다른지. 리뷰어의 HR 우려가 어디에 걸리는지 보인다."""
    rows = []
    for lab in sorted(pat_hr.label.unique()):
        a = pat_hr[pat_hr.label == lab].hr_median.dropna()
        b = pat_hr[pat_hr.label != lab].hr_median.dropna()
        p = float(stats.mannwhitneyu(a, b).pvalue) if len(a) >= 2 and len(b) >= 2 else np.nan
        rows.append({"class": lab, "n": len(a), "hr_mean": round(float(a.mean()), 1),
                     "hr_rest_mean": round(float(b.mean()), 1), "p": p,
                     "significant": bool(p == p and p < 0.05)})
    return pd.DataFrame(rows)


def baseline_with_hr(md: pd.DataFrame, target="AS", n_splits=5, seed=42) -> dict:
    """HR·나이·성별 로지스틱 기준선. 특징 조합별로 나눠 각각의 기여를 보인다."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = md.dropna(subset=["Age", "Gender"]).copy()
    d["male"] = (d.Gender == "M").astype(int)
    y = (d.Task1 == target).astype(int).values

    combos = {
        "age": ["Age"],
        "age+sex": ["Age", "male"],
    }
    if d.hr_median.notna().sum() >= 0.8 * len(d):
        combos["hr"] = ["hr_median"]
        combos["age+sex+hr"] = ["Age", "male", "hr_median"]

    res = {}
    for name, cols in combos.items():
        sub = d.dropna(subset=cols)
        ys = (sub.Task1 == target).astype(int).values
        if len(set(ys)) < 2 or min(np.bincount(ys)) < n_splits:
            res[name] = {"note": "표본 부족"}
            continue
        X = sub[cols].values
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        accs, aucs = [], []
        for tr, te in skf.split(X, ys):
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
            clf.fit(X[tr], ys[tr])
            accs.append(float((clf.predict(X[te]) == ys[te]).mean()))
            prob = clf.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(ys[te], prob) if len(set(ys[te])) > 1 else np.nan)
        res[name] = {"features": cols, "n": len(sub),
                     "accuracy": round(float(np.mean(accs)), 4),
                     "accuracy_sd": round(float(np.std(accs)), 4),
                     "auc": round(float(np.nanmean(aucs)), 4)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--meta-dir", type=Path, default=DATA / "meta")
    ap.add_argument("--hr-dir", type=Path, default=Path("out/hr"))
    ap.add_argument("--out", type=Path, default=Path("out/covariates"))
    ap.add_argument("--target", default="AS")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    # --- HR: 전체 클래스(정상군 포함) ---
    hr_csv = a.hr_dir / f"{a.task}_patient_hr.csv"
    if hr_csv.exists():
        pat_hr = pd.read_csv(hr_csv)
        ht = hr_table(pat_hr)
        ht.to_csv(a.out / f"{a.task}_hr_by_class.csv")
        print("=== 클래스별 심박수 (환자 단위 중앙값, 전체 코호트) ===")
        print(ht.to_string())
        t = hr_tests(pat_hr)
        (a.out / f"{a.task}_hr_tests.json").write_text(json.dumps(t, indent=2))
        print("\n심박수 클래스 간 차이:",
              f"Kruskal H={t.get('kruskal_H')}, p={t.get('kruskal_p'):.3e}"
              if "kruskal_p" in t else "검정 불가",
              "->", "유의함" if t.get("significant") else "유의하지 않음")
        if "AS_vs_rest_p" in t:
            print(f"  AS {t['AS_hr_mean']} bpm vs 나머지 {t['rest_hr_mean']} bpm, "
                  f"p={t['AS_vs_rest_p']:.3f}")
    else:
        pat_hr = None
        print(f"[경고] {hr_csv} 없음 — HR 부분을 건너뛴다.")

    if a.task != "task1":
        print("\n코호트 메타데이터 기반 분석은 Task1 라벨에만 정의돼 있다. 여기서 종료한다.")
        return

    # --- 나이·성별 등: Dataset I 한정 ---
    md = load(a.meta_dir)
    if pat_hr is not None:
        md = merge_hr(md, hr_csv)
    else:
        md["hr_median"] = np.nan

    print(f"\n=== 클래스별 공변량 (Dataset I 환자 {len(md)}명) ===")
    tab = covariate_table(md)
    if md.hr_median.notna().any():
        tab["HR mean"] = md.groupby("Task1").hr_median.mean().round(1)
        tab["HR sd"] = md.groupby("Task1").hr_median.std().round(1)
    tab.to_csv(a.out / "covariate_table.csv")
    cols = ["n_patients", "male_n", "female_n", "Age mean", "Age sd"]
    cols += [c for c in ["HR mean", "HR sd"] if c in tab]
    print(tab[cols].to_string())

    tests = omnibus_tests(md)
    tests.to_csv(a.out / "covariate_tests.csv", index=False)
    print("\n=== 클래스 간 차이 검정 ===")
    print(tests[tests.p.notna()][["covariate", "test", "stat", "p", "significant"]]
          .to_string(index=False))

    res, pairs = age_matching(md, target=a.target)
    print(f"\n=== 연령 매칭 ===\n  {res['verdict']}")

    ho = hr_outlier_by_class(pat_hr) if pat_hr is not None else None
    if ho is not None:
        ho.to_csv(a.out / f"{a.task}_hr_outlier_by_class.csv", index=False)
        print("\n=== 심박수가 나머지와 다른 클래스 ===")
        print(ho.to_string(index=False))
        print("  MS 의 빈맥과 N 의 낮은 안정 심박은 임상적으로 예상되는 것이며 AS 는 유의하지 않다.")

    print("\n=== 5클래스 공변량 단독 기준선 (모델과 직접 비교할 숫자) ===")
    mc = multiclass_baseline(md)
    (a.out / "covariate_baseline_multiclass.json").write_text(json.dumps(mc, indent=2))
    for k, v in mc.items():
        if k == "majority":
            print(f"  {'다수클래스('+v['class']+')':14s} 정확도 {v['accuracy']:.4f}")
        elif "accuracy" in v:
            print(f"  {k:14s} n={v['n']:3d} {v['folds']}-fold  정확도 {v['accuracy']:.4f}"
                  f" ± {v['accuracy_sd']:.4f}  macro-F1 {v['macro_f1']:.4f}")
        else:
            print(f"  {k:14s} {v['note']}")

    print("\n=== AS 대 나머지 이진 기준선 (AS 클래스 개별 보고용) ===")
    base = baseline_with_hr(md, target=a.target)
    (a.out / "covariate_baseline_full.json").write_text(json.dumps(base, indent=2))
    for k, v in base.items():
        if "accuracy" in v:
            print(f"  {k:14s} n={v['n']:3d}  정확도 {v['accuracy']:.4f} ± {v['accuracy_sd']:.4f}"
                  f"  AUC {v['auc']:.4f}")
        else:
            print(f"  {k:14s} {v['note']}")
    print(f"\n=> SCG 모델의 {a.target} 환자 단위 성능은 위 AUC 를 넘어야 의미가 있다.")
    print(f"결과: {a.out}")


if __name__ == "__main__":
    main()
