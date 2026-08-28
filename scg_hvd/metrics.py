# 지표를 원본 예측에서 직접 계산한다. 아카이브의 정규화-되곱 경로를 쓰지 않는다.
"""
왜 새로 쓰는가.

아카이브는 혼동행렬을 `normalize='true'` 로 정규화해 소수 4자리로 저장한 뒤,
`results_selection.ipynb` 가 그것을 하드코딩된 support 로 되곱아 절대 개수를 복원했다.
그 왕복에서 손실이 생겨 원고 Table 2·3 의 120셀 중 **17셀이 소수 둘째 자리에서 어긋난다**
(전부 0.015 퍼센트포인트 이하). 예를 들어 Task II 융합 실행의 AS-TR 행은 합이 1.0001 이다.

여기서는 `y_true` / `y_pred` 만 받아 정수 혼동행렬에서 바로 계산한다. 예측 자체도
`predictions.csv` 로 저장해 사후 재계산과 짝지은 검정이 가능하게 한다.

Overall 행은 micro-average 다. Referee 2 가 major 5 에서 지적했듯 이 정의에서는
Se·Sp·Ac·F1 이 서로 대수적으로 종속이며(`overall_metrics_are_dependent` 참조),
네 값을 모두 보고하는 것은 하나의 숫자를 네 번 적는 것에 가깝다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def per_class_metrics(y_true, y_pred, class_names) -> pd.DataFrame:
    """one-vs-rest 기준 클래스별 Se/Sp/Ac/F1 과 TP/FP/FN/TN."""
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    total = cm.sum()
    rows = []
    for i, name in enumerate(class_names):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(total - tp - fn - fp)
        rows.append({
            "class_name": name,
            "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "accuracy": (tp + tn) / total if total else np.nan,
            "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
            "support": tp + fn, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    return pd.DataFrame(rows)


def overall_metrics(y_true, y_pred, n_classes) -> dict:
    """micro-average. 원고의 Overall 행과 같은 정의다."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    total = cm.sum()
    tp = int(np.trace(cm))
    fn = fp = total - tp
    tn = total * n_classes - tp - fp - fn
    return {
        "class_name": "Overall",
        "sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "specificity": tn / (tn + fp) if tn + fp else np.nan,
        "accuracy": (tp + tn) / (tp + fp + fn + tn),
        "f1_score": 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan,
        "support": int(total), "TP": tp, "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


def macro_metrics(y_true, y_pred, class_names) -> dict:
    pc = per_class_metrics(y_true, y_pred, class_names)
    return {
        "macro_sensitivity": float(pc.sensitivity.mean()),
        "macro_specificity": float(pc.specificity.mean()),
        "macro_f1": float(pc.f1_score.mean()),
        "balanced_accuracy": float(pc.sensitivity.mean()),
    }


def metrics_table(y_true, y_pred, class_names) -> pd.DataFrame:
    """클래스별 행 + Overall 행. 원고 Table 2/3 과 같은 모양이다."""
    pc = per_class_metrics(y_true, y_pred, class_names)
    ov = pd.DataFrame([overall_metrics(y_true, y_pred, len(class_names))])
    return pd.concat([pc, ov], ignore_index=True)


def majority_baseline(y_true, class_names) -> dict:
    """항상 최빈 클래스를 답하는 기준선. 이게 없으면 환자 단위 숫자를 해석할 수 없다."""
    y_true = np.asarray(y_true)
    maj = int(pd.Series(y_true).value_counts().idxmax())
    y_pred = np.full_like(y_true, maj)
    ov = overall_metrics(y_true, y_pred, len(class_names))
    return {
        "majority_class": class_names[maj],
        "accuracy": ov["accuracy"],
        "micro_sensitivity": ov["sensitivity"],
        **macro_metrics(y_true, y_pred, class_names),
    }


def aggregate_to_patient(df_pred: pd.DataFrame, n_classes: int, how="mean_prob") -> pd.DataFrame:
    """세그먼트 예측을 환자 단위로 집계한다.

    df_pred 는 patient_id, y_true, y_pred 와 prob_0..prob_{n-1} 열을 가진다.
    임상 진단은 환자 단위로 내려지므로(R1-m4) 환자 단위 표를 반드시 함께 낸다.
    """
    prob_cols = [f"prob_{i}" for i in range(n_classes)]
    rows = []
    for pid, g in df_pred.groupby("patient_id"):
        if how == "mean_prob":
            pred = int(np.argmax(g[prob_cols].mean(axis=0).values))
        elif how == "majority_vote":
            pred = int(g.y_pred.value_counts().idxmax())
        else:
            raise ValueError(how)
        rows.append({"patient_id": pid, "y_true": int(g.y_true.iloc[0]),
                     "y_pred": pred, "n_segments": len(g)})
    return pd.DataFrame(rows)


def mcnemar_paired(y_true, pred_a, pred_b) -> dict:
    """두 구성의 짝지은 비교. 환자 단위 예측에 쓴다.

    Referee 2 major 5 가 요구한 대로 짝지은 검정이며, 세그먼트를 독립으로 보지 않는다.
    표본이 작으므로 이항 정확검정을 쓴다.
    """
    from scipy import stats
    y_true = np.asarray(y_true); a = np.asarray(pred_a); b = np.asarray(pred_b)
    a_ok, b_ok = a == y_true, b == y_true
    n01 = int((~a_ok & b_ok).sum())   # a 틀리고 b 맞음
    n10 = int((a_ok & ~b_ok).sum())   # a 맞고 b 틀림
    n = n01 + n10
    p = 1.0 if n == 0 else float(stats.binomtest(n10, n, 0.5).pvalue)
    return {"n_a_only_correct": n10, "n_b_only_correct": n01,
            "n_discordant": n, "p_value": p,
            "accuracy_a": float(a_ok.mean()), "accuracy_b": float(b_ok.mean())}


def overall_metrics_are_dependent(n_classes: int, sensitivity: float) -> dict:
    """micro-average 에서 Sp·Ac 가 Se 로부터 대수적으로 결정됨을 보인다 (R2-M5).

    C 클래스 micro-average 에서 FP = FN = N(1-Se) 이므로
        Sp = 1 - (1-Se)/(C-1),   Ac = 1 - 2(1-Se)/C,   F1 = Se
    가 성립한다.
    """
    c, se = n_classes, sensitivity
    return {"sensitivity": se,
            "specificity_implied": 1 - (1 - se) / (c - 1),
            "accuracy_implied": 1 - 2 * (1 - se) / c,
            "f1_implied": se}
