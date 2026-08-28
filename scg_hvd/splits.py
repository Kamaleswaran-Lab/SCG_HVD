# 분할을 한 곳에서 결정한다. 학습 코드는 분할 방식을 모르게 해서 "분할만 바꿨다"를 보장한다.
"""
네 가지 분할을 제공한다.

    segment      논문 재현용. 세그먼트 단위 stratified. 환자가 train/val/test에 걸친다.
    patient      환자 단위 단일 분할. 아카이브 `5_*` 실행과 동일한 절차.
    patient_cv   환자 단위 stratified group k-fold. 리비전의 기본값.
    loocv        leave-one-patient-out.

`segment`는 논문 숫자를 재현하기 위해 존재하며 **일반화 성능 추정에 쓰면 안 된다.**
아카이브 코드가 이 분할만 사용했고, 그 결과 Task I 테스트 세그먼트의 98.96%가 학습·검증
세트에 50% 겹치는 이웃 창을 갖게 됐다. `quantify_overlap_leakage()`로 재측정할 수 있다.

검증 세트 구성 원칙. 아카이브 LOOCV(`evaluate.py:122-131`)는 테스트 환자의 라벨을 읽어
같은 라벨 환자만 검증에 넣었고, 그 검증 손실로 조기 종료 지점을 골랐다. 테스트 라벨이
모델 선택에 흘러드는 결함이다. 여기서는 검증 환자를 **남은 학습 풀에서 클래스 균형만 맞춰**
뽑으며, 테스트 환자의 라벨을 어떤 경로로도 참조하지 않는다.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

SPLIT_COLUMN = "split"


# ---------------------------------------------------------------- 도우미

def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """환자당 한 행. 라벨이 환자 내에서 유일하다고 가정하지 않고 최빈값을 쓴다."""
    return (
        df.groupby("patient_id")["label"]
        .agg(lambda s: s.value_counts().idxmax())
        .reset_index()
        .rename(columns={"label": "patient_label"})
    )


def _assign(df, train_ids, val_ids, test_ids):
    out = df.copy()
    out[SPLIT_COLUMN] = pd.NA
    out.loc[out.patient_id.isin(train_ids), SPLIT_COLUMN] = "train"
    out.loc[out.patient_id.isin(val_ids), SPLIT_COLUMN] = "val"
    out.loc[out.patient_id.isin(test_ids), SPLIT_COLUMN] = "test"
    return out


# ---------------------------------------------------------------- 논문 재현

def segment_split(df: pd.DataFrame, test_size=0.1, val_size=0.2, random_state=42) -> pd.DataFrame:
    """논문이 사용한 세그먼트 단위 분할을 그대로 재현한다.

    아카이브 `Task1/1_hvdnet/hvdnet_model.py:314-315` 와 동일한 두 번의 호출이다.
    sklearn 버전에 따라 결과가 달라질 수 있으므로 결과를 CSV로 동결해 쓰는 것을 권한다.
    """
    trainval, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    train, val = train_test_split(
        trainval, test_size=val_size, stratify=trainval["label"], random_state=random_state
    )
    out = df.copy()
    out[SPLIT_COLUMN] = pd.NA
    out.loc[train.index, SPLIT_COLUMN] = "train"
    out.loc[val.index, SPLIT_COLUMN] = "val"
    out.loc[test.index, SPLIT_COLUMN] = "test"
    return out


# ---------------------------------------------------------------- 환자 단위

class SingletonClassError(ValueError):
    """환자가 1명뿐인 클래스가 있어 환자 단위 분할이 정의되지 않을 때."""


def singleton_classes(df: pd.DataFrame) -> dict:
    """환자가 2명 미만인 클래스를 {라벨: [환자…]} 로 돌려준다.

    Task II 의 AS-AR 이 여기 걸린다(환자 CP-05 1명). 이런 클래스는 환자 단위 평가가
    **정의상 불가능하다.** 리뷰어 R2-M3 가 "1명으로 대표되는 클래스는 통상적 test class 로
    둘 수 없다"고 지적한 바로 그 지점이므로, 조용히 넘기지 말고 호출자가 결정하게 한다.
    """
    pt = patient_table(df)
    counts = pt.patient_label.value_counts()
    return {
        lab: sorted(pt[pt.patient_label == lab].patient_id)
        for lab in counts[counts < 2].index
    }


def patient_split(
    df: pd.DataFrame, test_size=0.1, val_size=0.2, random_state=42, singleton_policy="error"
) -> pd.DataFrame:
    """환자 단위 단일 분할. 아카이브 `Task1/5_hvdnet` 과 같은 절차다.

    singleton_policy
        "error"        환자 1명짜리 클래스가 있으면 예외를 낸다(기본값).
        "fix_to_train" 그 환자를 학습에 고정하고 평가에서 제외한다. 제외 사실을 반드시 보고할 것.
    """
    singles = singleton_classes(df)
    fixed: set[str] = set()
    if singles:
        if singleton_policy == "error":
            raise SingletonClassError(
                "환자 단위 분할이 불가능한 클래스가 있다: "
                + ", ".join(f"{k}({len(v)}명: {','.join(v)})" for k, v in singles.items())
                + ". singleton_policy='fix_to_train' 으로 학습 고정하거나 해당 클래스를 제외하라."
            )
        if singleton_policy != "fix_to_train":
            raise ValueError(f"unknown singleton_policy {singleton_policy!r}")
        fixed = {p for ps in singles.values() for p in ps}

    pt = patient_table(df)
    pt = pt[~pt.patient_id.isin(fixed)]

    n_test = int(round(len(pt) * test_size))
    n_classes = pt.patient_label.nunique()
    if n_test < n_classes:
        raise SingletonClassError(
            f"환자 {len(pt)}명에 test_size={test_size} 이면 테스트 환자가 {n_test}명인데 "
            f"클래스가 {n_classes}개다. 클래스마다 최소 1명을 담을 수 없으므로 환자 단위 "
            f"단일 분할이 성립하지 않는다. patient_cv_folds() 나 loocv_folds() 를 쓰거나 "
            f"test_size 를 {n_classes / len(pt):.2f} 이상으로 올려라."
        )

    trainval_ids, test_ids = train_test_split(
        pt.patient_id, test_size=test_size, stratify=pt.patient_label, random_state=random_state
    )
    remain = pt[pt.patient_id.isin(trainval_ids)]
    train_ids, val_ids = train_test_split(
        remain.patient_id, test_size=val_size, stratify=remain.patient_label,
        random_state=random_state,
    )
    # 단일 환자 클래스는 학습에 고정한다. 평가 대상이 아니다.
    return _assign(df, set(train_ids) | fixed, set(val_ids), set(test_ids))


def patient_cv_folds(df: pd.DataFrame, n_splits=5, val_size=0.2, random_state=42):
    """환자 단위 stratified group k-fold. fold 마다 분할 열이 붙은 DataFrame 을 내놓는다.

    검증 환자는 그 fold 의 학습 풀에서만 뽑으며 테스트 환자의 라벨을 참조하지 않는다.
    """
    pt = patient_table(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for k, (tr_idx, te_idx) in enumerate(skf.split(pt.patient_id, pt.patient_label)):
        test_ids = set(pt.iloc[te_idx].patient_id)
        pool = pt.iloc[tr_idx]
        # 클래스당 2명 미만이면 stratify 가 불가능하므로 그 경우만 무작위로 뽑는다.
        strat = pool.patient_label if pool.patient_label.value_counts().min() >= 2 else None
        train_ids, val_ids = train_test_split(
            pool.patient_id, test_size=val_size, stratify=strat, random_state=random_state + k
        )
        yield k, _assign(df, set(train_ids), set(val_ids), test_ids)


def loocv_folds(df: pd.DataFrame, val_size=0.15, random_state=42):
    """leave-one-patient-out. 단일 환자 클래스는 테스트로 내보내지 않고 학습에 고정한다.

    아카이브 하니스와 달리 검증 환자를 테스트 환자의 라벨로 고르지 않는다.
    """
    pt = patient_table(df)
    counts = pt.patient_label.value_counts()
    singleton_labels = set(counts[counts < 2].index)
    fixed = set(pt[pt.patient_label.isin(singleton_labels)].patient_id)

    rng = np.random.default_rng(random_state)
    testable = [p for p in pt.patient_id if p not in fixed]
    for k, test_pid in enumerate(testable):
        pool = pt[(pt.patient_id != test_pid)]
        n_val = max(1, int(round(len(pool) * val_size)))
        # 클래스 균형을 맞춰 뽑되 테스트 환자의 라벨은 보지 않는다.
        val_ids = set()
        for lab, g in pool.groupby("patient_label"):
            cand = [p for p in g.patient_id if p not in fixed]
            if not cand:
                continue
            take = max(1, int(round(len(g) * val_size)))
            val_ids |= set(pd.Series(cand).sample(min(take, len(cand)),
                                                  random_state=random_state + k))
        if not val_ids:
            val_ids = set(rng.choice(pool.patient_id, size=n_val, replace=False))
        train_ids = set(pool.patient_id) - val_ids
        yield k, test_pid, _assign(df, train_ids, val_ids, {test_pid})


# ---------------------------------------------------------------- 진단

def check_patient_disjoint(split_df: pd.DataFrame) -> dict:
    """환자가 두 split 에 걸치지 않는지 확인한다. 환자 단위 분할이면 violations 가 0이어야 한다."""
    per = split_df.groupby("patient_id")[SPLIT_COLUMN].nunique()
    return {
        "n_patients": int(per.size),
        "violations": int((per > 1).sum()),
        "offenders": sorted(per[per > 1].index.tolist()),
    }


def quantify_overlap_leakage(split_df: pd.DataFrame, stride_sec=5.0, against=("train",)) -> dict:
    """테스트 세그먼트 중 학습 쪽에 50% 겹치는 이웃 창을 가진 비율을 센다.

    창 10초 / 이동 5초이므로 시작 시각이 정확히 stride_sec 만큼 떨어진 같은 환자의 세그먼트가
    원신호 5초를 공유한다. 감사 기준값은 Task I 92.63%(train) / 98.96%(train+val) 이다.
    """
    test = split_df[split_df[SPLIT_COLUMN] == "test"]
    ref = split_df[split_df[SPLIT_COLUMN].isin(against)]
    ref_keys = set(zip(ref.patient_id, np.round(ref.start_time_sec, 3)))

    n_with, counts = 0, []
    for pid, t in zip(test.patient_id, test.start_time_sec):
        c = sum(
            (pid, round(t + d, 3)) in ref_keys for d in (-stride_sec, +stride_sec)
        )
        counts.append(c)
        n_with += c > 0
    n = len(test)
    return {
        "n_test": n,
        "n_with_overlapping_neighbour": n_with,
        "fraction": n_with / n if n else float("nan"),
        "neighbour_count_hist": dict(pd.Series(counts).value_counts().sort_index()),
        "against": list(against),
    }
