# Every partitioning decision lives here. The training code never learns which split it was
# handed, which is what lets us claim that two runs differ only in the split.
"""Four ways to partition the segment table.

    segment      Reproduces the published numbers. Stratified over segments, so a patient's
                 windows land in train, val and test alike.
    patient      A single patient-disjoint split, matching the archived `5_*` runs.
    patient_cv   Patient-level stratified group k-fold. The default for the revision.
    loocv        Leave one patient out.

`segment` exists to reproduce the published figures and must not be read as an estimate of
generalization. The archived code used it exclusively, and as a consequence 98.96% of Task I
test segments have a 50%-overlapping neighbour somewhere in training or validation.
`quantify_overlap_leakage()` recomputes that number.

How the validation set is chosen matters here. The archived LOOCV harness
(`evaluate.py:122-131`) read the test patient's label and filled the validation set with
patients carrying that same label, then picked the early-stopping epoch from the resulting
validation loss, which leaks the test label into model selection. Below, validation patients
come from the remaining training pool with class balance as the only criterion, and the test
patient's label is never consulted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

SPLIT_COLUMN = "split"


# ---------------------------------------------------------------- helpers

def patient_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per patient. Takes the modal label rather than assuming a patient has only one."""
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


# ------------------------------------------------------- reproducing the paper

def segment_split(df: pd.DataFrame, test_size=0.1, val_size=0.2, random_state=42) -> pd.DataFrame:
    """The segment-level split the paper used, reproduced call for call.

    Mirrors the two `train_test_split` calls at `Task1/1_hvdnet/hvdnet_model.py:314-315` in
    the archive. The outcome can shift between scikit-learn versions, so freeze the result to
    CSV and load that rather than relying on the seed to be portable.
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


# ---------------------------------------------------------- patient-disjoint

class SingletonClassError(ValueError):
    """Raised when a class has too few patients for a patient-level split to be defined."""


def singleton_classes(df: pd.DataFrame) -> dict:
    """Classes with fewer than two patients, as {label: [patient_id, ...]}.

    Task II's AS-AR falls here: it is one patient, CP-05. Such a class cannot be evaluated at
    the patient level at all, because holding that patient out removes the class from training.
    We surface it instead of proceeding quietly, so the caller has to decide what to do.
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
    """A single patient-disjoint split, following the archived `Task1/5_hvdnet` procedure.

    singleton_policy
        "error"        raise if any class has a single patient (the default).
        "fix_to_train" pin that patient to training and drop the class from evaluation.
                       Whoever chooses this must report the exclusion.
    """
    singles = singleton_classes(df)
    fixed: set[str] = set()
    if singles:
        if singleton_policy == "error":
            raise SingletonClassError(
                "no patient-level split exists for: "
                + ", ".join(f"{k} ({len(v)} patient(s): {','.join(v)})"
                            for k, v in singles.items())
                + ". Pass singleton_policy='fix_to_train' to pin them to training, or drop "
                  "the class."
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
            f"test_size={test_size} over {len(pt)} patients leaves {n_test} test patients "
            f"for {n_classes} classes, so at least one class would be absent from the test "
            f"set and a single patient-level split is not defined. Use patient_cv_folds() or "
            f"loocv_folds(), or raise test_size to at least {n_classes / len(pt):.2f}."
        )

    trainval_ids, test_ids = train_test_split(
        pt.patient_id, test_size=test_size, stratify=pt.patient_label, random_state=random_state
    )
    remain = pt[pt.patient_id.isin(trainval_ids)]
    train_ids, val_ids = train_test_split(
        remain.patient_id, test_size=val_size, stratify=remain.patient_label,
        random_state=random_state,
    )
    # Single-patient classes stay in training; they are not evaluated.
    return _assign(df, set(train_ids) | fixed, set(val_ids), set(test_ids))


def patient_cv_folds(df: pd.DataFrame, n_splits=5, val_size=0.2, random_state=42):
    """Patient-level stratified group k-fold, yielding one split-annotated frame per fold.

    Validation patients come from that fold's training pool only, and the test patients'
    labels are never consulted.
    """
    pt = patient_table(df)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for k, (tr_idx, te_idx) in enumerate(skf.split(pt.patient_id, pt.patient_label)):
        test_ids = set(pt.iloc[te_idx].patient_id)
        pool = pt.iloc[tr_idx]
        # Stratifying needs two patients per class; fall back to a plain draw when it cannot.
        strat = pool.patient_label if pool.patient_label.value_counts().min() >= 2 else None
        train_ids, val_ids = train_test_split(
            pool.patient_id, test_size=val_size, stratify=strat, random_state=random_state + k
        )
        yield k, _assign(df, set(train_ids), set(val_ids), test_ids)


def loocv_folds(df: pd.DataFrame, val_size=0.15, random_state=42):
    """Leave one patient out. Single-patient classes stay in training and are never tested.

    Unlike the archived harness, the validation patients are not chosen using the test
    patient's label.
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
        # Draw with class balance, without looking at the held-out patient's label.
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


# ---------------------------------------------------------------- diagnostics

def check_patient_disjoint(split_df: pd.DataFrame) -> dict:
    """Check that no patient spans two splits. For a patient-level split, violations is 0."""
    per = split_df.groupby("patient_id")[SPLIT_COLUMN].nunique()
    return {
        "n_patients": int(per.size),
        "violations": int((per > 1).sum()),
        "offenders": sorted(per[per > 1].index.tolist()),
    }


def quantify_overlap_leakage(split_df: pd.DataFrame, stride_sec=5.0, against=("train",)) -> dict:
    """Fraction of test segments that have a 50%-overlapping neighbour on the training side.

    Windows are 10 s on a 5 s stride, so two segments of the same patient whose start times
    differ by exactly `stride_sec` share five seconds of raw signal. Reference values from our
    audit of Task I: 92.63% against training, 98.96% against training and validation together.
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
