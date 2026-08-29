# R2-m2 대응. 두 데이터셋의 축 라벨이 같은 물리 축을 가리키는지 신호로 확인한다.
"""
Referee 2 minor 2.

    "Please clarify whether the three accelerometer axes were consistently aligned across subjects
     and datasets. Variation in sensor orientation could substantially affect axis-specific SCG
     morphology."

두 데이터셋 모두 채널을 SCG_x / SCG_y / SCG_z 로 이름 붙였지만, 서로 다른 연구에서 다른
가속도계로 기록됐다. 라벨이 같다고 물리 축이 같다는 보장은 없다. 원 문헌으로 확정할 수 없으므로
신호에서 확인할 수 있는 만큼 확인한다.

무엇을 볼 수 있나. 흉벽 SCG 에서 축마다 진동 특성이 다르다. 특히 배후-전방(dorsoventral) 축이
대개 가장 큰 진폭을 갖는다. 따라서 데이터셋 간에

  (a) 축별 분산의 상대적 순서,
  (b) 축별 주파수 대역 에너지 분포,
  (c) 축 사이 상관구조

가 비슷하면 라벨이 일관될 가능성이 높고, 순서가 뒤바뀌어 있으면 축이 치환됐을 가능성을 시사한다.

무엇을 확정할 수 없나. 이 분석은 **필요조건만 본다.** 통계가 비슷해도 회전이 섞여 있을 수 있고,
개인 간 센서 부착 각도 편차는 어차피 분리되지 않는다. 결론은 "치환 증거가 없다" 또는 "치환이
의심된다" 까지이며, "정렬돼 있다" 로 단정하지 않는다. 원고에는 그 한계까지 함께 쓴다.

사용법.
    python analysis/axis_alignment.py --out out/axis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal as sps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.datasets import select_scg_channels  # noqa: E402

DATA = Path("/work/jkim1/SCG_HVD_data")
FS = 256
AXES = ["x", "y", "z"]


def band_energy(sig, fs=FS, bands=((1, 5), (5, 15), (15, 30))):
    f, p = sps.welch(sig, fs=fs, nperseg=min(512, len(sig)))
    tot = np.trapz(p, f) + 1e-12
    return [float(np.trapz(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]) / tot)
            for lo, hi in bands]


def per_segment_features(x):
    """(T,3) SCG 에서 축별 특징. 진폭 스케일은 장비마다 다르므로 비율 위주로 본다."""
    out = {}
    v = x.var(axis=0)
    frac = v / (v.sum() + 1e-12)
    for i, ax in enumerate(AXES):
        out[f"var_frac_{ax}"] = float(frac[i])
        b = band_energy(x[:, i])
        for (lo, hi), val in zip(((1, 5), (5, 15), (15, 30)), b):
            out[f"band{lo}_{hi}_{ax}"] = val
    c = np.corrcoef(x.T)
    out["corr_xy"], out["corr_xz"], out["corr_yz"] = float(c[0, 1]), float(c[0, 2]), float(c[1, 2])
    out["dominant_axis"] = AXES[int(np.argmax(v))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--per-patient", type=int, default=10, help="환자당 표본 세그먼트 수")
    ap.add_argument("--out", type=Path, default=Path("out/axis"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{a.task}.csv")
    df["filepath"] = df.filepath.str.replace(
        "/hpc/dctrl/jk622/exp/2025_BHI/data/Data/", str(DATA) + "/", regex=False)
    df["dataset"] = np.where(df.patient_id.str.startswith("sub"), "Dataset II", "Dataset I")

    rows = []
    for pid, g in df.groupby("patient_id"):
        sub = g.sample(min(a.per_patient, len(g)), random_state=42)
        for _, r in sub.iterrows():
            x = select_scg_channels(np.load(r.filepath).astype(np.float64))
            rows.append({"patient_id": pid, "dataset": r.dataset, "label": r.label,
                         **per_segment_features(x)})
    d = pd.DataFrame(rows)
    d.to_csv(a.out / f"{a.task}_axis_features.csv", index=False)

    print(f"=== {a.task}: 축별 분산 비율 (데이터셋별 평균) ===")
    vf = d.groupby("dataset")[[f"var_frac_{ax}" for ax in AXES]].mean().round(3)
    print(vf.to_string())
    print("\n  축별 분산 비율의 순서가 데이터셋 간 같으면 라벨이 일관될 가능성이 높다.")
    for ds, row in vf.iterrows():
        order = [AXES[i] for i in np.argsort(-row.values)]
        print(f"    {ds}: {' > '.join(order)}")

    print(f"\n=== 우세 축 분포 ===")
    print(pd.crosstab(d.dataset, d.dominant_axis, normalize="index").round(3).to_string())

    print(f"\n=== 대역별 에너지 비율 (데이터셋별 평균) ===")
    cols = [f"band{lo}_{hi}_{ax}" for ax in AXES for lo, hi in ((1, 5), (5, 15), (15, 30))]
    print(d.groupby("dataset")[cols].mean().round(3).T.to_string())

    print(f"\n=== 축 간 상관 ===")
    print(d.groupby("dataset")[["corr_xy", "corr_xz", "corr_yz"]].mean().round(3).to_string())

    # 판정
    if d.dataset.nunique() < 2:
        print("\n판정 불가: 데이터셋이 하나뿐이다.")
        return
    o = {ds: [AXES[i] for i in np.argsort(-vf.loc[ds].values)] for ds in vf.index}
    same = len(set(tuple(v) for v in o.values())) == 1
    print("\n" + "=" * 66)
    if same:
        print("판정: 축별 분산 순서가 두 데이터셋에서 일치한다. 축 치환의 증거는 없다.")
    else:
        print("판정: 축별 분산 순서가 데이터셋 간 다르다. 축 치환 또는 방향 차이가 의심된다.")
    print("주의: 이 분석은 필요조건만 본다. 순서가 같아도 회전이 섞여 있을 수 있고,")
    print("      개인별 부착 각도 편차는 분리되지 않는다. 원고에는 한계와 함께 기술한다.")
    print(f"\n결과: {a.out}")


if __name__ == "__main__":
    main()
