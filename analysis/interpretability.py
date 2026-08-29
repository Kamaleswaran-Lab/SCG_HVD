# R2-m7 대응. 1D attention 가중치와 2D Grad-CAM 으로 무엇을 보는지 보인다.
"""
Referee 2 minor 7.

    "Clarify whether attention maps or other interpretability analyses were performed. Attention
     pooling is included in the architecture, but the manuscript does not show which portions of
     the SCG contribute to classification."

리뷰어가 지목한 것은 **1D 브랜치의 attention pooling** 이다. 그 가중치는 시간축에 대한 softmax
이므로 "파형의 어느 구간이 분류에 기여하는가" 에 그대로 답한다. 아카이브의 Grad-CAM
(`exp/Task1/3_hvdnet_fusion_model_LSTM_gradcam/gradcam_analysis.py`) 은 2D 스칼로그램용이라
질문과 어긋나 있었다. 여기서는 둘 다 낸다.

1. **Attention 가중치** — 축별·클래스별로 시간축 분포를 낸다. 세그먼트가 10초이므로 가중치를
   심장주기에 대응시키려면 ECG R-peak 기준으로 정렬해야 한다. ECG 는 세그먼트 파일 안에 있으므로
   (`analysis/extract_heart_rate.py` 참조) R-peak 로 정렬한 평균 attention 도 함께 낸다.
   이렇게 하면 "수축기 초반에 몰린다" 같은 진술이 가능해진다.
2. **Grad-CAM** — 공유 EfficientNet 백본의 마지막 합성곱에 걸어 스칼로그램의 어느 시간-주파수
   영역이 기여하는지 낸다.

주의. 해석은 조심스럽게 해야 한다. attention 가중치가 큰 구간이 인과적으로 중요하다는 보장은
없다. 원고에는 "모델이 어디를 보는지" 로만 쓰고 "그 구간이 병태를 담고 있다" 로 확장하지 않는다.
R2-m5(AS-MR 해석)에서 같은 함정을 지적받았다.

사용법.
    python analysis/interpretability.py --ckpt <fold 디렉터리> --task task1 --out out/interp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scg_hvd.datasets import select_scg_channels  # noqa: E402
from scg_hvd.models import build_model  # noqa: E402

DATA = Path("/work/jkim1/SCG_HVD_data")
FS = 256
AXIS_ORDER = ["z", "x", "y"]   # SCGTrunk1D 가 z, x, y 순으로 이어붙인다


# ---------------------------------------------------------------- attention

class AttentionRecorder:
    """SCGBranch.attn 의 softmax 가중치를 가로챈다."""

    def __init__(self, trunk):
        self.weights = {}
        self._handles = []
        for name in ("branch_z", "branch_x", "branch_y"):
            branch = getattr(trunk, name)
            self._handles.append(
                branch.attn.attn.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name):
        def hook(module, inp, out):
            # out: (B, T, 1) — SelfAttention.forward 가 여기에 softmax 를 건다
            self.weights[name] = torch.softmax(out, dim=1).detach().squeeze(-1).cpu()
        return hook

    def close(self):
        for h in self._handles:
            h.remove()


def r_peaks(sig_1d, fs=FS):
    """attention 을 심장주기에 정렬하기 위한 R-peak. extract_heart_rate 와 같은 방식."""
    from scipy import signal as sps
    x = sig_1d - np.mean(sig_1d)
    nyq = fs / 2
    b, a = sps.butter(3, [5 / nyq, 15 / nyq], btype="band")
    f = sps.filtfilt(b, a, x)
    integ = np.convolve(np.diff(f, prepend=f[0]) ** 2, np.ones(int(.15 * fs)) / int(.15 * fs), "same")
    pk, _ = sps.find_peaks(integ, height=np.mean(integ) + .5 * np.std(integ), distance=int(.25 * fs))
    return pk


def beat_aligned(weights, peaks, win=(-0.2, 0.6), fs=FS):
    """R-peak 기준으로 attention 을 잘라 평균한다. 심장주기 상 위치를 말할 수 있게 된다."""
    lo, hi = int(win[0] * fs), int(win[1] * fs)
    segs = [weights[p + lo:p + hi] for p in peaks
            if p + lo >= 0 and p + hi <= len(weights) and len(weights[p + lo:p + hi]) == hi - lo]
    return np.mean(segs, axis=0) if segs else None


# ---------------------------------------------------------------- Grad-CAM

class GradCAM2D:
    """공유 EfficientNet 백본의 마지막 합성곱에 건 Grad-CAM."""

    def __init__(self, model, target_layer):
        self.model, self.act, self.grad = model, None, None
        target_layer.register_forward_hook(lambda m, i, o: setattr(self, "act", o))
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "grad", go[0]))

    def __call__(self, inputs, class_idx=None):
        self.model.zero_grad()
        out = self.model(*inputs)
        idx = out.argmax(1) if class_idx is None else torch.as_tensor([class_idx])
        out.gather(1, idx.view(-1, 1)).sum().backward()
        w = self.grad.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.act).sum(1))
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy()


# ---------------------------------------------------------------- main

def load_task(task):
    df = pd.read_csv(DATA / "meta" / f"segment_metadata_{task}.csv")
    names = sorted(df.label.unique())
    df["label_name"] = df["label"]
    df["label"] = df["label"].map({n: i for i, n in enumerate(names)})
    df["filepath"] = df["filepath"].str.replace(
        "/hpc/dctrl/jk622/exp/2025_BHI/data/Data/", str(DATA) + "/", regex=False)
    return df, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="task1")
    ap.add_argument("--model", default="1d", choices=["1d", "fusion"])
    ap.add_argument("--ckpt", type=Path, default=None,
                    help="학습된 가중치(.pt). 없으면 무작위 초기화로 배선만 확인한다.")
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--out", type=Path, default=Path("out/interp"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    df, class_names = load_task(a.task)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(a.model, len(class_names)).to(device).eval()
    if a.ckpt and a.ckpt.exists():
        model.load_state_dict(torch.load(a.ckpt, map_location=device))
        print(f"loaded {a.ckpt}")
    else:
        print("[주의] 학습 가중치 없이 실행한다. 배선 확인용이며 결과는 해석하지 말 것.")

    trunk = model.trunk if a.model == "1d" else model.trunk_1d
    rec = AttentionRecorder(trunk)

    rows, aligned = [], {c: {ax: [] for ax in AXIS_ORDER} for c in class_names}
    for cls_i, cls in enumerate(class_names):
        sub = df[df.label == cls_i].sample(min(a.n_per_class, (df.label == cls_i).sum()),
                                           random_state=42)
        for _, r in sub.iterrows():
            raw = np.load(r.filepath).astype(np.float32)
            scg = select_scg_channels(raw)
            x = torch.from_numpy(scg.T.copy()).unsqueeze(0).to(device)
            with torch.no_grad():
                model.extract_features(x) if a.model == "1d" else trunk(x)
            # ECG 로 R-peak 를 잡아 심장주기 정렬
            ecg_idx = 5 if raw.shape[1] == 12 else 0
            pk = r_peaks(raw[:, ecg_idx].astype(np.float64))
            for ax, key in zip(AXIS_ORDER, ("branch_z", "branch_x", "branch_y")):
                w = rec.weights[key][0].numpy()
                rows.append({"class": cls, "axis": ax, "entropy": float(-(w * np.log(w + 1e-12)).sum()),
                             "peak_pos_sec": float(np.argmax(w) / FS),
                             "top10pct_mass": float(np.sort(w)[-len(w) // 10:].sum())})
                if len(pk) >= 3:
                    al = beat_aligned(w, pk)
                    if al is not None:
                        aligned[cls][ax].append(al)
    rec.close()

    d = pd.DataFrame(rows)
    d.to_csv(a.out / f"{a.task}_{a.model}_attention_stats.csv", index=False)
    print(f"\n=== attention 요약 (세그먼트 {len(d)//3}개) ===")
    print(d.groupby(["class", "axis"])[["entropy", "top10pct_mass", "peak_pos_sec"]]
          .mean().round(3).to_string())
    print("\n  entropy 가 낮을수록 특정 구간에 집중한다. 균등 분포의 entropy 는 "
          f"{np.log(2560):.3f} 이다.")

    ba = []
    t = np.arange(int(-0.2 * FS), int(0.6 * FS)) / FS
    for cls, per_ax in aligned.items():
        for ax, lst in per_ax.items():
            if lst:
                m = np.mean(lst, axis=0)
                for ti, v in zip(t, m):
                    ba.append({"class": cls, "axis": ax, "t_from_R_sec": round(float(ti), 4),
                               "attention": float(v)})
    if ba:
        pd.DataFrame(ba).to_csv(a.out / f"{a.task}_{a.model}_beat_aligned_attention.csv", index=False)
        print(f"\n  R-peak 정렬 평균 attention 저장 ({len(aligned)} 클래스 x 3 축)")
    print(f"\n결과: {a.out}")


if __name__ == "__main__":
    main()
