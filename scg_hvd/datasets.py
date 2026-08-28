# 세그먼트 신호와 축별 CWT 스칼로그램을 읽는 Dataset. 정본의 전처리를 그대로 옮겼다.
"""
채널 선택 규칙은 정본을 따른다(`archive/_canonical/1d__hvdnet_model.py`).

    7채널  Dataset II 계열 -> SCG 는 열 1,2,3
    12채널 Dataset I  계열 -> SCG 는 열 0,1,2

이미지 정규화는 ImageNet 통계다. 논문 실행 중 Task II 융합만 `[0.5],[0.5]` 를 썼는데,
이는 ImageNet 사전학습 백본에 맞지 않는 설정이며 감사에서 결함으로 기록했다. 재현이 필요하면
`image_norm="half"` 로 명시적으로 요청해야 한다.

실제 학습에 쓰인 스칼로그램은 **그레이스케일**이다(224x224, R=G=B). `.convert("RGB")` 로
3채널로 복제해 백본에 넣는다. viridis 컬러맵 변형(`Task2_images_RGB`)은 논문 실행에서 쓰이지
않았고, 리뷰어 R1-m6 가 요구한 비교 실험용으로만 존재한다.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def select_scg_channels(x: np.ndarray) -> np.ndarray:
    """(T, C) 배열에서 tri-axial SCG 세 열을 고른다."""
    if x.shape[1] == 7:
        return x[:, 1:4]
    if x.shape[1] == 12:
        return x[:, 0:3]
    raise ValueError(f"예상하지 못한 채널 수: {x.shape[1]}")


def build_image_transform(image_size=224, image_norm="imagenet"):
    if image_norm == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif image_norm == "half":
        mean, std = [0.5] * 3, [0.5] * 3
    else:
        raise ValueError(f"unknown image_norm {image_norm!r}")
    return transforms.Compose([
        transforms.Resize((image_size, image_size),
                          interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class SegmentDataset(Dataset):
    """1D 전용. (3, 2560) 파형과 라벨을 돌려준다."""

    def __init__(self, df: pd.DataFrame, path_column="filepath"):
        self.df = df.reset_index(drop=True)
        self.path_column = path_column

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row[self.path_column]).astype(np.float32)
        x = select_scg_channels(x)
        return torch.from_numpy(x.T.copy()), torch.tensor(int(row["label"]))


class TripleImageDataset(Dataset):
    """2D 전용. 축별 스칼로그램 세 장을 돌려준다."""

    def __init__(self, df, image_dir, image_size=224, image_norm="imagenet"):
        self.df = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.tf = build_image_transform(image_size, image_norm)

    def __len__(self):
        return len(self.df)

    def _images(self, row):
        stem = Path(row["filepath"]).stem
        base = self.image_dir / row["label_name"]
        return tuple(
            self.tf(Image.open(base / f"{stem}_{ax}.png").convert("RGB"))
            for ax in ("x", "y", "z")
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ix, iy, iz = self._images(row)
        return ix, iy, iz, torch.tensor(int(row["label"]))


class FusionDataset(TripleImageDataset):
    """융합 전용. 파형 하나와 스칼로그램 세 장을 함께 돌려준다."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["filepath"]).astype(np.float32)
        x = select_scg_channels(x)
        ix, iy, iz = self._images(row)
        return torch.from_numpy(x.T.copy()), ix, iy, iz, torch.tensor(int(row["label"]))


def build_dataset(model_name, df, image_dir=None, **kw):
    """모델 종류에 맞는 Dataset 을 만든다."""
    if model_name == "1d":
        return SegmentDataset(df)
    if image_dir is None:
        raise ValueError(f"{model_name} 은 image_dir 가 필요하다")
    if model_name.startswith("2d"):
        return TripleImageDataset(df, image_dir, **kw)
    if model_name.startswith("fusion"):
        return FusionDataset(df, image_dir, **kw)
    raise KeyError(model_name)


def unpack_batch(model_name, batch, device):
    """(모델 입력 튜플, 라벨) 로 나눈다. 학습 루프가 모델 종류를 몰라도 되게 한다."""
    *inputs, y = batch
    inputs = tuple(t.to(device, non_blocking=True) for t in inputs)
    return inputs, y.to(device, non_blocking=True)
