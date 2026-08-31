# Datasets for the segment waveforms and their per-axis CWT scalograms, carrying over the
# preprocessing of the canonical archived scripts unchanged.
"""
Which columns hold the SCG channels depends on the source file, and the rule follows the
canonical script (`archive/_canonical/1d__hvdnet_model.py`):

    7 columns   Dataset II family  -> SCG in columns 1, 2, 3
    12 columns  Dataset I family   -> SCG in columns 0, 1, 2

Images are normalised with the ImageNet statistics. One published run, the Task II fusion
model, used `[0.5], [0.5]` instead; that does not match an ImageNet-pretrained backbone and we
recorded it as a defect during the audit. Reproducing it requires asking for
`image_norm="half"` explicitly rather than getting it by accident.

The scalograms the models were actually trained on are grayscale, 224x224 with R=G=B, and
`.convert("RGB")` replicates the single channel across the three the backbone expects. The
viridis-colormapped variant (`Task2_images_RGB`) was not used for any published number and
exists only for the comparison Referee 1 asked about.
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
    """Pick the three tri-axial SCG columns out of a (T, C) array."""
    if x.shape[1] == 7:
        return x[:, 1:4]
    if x.shape[1] == 12:
        return x[:, 0:3]
    raise ValueError(f"unexpected channel count: {x.shape[1]}")


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
    """Waveforms only: returns a (3, 2560) tensor and its label."""

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
    """Images only: returns the three per-axis scalograms."""

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
    """Fusion: returns the waveform and the three scalograms together."""

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = np.load(row["filepath"]).astype(np.float32)
        x = select_scg_channels(x)
        ix, iy, iz = self._images(row)
        return torch.from_numpy(x.T.copy()), ix, iy, iz, torch.tensor(int(row["label"]))


#: Models that read waveforms only and need no image directory, including the baselines added
#: during the revision.
WAVEFORM_MODELS = {"1d", "resnet1d", "tcn",
                   "temporal_matched", "resnet1d_matched", "tcn_matched"}


def build_dataset(model_name, df, image_dir=None, **kw):
    """Build whichever dataset the named model needs."""
    if model_name in WAVEFORM_MODELS:
        return SegmentDataset(df)
    if image_dir is None:
        raise ValueError(f"{model_name} needs image_dir")
    if model_name.startswith("2d"):
        return TripleImageDataset(df, image_dir, **kw)
    if model_name.startswith("fusion"):
        return FusionDataset(df, image_dir, **kw)
    raise KeyError(model_name)


def unpack_batch(model_name, batch, device):
    """Split a batch into (model inputs, label) so the training loop need not know the model."""
    *inputs, y = batch
    inputs = tuple(t.to(device, non_blocking=True) for t in inputs)
    return inputs, y.to(device, non_blocking=True)
