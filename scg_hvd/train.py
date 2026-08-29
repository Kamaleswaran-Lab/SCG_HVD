# 학습 루프. 모델 종류와 분할 방식을 모르게 해서 한 경로로 모든 실험을 돌린다.
"""
정본(`archive/_canonical/`)의 학습 설정을 그대로 옮겼다.

    optimizer   AdamW(lr, weight_decay=1e-4)
    scheduler   CosineAnnealingLR(T_max=epochs)
    criterion   CrossEntropyLoss(weight=balanced class weights)
    선택 기준   검증 정확도 최대 시점의 체크포인트

`class_weight_scope` 만 정본과 다르게 **선택 가능**하게 했다. 정본은 전체 데이터로 가중치를
계산했고(`archive/_canonical/1d__hvdnet_model.py:310`) 이는 엄밀히는 학습 외 정보를 쓴 것이다
(Referee 1 minor 2). 감사에서 실측한 편차는 Task I 0.041%, Task II 0.52% 로 미미하지만,
논문 재현에는 `"all"`, 새 실험에는 `"train"` 을 쓴다.

`drop_last` 는 조건부다. 무조건 True 로 두면 모든 fold 에서 학습 표본 약 0.4% 와 옵티마이저
스텝 하나가 빠져 아카이브 결과와 비교가 깨진다. 마지막 배치가 정확히 1개일 때만 버리며,
이는 BatchNorm 이 배치 1에서 실패하는 것(아카이브 LOOCV fold 27 크래시)만 막는다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from .datasets import WAVEFORM_MODELS, build_dataset, unpack_batch
from .models import build_model


@dataclass
class TrainConfig:
    model: str = "1d"
    num_classes: int = 5
    epochs: int = 25
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 8
    seed: int = 42
    image_norm: str = "imagenet"
    class_weight_scope: str = "all"     # "all" = 논문 재현, "train" = 누수 없는 새 실험
    early_stop_patience: int | None = None
    amp: bool = True
    save_checkpoint: bool = False   # 해석성 분석(R2-m7)에 쓰려면 켠다


def balanced_class_weights(labels, num_classes):
    """전체 클래스 길이의 balanced 가중치 벡터를 만든다.

    환자 단위 fold 에서는 학습 분할에 아예 없는 클래스가 생길 수 있다. Task II 의 AS-AR 은
    환자가 1명뿐이라 그 환자가 test 로 가는 fold 에서는 학습에 등장하지 않는다.
    `compute_class_weight` 는 존재하는 클래스만 돌려주므로 그대로 쓰면 길이가 어긋난다.
    없는 클래스는 가중치 1.0 으로 채운다. 학습에 등장하지 않으므로 손실에 기여하지 않는다.

    Returns (weights, missing_classes).
    """
    labels = np.asarray(labels)
    present = np.unique(labels)
    w = np.ones(num_classes, dtype=np.float64)
    if len(present) >= 2:
        cw = compute_class_weight("balanced", classes=present, y=labels)
        w[present] = cw
    missing = sorted(set(range(num_classes)) - set(present.tolist()))
    return w, missing


def make_loader(ds, batch_size, shuffle, num_workers, drop_last=False):
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        pin_memory=True, drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )


def _drop_last_needed(n, batch_size):
    """마지막 배치가 정확히 1개면 BatchNorm 이 터진다. 그 경우에만 버린다."""
    return n % batch_size == 1


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, model_name, device, n_classes):
    """예측과 확률을 모두 돌려준다. 지표는 이 원본에서 계산한다."""
    model.eval()
    ys, ps, probs = [], [], []
    for batch in loader:
        inputs, y = unpack_batch(model_name, batch, device)
        out = model(*inputs)
        pr = torch.softmax(out.float(), dim=1)
        ys.append(y.cpu().numpy())
        ps.append(out.argmax(1).cpu().numpy())
        probs.append(pr.cpu().numpy())
    return (np.concatenate(ys), np.concatenate(ps), np.concatenate(probs))


def run_training(split_df: pd.DataFrame, cfg: TrainConfig, image_dir=None,
                 out_dir: Path | None = None, class_names=None, verbose=True):
    """한 번의 학습·평가. split_df 는 'split' 열로 train/val/test 를 지정한다."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.seed)

    tr = split_df[split_df.split == "train"]
    va = split_df[split_df.split == "val"]
    te = split_df[split_df.split == "test"]

    ds_kw = {} if cfg.model in WAVEFORM_MODELS else {"image_norm": cfg.image_norm}
    train_ds = build_dataset(cfg.model, tr, image_dir, **ds_kw)
    val_ds = build_dataset(cfg.model, va, image_dir, **ds_kw)
    test_ds = build_dataset(cfg.model, te, image_dir, **ds_kw)

    train_loader = make_loader(train_ds, cfg.batch_size, True, cfg.num_workers,
                               drop_last=_drop_last_needed(len(train_ds), cfg.batch_size))
    val_loader = make_loader(val_ds, cfg.batch_size, False, cfg.num_workers)
    test_loader = make_loader(test_ds, cfg.batch_size, False, cfg.num_workers)

    model = build_model(cfg.model, cfg.num_classes).to(device)

    weight_src = split_df if cfg.class_weight_scope == "all" else tr
    cw_full, missing = balanced_class_weights(weight_src["label"], cfg.num_classes)
    if missing:
        print(f"  [주의] 학습 분할에 없는 클래스 {missing} — 이 fold 에서는 예측될 수 없다.",
              flush=True)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(cw_full, dtype=torch.float, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")

    best_val, best_state, bad, log = -1.0, None, 0, []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        tl = tc = tn = 0
        for batch in train_loader:
            inputs, y = unpack_batch(cfg.model, batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                out = model(*inputs)
                loss = criterion(out, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            tl += loss.item() * y.size(0)
            tc += (out.argmax(1) == y).sum().item()
            tn += y.size(0)
        scheduler.step()

        model.eval()
        vl = vc = vn = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, y = unpack_batch(cfg.model, batch, device)
                with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
                    out = model(*inputs)
                    loss = criterion(out, y)
                vl += loss.item() * y.size(0)
                vc += (out.argmax(1) == y).sum().item()
                vn += y.size(0)

        tr_acc, va_acc = tc / max(tn, 1), vc / max(vn, 1)
        log.append({"epoch": epoch + 1, "train_loss": tl / max(tn, 1), "train_acc": tr_acc,
                    "val_loss": vl / max(vn, 1), "val_acc": va_acc})
        if verbose:
            print(f"  epoch {epoch+1:03d}  train {tr_acc:.4f}  val {va_acc:.4f}", flush=True)

        if va_acc > best_val:
            best_val, bad = va_acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if cfg.early_stop_patience and bad >= cfg.early_stop_patience:
                if verbose:
                    print(f"  early stop at epoch {epoch+1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_pred, y_prob = predict(model, test_loader, cfg.model, device, cfg.num_classes)
    elapsed = time.time() - t0

    result = {
        "config": asdict(cfg),
        "best_val_acc": best_val,
        "epochs_run": len(log),
        "elapsed_sec": round(elapsed, 1),
        "n_train": len(tr), "n_val": len(va), "n_test": len(te),
        "n_train_patients": tr.patient_id.nunique(),
        "n_test_patients": te.patient_id.nunique(),
        "drop_last": _drop_last_needed(len(train_ds), cfg.batch_size),
    }

    if out_dir:
        out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        if cfg.save_checkpoint and best_state is not None:
            torch.save(best_state, out_dir / "model.pt")
        pd.DataFrame(log).to_csv(out_dir / "training_log.csv", index=False)
        pred = pd.DataFrame({
            "segment_id": te.segment_id.values if "segment_id" in te else np.arange(len(te)),
            "patient_id": te.patient_id.values,
            "y_true": y_true, "y_pred": y_pred,
        })
        for i in range(cfg.num_classes):
            pred[f"prob_{i}"] = y_prob[:, i]
        pred.to_csv(out_dir / "predictions.csv", index=False)
        (out_dir / "resolved_config.json").write_text(json.dumps(result, indent=2))

    return model, (y_true, y_pred, y_prob), result
