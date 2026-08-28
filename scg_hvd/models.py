# 논문의 1D/2D/융합 모델을 하나로 통합한 정의. 아카이브의 판본 분기를 여기서 끝낸다.
"""
출처. `archive/_canonical/` 의 세 정본에서 이식했다.

    1D      1d__hvdnet_model.py            md5 c1bf7a2918d802a948824007a230b051
    2D      2d__hvdnet_fusion_model.py     md5 2c6851b26eda18f556c041ce34e90913
    융합    fusion__hvdnet_fusion_model.py md5 3c1eeee137328ba62bcb724984654bb3

이식 시 주의했던 함정 두 가지를 기록해 둔다.

1. 2D 정본과 융합 정본은 **클래스 이름이 같다**(`MultiEfficientNetFusion`,
   `OptimizedMultiEfficientNetFusion`). 그러나 2D 쪽은 1D 브랜치가 제거돼 분류기 입력이
   126차원이고 융합 쪽은 426차원이다. 여기서는 이름을 갈라 놓았다.
2. 저장소 최상위에 있던 `exp/hvdnet_fusion_model.py` 등 top-level 사본은 SelfAttention이
   `weights * x * weights`로 가중치를 두 번 곱한다. 논문 실행에 쓰인 정본은 모두
   `weights * x`이며, 아래 구현도 정본을 따른다.

파라미터 수(검증됨). Shared 계열이 논문 실행이고 Independent 계열은 원고 기술이다.

    HVDNet1D                    526,740
    Image2DShared             4,203,263      Image2DIndependent    12,614,905
    FusionShared              4,767,374      FusionIndependent     13,179,016
"""

import torch
import torch.nn as nn
import timm


# ---------------------------------------------------------------- 1D 구성요소

class SelfAttention(nn.Module):
    """시퀀스 축을 가중 합으로 축약한다."""

    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Linear(input_dim, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)
        return torch.sum(weights * x, dim=1)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.resample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.resample:
            identity = self.resample(identity)
        return self.relu(out + identity)


class SCGBranch(nn.Module):
    """단일 축 SCG 파형을 100차원으로 인코딩한다."""

    def __init__(self, in_channels=1, conv_channels=64, lstm_hidden=100):
        super().__init__()
        self.resblock1 = ResidualBlock(in_channels, conv_channels, kernel_size=7, padding=3)
        self.resblock2 = ResidualBlock(conv_channels, conv_channels, kernel_size=5, padding=2)
        self.resblock3 = ResidualBlock(conv_channels, conv_channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(conv_channels, lstm_hidden, batch_first=True)
        self.attn = SelfAttention(lstm_hidden)
        self.dropout = nn.Dropout(0.4)

    def forward(self, x):
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        return self.dropout(self.attn(lstm_out))


class SCGTrunk1D(nn.Module):
    """세 축을 각각 인코딩해 300차원으로 이어 붙인다. 축 순서는 정본대로 z, x, y이다."""

    def __init__(self):
        super().__init__()
        self.branch_x = SCGBranch()
        self.branch_y = SCGBranch()
        self.branch_z = SCGBranch()

    def forward(self, x):
        xz = self.branch_z(x[:, 2:3, :])
        xx = self.branch_x(x[:, 0:1, :])
        xy = self.branch_y(x[:, 1:2, :])
        return torch.cat([xz, xx, xy], dim=1)


class HVDNet1D(nn.Module):
    """Temporal Encoder (1D). 원고 Table 2/3의 'Temporal Encoder (1D)' 행."""

    def __init__(self, num_classes):
        super().__init__()
        self.trunk = SCGTrunk1D()
        self.fc = nn.Sequential(
            nn.Linear(3 * 100, 128), nn.ReLU(),
            nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.trunk(x))

    def extract_features(self, x):
        return self.trunk(x)


# ---------------------------------------------------------------- 2D 구성요소

class EfficientNetProjector(nn.Module):
    """축 하나의 스칼로그램을 out_dim 으로 투영한다. Independent 계열에서 축마다 하나씩 쓴다."""

    def __init__(self, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, drop_rate=0.2)
        self.proj = nn.Sequential(
            nn.Linear(self.backbone.num_features, out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

    def forward(self, x):
        return self.proj(self.backbone(x))


class _SharedImageTrunk(nn.Module):
    """백본 하나를 세 축에 재사용한다. 축당 out_dim//3 이므로 총 (out_dim//3)*3 차원이다."""

    def __init__(self, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.shared_backbone = timm.create_model(
            model_name, pretrained=True, num_classes=0, drop_rate=0.2
        )
        d = out_dim // 3
        self.proj_x = nn.Linear(self.shared_backbone.num_features, d)
        self.proj_y = nn.Linear(self.shared_backbone.num_features, d)
        self.proj_z = nn.Linear(self.shared_backbone.num_features, d)
        self.out_features = d * 3

    def forward(self, ix, iy, iz):
        fx = self.proj_x(self.shared_backbone(ix))
        fy = self.proj_y(self.shared_backbone(iy))
        fz = self.proj_z(self.shared_backbone(iz))
        return torch.cat([fx, fy, fz], dim=1)


class _IndependentImageTrunk(nn.Module):
    """축마다 독립 백본. 원고가 기술한 구성이며 실제 논문 실행에는 쓰이지 않았다."""

    def __init__(self, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.eff_x = EfficientNetProjector(model_name, out_dim)
        self.eff_y = EfficientNetProjector(model_name, out_dim)
        self.eff_z = EfficientNetProjector(model_name, out_dim)
        self.out_features = out_dim * 3

    def forward(self, ix, iy, iz):
        return torch.cat([self.eff_x(ix), self.eff_y(iy), self.eff_z(iz)], dim=1)


def _head(in_dim, num_classes):
    return nn.Sequential(
        nn.Linear(in_dim, 256), nn.ReLU(inplace=True),
        nn.BatchNorm1d(256), nn.Dropout(0.3),
        nn.Linear(256, num_classes),
    )


class Image2DShared(nn.Module):
    """Image Encoder (2D), 공유 백본. 원고 Table 2/3의 'Image Encoder (2D)' 행에 해당한다."""

    def __init__(self, num_classes, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.trunk = _SharedImageTrunk(model_name, out_dim)
        self.classifier = _head(self.trunk.out_features, num_classes)

    def forward(self, ix, iy, iz):
        return self.classifier(self.trunk(ix, iy, iz))

    def extract_features(self, ix, iy, iz):
        return self.trunk(ix, iy, iz)


class Image2DIndependent(nn.Module):
    """Image Encoder (2D), 독립 백본 3개. 파라미터 매칭 비교용(R1-M6)."""

    def __init__(self, num_classes, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.trunk = _IndependentImageTrunk(model_name, out_dim)
        self.classifier = _head(self.trunk.out_features, num_classes)

    def forward(self, ix, iy, iz):
        return self.classifier(self.trunk(ix, iy, iz))

    def extract_features(self, ix, iy, iz):
        return self.trunk(ix, iy, iz)


# ---------------------------------------------------------------- 융합

class FusionShared(nn.Module):
    """Proposed (1D + 2D), 공유 백본. 논문이 실제로 돌린 구성이다."""

    def __init__(self, num_classes, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.trunk_1d = SCGTrunk1D()
        self.trunk_2d = _SharedImageTrunk(model_name, out_dim)
        self.classifier = _head(300 + self.trunk_2d.out_features, num_classes)

    def forward(self, x1d, ix, iy, iz):
        return self.classifier(self.extract_features(x1d, ix, iy, iz))

    def extract_features(self, x1d, ix, iy, iz):
        return torch.cat([self.trunk_1d(x1d), self.trunk_2d(ix, iy, iz)], dim=1)


class FusionIndependent(nn.Module):
    """Proposed (1D + 2D), 독립 백본 3개. 원고가 기술한 구성이다."""

    def __init__(self, num_classes, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.trunk_1d = SCGTrunk1D()
        self.trunk_2d = _IndependentImageTrunk(model_name, out_dim)
        self.classifier = _head(300 + self.trunk_2d.out_features, num_classes)

    def forward(self, x1d, ix, iy, iz):
        return self.classifier(self.extract_features(x1d, ix, iy, iz))

    def extract_features(self, x1d, ix, iy, iz):
        return torch.cat([self.trunk_1d(x1d), self.trunk_2d(ix, iy, iz)], dim=1)


# ---------------------------------------------------------------- 팩토리

MODELS = {
    "1d": HVDNet1D,
    "2d": Image2DShared,
    "2d_independent": Image2DIndependent,
    "fusion": FusionShared,
    "fusion_independent": FusionIndependent,
}

#: 논문 실행이 사용한 구성. 재현 테스트가 이 값을 검사한다.
PAPER_PARAM_COUNTS = {
    "1d": 526_740,
    "2d": 4_203_263,
    "fusion": 4_767_374,
    "2d_independent": 12_614_905,
    "fusion_independent": 13_179_016,
}


def build_model(name, num_classes, **kwargs):
    if name not in MODELS:
        raise KeyError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](num_classes=num_classes, **kwargs)


def count_parameters(model, trainable_only=True):
    ps = model.parameters()
    if trainable_only:
        ps = (p for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in ps)
