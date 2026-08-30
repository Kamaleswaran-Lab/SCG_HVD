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
        # 기본값 (64, 100) 이 논문 구성이다. parameter-matched 비교(R1-M6, R2-M4)에서만 키운다.
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

    def __init__(self, conv_channels=64, lstm_hidden=100):
        super().__init__()
        self.lstm_hidden = lstm_hidden
        self.branch_x = SCGBranch(conv_channels=conv_channels, lstm_hidden=lstm_hidden)
        self.branch_y = SCGBranch(conv_channels=conv_channels, lstm_hidden=lstm_hidden)
        self.branch_z = SCGBranch(conv_channels=conv_channels, lstm_hidden=lstm_hidden)

    def forward(self, x):
        xz = self.branch_z(x[:, 2:3, :])
        xx = self.branch_x(x[:, 0:1, :])
        xy = self.branch_y(x[:, 1:2, :])
        return torch.cat([xz, xx, xy], dim=1)


class HVDNet1D(nn.Module):
    """Temporal Encoder (1D). 원고 Table 2/3의 'Temporal Encoder (1D)' 행."""

    def __init__(self, num_classes, conv_channels=64, lstm_hidden=100):
        super().__init__()
        self.trunk = SCGTrunk1D(conv_channels, lstm_hidden)
        self.fc = nn.Sequential(
            nn.Linear(3 * lstm_hidden, 128), nn.ReLU(),
            nn.BatchNorm1d(128), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.fc(self.trunk(x))

    def extract_features(self, x):
        return self.trunk(x)


# ---------------------------------------------------------------- 2D 구성요소


def _feature_dim(backbone) -> int:
    """백본이 실제로 내놓는 차원을 순전파로 잰다.

    `num_features` 속성을 믿으면 안 된다. timm 의 MobileNetV3 는 `num_features` 가 960 이지만
    `num_classes=0` 으로 만들면 conv_head 를 거쳐 1280 을 내놓는다. 속성값으로 투영층을 만들면
    형상 불일치로 터진다. EfficientNet-B0 에서는 둘이 같아 논문 구성에서는 드러나지 않았다.
    """
    import torch as _t
    was_training = backbone.training
    backbone.eval()
    with _t.no_grad():
        d = int(backbone(_t.zeros(1, 3, 224, 224)).shape[1])
    backbone.train(was_training)
    return d


class EfficientNetProjector(nn.Module):
    """축 하나의 스칼로그램을 out_dim 으로 투영한다. Independent 계열에서 축마다 하나씩 쓴다."""

    def __init__(self, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0, drop_rate=0.2)
        self.proj = nn.Sequential(
            nn.Linear(_feature_dim(self.backbone), out_dim),
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
        feat = _feature_dim(self.shared_backbone)
        self.proj_x = nn.Linear(feat, d)
        self.proj_y = nn.Linear(feat, d)
        self.proj_z = nn.Linear(feat, d)
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


# ---------------------------------------------------------------- 리비전 추가 베이스라인
# Referee 1 concern 5 가 요구한 1D ResNet 과 TCN. 원고의 1D 인코더와 같은 입력
# (3, 2560) 을 받고 같은 학습 경로를 쓴다. `width` 로 파라미터를 맞출 수 있게 해
# concern 6 의 parameter-matched 비교에도 쓴다.

class _BasicBlock1D(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = nn.Conv1d(cin, cout, 7, stride=stride, padding=3, bias=False)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, 7, padding=3, bias=False)
        self.b2 = nn.BatchNorm1d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = (nn.Sequential(nn.Conv1d(cin, cout, 1, stride=stride, bias=False),
                                   nn.BatchNorm1d(cout))
                     if (stride != 1 or cin != cout) else None)

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        return self.relu(out + idt)


class ResNet1D(nn.Module):
    """1D ResNet 베이스라인 (R1-M5). 채널 축은 축 3개를 그대로 받는다."""

    def __init__(self, num_classes, width=64, layers=(2, 2, 2, 2), in_channels=3):
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, w, 15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(w), nn.ReLU(inplace=True), nn.MaxPool1d(3, stride=2, padding=1))
        blocks, cin = [], w
        for i, n in enumerate(layers):
            cout = w * (2 ** i)
            for j in range(n):
                blocks.append(_BasicBlock1D(cin, cout, stride=2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(cin, num_classes))
        self.feat_dim = cin

    def extract_features(self, x):
        h = self.blocks(self.stem(x))
        return torch.flatten(nn.functional.adaptive_avg_pool1d(h, 1), 1)

    def forward(self, x):
        return self.head(self.blocks(self.stem(x)))


class _TCNBlock(nn.Module):
    def __init__(self, cin, cout, k, dilation, dropout=0.2):
        super().__init__()
        pad = (k - 1) * dilation
        self.pad = pad
        self.c1 = nn.Conv1d(cin, cout, k, padding=pad, dilation=dilation)
        self.b1 = nn.BatchNorm1d(cout)
        self.c2 = nn.Conv1d(cout, cout, k, padding=pad, dilation=dilation)
        self.b2 = nn.BatchNorm1d(cout)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.down = nn.Conv1d(cin, cout, 1) if cin != cout else None

    def _chomp(self, x):
        return x[:, :, :-self.pad] if self.pad else x

    def forward(self, x):
        idt = x if self.down is None else self.down(x)
        out = self.drop(self.relu(self.b1(self._chomp(self.c1(x)))))
        out = self.drop(self.relu(self.b2(self._chomp(self.c2(out)))))
        return self.relu(out + idt)


class TCN(nn.Module):
    """Temporal Convolutional Network 베이스라인 (R1-M5). 인과 팽창 합성곱 스택."""

    def __init__(self, num_classes, width=64, levels=6, kernel_size=7, in_channels=3):
        super().__init__()
        layers, cin = [], in_channels
        for i in range(levels):
            layers.append(_TCNBlock(cin, width, kernel_size, dilation=2 ** i))
            cin = width
        self.net = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                  nn.Dropout(0.3), nn.Linear(width, num_classes))
        self.feat_dim = width

    def extract_features(self, x):
        return torch.flatten(nn.functional.adaptive_avg_pool1d(self.net(x), 1), 1)

    def forward(self, x):
        return self.head(self.net(x))


MODELS.update({"resnet1d": ResNet1D, "tcn": TCN})


# ---------------------------------------------------------------- parameter-matched 구성
# R1-M6 과 R2-M4 는 "융합 모델이 용량이 커서 좋아진 것 아니냐" 를 묻는다. 아래 세 구성은
# 융합 모델(4,767,374) 과 파라미터 수를 맞춘 것이며, 동시에 R1-M5 가 요구한 강한 베이스라인
# 역할을 한다. 용량을 통제했으므로 차이가 나면 아키텍처 차이로 읽을 수 있다.

def temporal_matched(num_classes, **kw):
    """논문의 1D 인코더를 융합 모델 크기까지 키운 것. R2-M4 의 parameter-matched temporal baseline."""
    return HVDNet1D(num_classes, conv_channels=208, lstm_hidden=288)


def resnet1d_matched(num_classes, **kw):
    return ResNet1D(num_classes, width=48, layers=(2, 2, 2, 2))


def tcn_matched(num_classes, **kw):
    return TCN(num_classes, width=256, levels=6)


MODELS.update({
    "temporal_matched": temporal_matched,
    "resnet1d_matched": resnet1d_matched,
    "tcn_matched": tcn_matched,
})


# ---------------------------------------------------------------- 2D 백본 교체 ablation
# R2-M4 가 요구한 ablation 이자, 논문의 주장을 직접 지지하는 실험이다.
# 주장은 "우리 아키텍처가 다른 아키텍처보다 낫다" 가 아니라 "시간 인코더에 스펙트로템포럴
# 브랜치를 더하면 좋아진다" 이므로, 2D 백본을 바꿔 가며 그 이득이 백본 선택에 의존하지 않음을
# 보이는 편이 standalone 베이스라인과 경쟁하는 것보다 주장에 맞는다.

def _fusion_with(backbone):
    def f(num_classes, **kw):
        return FusionShared(num_classes, model_name=backbone)
    f.__name__ = f"fusion_{backbone}"
    return f


def _image_with(backbone):
    def f(num_classes, **kw):
        return Image2DShared(num_classes, model_name=backbone)
    f.__name__ = f"2d_{backbone}"
    return f


#: 교체해 볼 2D 백본. EfficientNet-B0 이 논문 구성이다.
ABLATION_BACKBONES = {
    "resnet18": "resnet18",
    "mobilenet": "mobilenetv3_large_100",
    "densenet": "densenet121",
}

for _alias, _name in ABLATION_BACKBONES.items():
    MODELS[f"fusion_{_alias}"] = _fusion_with(_name)
    MODELS[f"2d_{_alias}"] = _image_with(_name)
