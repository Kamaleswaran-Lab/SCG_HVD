# One definition of the 1D, 2D and fusion models. The original tree forked near-identical
# variants; this file is the single definition.
"""Ported from the three original scripts that produced the published numbers, identified
here by md5 so the port can be checked against them:

    1D       1d__hvdnet_model.py             md5 c1bf7a2918d802a948824007a230b051
    2D       2d__hvdnet_fusion_model.py      md5 2c6851b26eda18f556c041ce34e90913
    fusion   fusion__hvdnet_fusion_model.py  md5 3c1eeee137328ba62bcb724984654bb3

Two traps caught us during the port, recorded so nobody has to find them twice.

1. The 2D and fusion originals define classes with the same names,
   `MultiEfficientNetFusion` and `OptimizedMultiEfficientNetFusion`. They are not the same
   model: the 2D one has the 1D branch removed, so its classifier takes 126 dimensions where
   the fusion one takes 426. The names are kept distinct here.
2. The top-level copies in the original tree, such as `exp/hvdnet_fusion_model.py`, apply the
   attention weights twice, `weights * x * weights`. Every script used for a published run
   applies them once, and so does the implementation below.

Parameter counts, verified against the code. The Shared variants are what actually ran; the
Independent variants are what the manuscript described before we corrected it.

    HVDNet1D                    526,740
    Image2DShared             4,203,263      Image2DIndependent    12,614,905
    FusionShared              4,767,374      FusionIndependent     13,179,016
"""

import torch
import torch.nn as nn
import timm


# --------------------------------------------------------- 1D building blocks

class SelfAttention(nn.Module):
    """Collapse the sequence axis into a weighted sum."""

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
    """Encode one axis of the SCG waveform into 100 dimensions."""

    def __init__(self, in_channels=1, conv_channels=64, lstm_hidden=100):
        # The defaults (64, 100) are the published configuration. They are widened only for
        # the capacity-controlled comparisons.
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
    """Encode the three axes separately and concatenate to 300 dimensions.

    The axis order is z, x, y, following the original scripts rather than the obvious x, y, z.
    """

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
    """The 'Temporal Encoder (1D)' row of the manuscript's Tables 2 and 3."""

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


# --------------------------------------------------------- 2D building blocks


def _feature_dim(backbone) -> int:
    """Measure the backbone's real output width with a forward pass.

    Do not trust the `num_features` attribute. timm's MobileNetV3 reports 960 there but emits
    1280 when built with `num_classes=0`, because the output still passes through conv_head.
    Sizing the projection from the attribute raises a shape error at the first batch. The two
    agree for EfficientNet-B0, so the published configuration never exposed this.
    """
    import torch as _t
    was_training = backbone.training
    backbone.eval()
    with _t.no_grad():
        d = int(backbone(_t.zeros(1, 3, 224, 224)).shape[1])
    backbone.train(was_training)
    return d


class EfficientNetProjector(nn.Module):
    """Project one axis' scalogram to out_dim. The Independent variants use one per axis."""

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
    """Reuse a single backbone across the three axes, out_dim//3 each, (out_dim//3)*3 total."""

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
    """A separate backbone per axis: what the manuscript described, not what was ever run."""

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
    """The 'Image Encoder (2D)' row of Tables 2 and 3, with the backbone shared across axes."""

    def __init__(self, num_classes, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.trunk = _SharedImageTrunk(model_name, out_dim)
        self.classifier = _head(self.trunk.out_features, num_classes)

    def forward(self, ix, iy, iz):
        return self.classifier(self.trunk(ix, iy, iz))

    def extract_features(self, ix, iy, iz):
        return self.trunk(ix, iy, iz)


class Image2DIndependent(nn.Module):
    """Image Encoder (2D) with three independent backbones, for the capacity comparison."""

    def __init__(self, num_classes, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.trunk = _IndependentImageTrunk(model_name, out_dim)
        self.classifier = _head(self.trunk.out_features, num_classes)

    def forward(self, ix, iy, iz):
        return self.classifier(self.trunk(ix, iy, iz))

    def extract_features(self, ix, iy, iz):
        return self.trunk(ix, iy, iz)


# ---------------------------------------------------------------------- fusion

class FusionShared(nn.Module):
    """Proposed (1D + 2D) with a shared backbone: the configuration that produced the paper."""

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
    """Proposed (1D + 2D) with three independent backbones: what the manuscript described."""

    def __init__(self, num_classes, model_name="efficientnet_b0", out_dim=128):
        super().__init__()
        self.trunk_1d = SCGTrunk1D()
        self.trunk_2d = _IndependentImageTrunk(model_name, out_dim)
        self.classifier = _head(300 + self.trunk_2d.out_features, num_classes)

    def forward(self, x1d, ix, iy, iz):
        return self.classifier(self.extract_features(x1d, ix, iy, iz))

    def extract_features(self, x1d, ix, iy, iz):
        return torch.cat([self.trunk_1d(x1d), self.trunk_2d(ix, iy, iz)], dim=1)


# --------------------------------------------------------------------- factory

MODELS = {
    "1d": HVDNet1D,
    "2d": Image2DShared,
    "2d_independent": Image2DIndependent,
    "fusion": FusionShared,
    "fusion_independent": FusionIndependent,
}

#: What the published runs used. The reproduction check asserts against these.
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


# ------------------------------------------------- baselines added for the revision
# A 1D ResNet and a TCN, named by Referee 1. Both take the same (3, 2560) input as the
# paper's temporal encoder and run through the same training path. `width` lets them be
# sized to match the fusion model, so they double as capacity-controlled baselines.
#
# These were run but are not reported in the manuscript: the paper's claim is about adding a
# modality, not about competing with other sequence architectures, so the ablation that
# varies the image backbone is the one that tests it. They are kept here because the runs
# happened and the record should show them.

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
    """1D ResNet baseline. The three SCG axes enter as the channel dimension."""

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
    """Temporal Convolutional Network baseline: a stack of causal dilated convolutions."""

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


# ------------------------------------------------- capacity-matched configurations
# Both referees asked whether the fusion model simply wins by being larger. These three are
# sized to the fusion model's 4,767,374 parameters, so a remaining difference cannot be
# attributed to capacity. The manuscript answers the question with the backbone ablation
# below instead, which isolates the same thing without inflating a model nobody would
# propose; these are kept for completeness.

def temporal_matched(num_classes, **kw):
    """The paper's temporal encoder widened to the size of the fusion model."""
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


# ----------------------------------------------------- swapping the image backbone
# This is the ablation the paper rests on. The claim is not that this architecture beats
# other architectures; it is that adding a spectrotemporal branch to a temporal encoder
# helps. Holding the temporal branch fixed and swapping the image backbone tests exactly
# that, and the temporal branch costs the same 564,111 parameters whichever backbone is
# underneath, so capacity is held constant too.

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


#: The temporal branch costs this much wherever it is attached: the 1D encoder plus the wider
#: classifier input. Verified identical for every backbone in ABLATION_BACKBONES, which is what
#: lets the ablation attribute a difference to the branch rather than to capacity.
FUSION_PARAM_INCREMENT = 564_111


def verify_fusion_increment(backbones=None, num_classes=5):
    """Check that adding the temporal branch costs the same under every image backbone.

    The ablation's whole argument is that only the modality changes. If a backbone swap also
    changed the size of the increment, a difference in accuracy could be capacity after all.
    Raises rather than warning, because a silent violation would invalidate the comparison.
    """
    # The published configuration is the reference point, so it belongs in the check by
    # default; ABLATION_BACKBONES holds only the substitutes.
    names = list(backbones) if backbones else ["efficientnet_b0", *ABLATION_BACKBONES]
    got = {}
    for bb in names:
        pair = (f"2d_{bb}", f"fusion_{bb}") if bb in ABLATION_BACKBONES else ("2d", "fusion")
        a, b = (count_parameters(build_model(n, num_classes)) for n in pair)
        got[bb] = b - a
    bad = {k: v for k, v in got.items() if v != FUSION_PARAM_INCREMENT}
    if bad:
        raise AssertionError(
            f"the temporal branch does not cost the same everywhere: {bad}, "
            f"expected {FUSION_PARAM_INCREMENT} for all of {names}"
        )
    return got


#: Backbones to swap in. EfficientNet-B0 is the published configuration.
ABLATION_BACKBONES = {
    "resnet18": "resnet18",
    "mobilenet": "mobilenetv3_large_100",
    "densenet": "densenet121",
}

for _alias, _name in ABLATION_BACKBONES.items():
    MODELS[f"fusion_{_alias}"] = _fusion_with(_name)
    MODELS[f"2d_{_alias}"] = _image_with(_name)
