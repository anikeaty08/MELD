"""CIFAR-style ResNet backbones that return embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn import init


class DownsampleA(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int) -> None:
        super().__init__()
        if stride != 2:
            raise ValueError("DownsampleA only supports stride=2.")
        self.avg = nn.AvgPool2d(kernel_size=1, stride=stride)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avg(x)
        return torch.cat((x, x.mul(0.0)), dim=1)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv_a = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn_a = nn.BatchNorm2d(planes)
        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_b = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = F.relu(self.bn_a(self.conv_a(x)), inplace=True)
        out = self.bn_b(self.conv_b(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(residual + out, inplace=True)


class ResNetBackbone(nn.Module):
    def __init__(self, depth: int, channels: int = 3, pretrained: bool = False) -> None:
        super().__init__()
        if (depth - 2) % 6 != 0:
            raise ValueError("depth should be one of 20, 32, 44, or 56")
        blocks_per_layer = (depth - 2) // 6
        self.in_planes = 16
        self.conv_1_3x3 = nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn_1 = nn.BatchNorm2d(16)
        self.stage_1 = self._make_layer(16, blocks_per_layer, stride=1)
        self.stage_2 = self._make_layer(32, blocks_per_layer, stride=2)
        self.stage_3 = self._make_layer(64, blocks_per_layer, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.out_dim = 64
        self._init_weights()
        if pretrained:
            self._init_from_torchvision()

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != planes * BasicBlock.expansion:
            downsample = DownsampleA(self.in_planes, planes * BasicBlock.expansion, stride)
        layers = []
        layers.append(BasicBlock(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_planes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, (2.0 / n) ** 0.5)
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1.0)
                module.bias.data.zero_()
            elif isinstance(module, nn.Linear):
                init.kaiming_normal_(module.weight)
                module.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.bn_1(self.conv_1_3x3(x)), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def _init_from_torchvision(self) -> None:
        """Map ImageNet-pretrained ResNet-18 weights into this CIFAR ResNet.

        Channel mapping (CIFAR stage → ResNet-18 layer → slice):
          stem   :  16 channels  ←  conv1  [64]       → [:16]
          stage_1:  16 channels  ←  layer1 [64]       → [:16, :16]
          stage_2:  32 channels  ←  layer2 [128]      → [:32, :32]
          stage_3:  64 channels  ←  layer3 [256]      → [:64, :64]

        Kernel mapping:
          stem: 3×3 ← centre-crop of 7×7 (rows 2:5, cols 2:5)
          block convs: 3×3 ← direct (same kernel size in both architectures)

        ResNet-18 has 2 blocks per layer; CIFAR ResNets may have more.
        Only the first min(dst, src) blocks per stage receive pretrained
        weights — remaining blocks keep their Kaiming initialisation.
        DownsampleA uses zero-padding with no learnable parameters, so
        there is nothing to copy for the stride-2 downsample blocks.
        """
        try:
            from torchvision.models import ResNet18_Weights, resnet18
        except Exception:
            return
        try:
            ref = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            return

        with torch.no_grad():
            # ── Stem ────────────────────────────────────────────────────
            # ref.conv1 : [64, 3, 7, 7]  →  conv_1_3x3 : [16, 3, 3, 3]
            # centre-crop the 7×7 kernel down to 3×3 first, then partial-copy
            stem_src = ref.conv1.weight[:, :, 2:5, 2:5].contiguous()
            _partial_copy(self.conv_1_3x3.weight, stem_src)
            _copy_bn(self.bn_1, ref.bn1)

            # ── Stages ──────────────────────────────────────────────────
            _copy_stage(self.stage_1, ref.layer1)
            _copy_stage(self.stage_2, ref.layer2)
            _copy_stage(self.stage_3, ref.layer3)


# ── Pretrained weight-mapping helpers ────────────────────────────────────────

def _partial_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Write src into the matching prefix of dst, leaving the rest untouched.

    For every dimension, copies min(dst_size, src_size) elements.
    The destination tensor retains its Kaiming-initialised values outside
    the copied region, so no uninitialised or zero-padded channels exist.

    This is safer than a bare .copy_() call when dst and src differ in size,
    and avoids the silent semantic issue of leaving excess channels at zero.
    """
    slices = tuple(slice(0, min(d, s)) for d, s in zip(dst.shape, src.shape))
    dst[slices].copy_(src[slices])


def _copy_bn(dst: nn.BatchNorm2d, src: nn.BatchNorm2d) -> None:
    """Copy BN parameters channel-by-channel up to min(dst_ch, src_ch).

    Uses _partial_copy so the destination is never partially zero-filled
    when dst has more channels than src.
    """
    for dst_t, src_t in (
        (dst.weight, src.weight),
        (dst.bias, src.bias),
        (dst.running_mean, src.running_mean),
        (dst.running_var, src.running_var),
    ):
        _partial_copy(dst_t, src_t)


def _copy_stage(
    dst_stage: nn.Sequential,
    src_layer: nn.Sequential,
) -> None:
    """Copy weights from a torchvision ResNet layer into a MELD stage.

    Pairs blocks by position up to min(len(dst), len(src)).
    Uses _partial_copy for every conv and BN tensor so:
      - no destination element is left zero-filled or uninitialised,
      - mismatches in channel count are handled gracefully in both directions
        (dst wider than src, or src wider than dst).
    DownsampleA carries no learnable parameters, so stride-2 downsample
    blocks require no special handling.
    """
    for dst_block, src_block in zip(dst_stage, src_layer):
        _partial_copy(dst_block.conv_a.weight, src_block.conv1.weight)
        _copy_bn(dst_block.bn_a, src_block.bn1)
        _partial_copy(dst_block.conv_b.weight, src_block.conv2.weight)
        _copy_bn(dst_block.bn_b, src_block.bn2)


def _build_resnet(depth: int, pretrained: bool = False) -> ResNetBackbone:
    return ResNetBackbone(depth=depth, pretrained=pretrained)


def resnet20(pretrained: bool = False) -> ResNetBackbone:
    return _build_resnet(20, pretrained=pretrained)


def resnet32(pretrained: bool = False) -> ResNetBackbone:
    return _build_resnet(32, pretrained=pretrained)


def resnet44(pretrained: bool = False) -> ResNetBackbone:
    return _build_resnet(44, pretrained=pretrained)


def resnet56(pretrained: bool = False) -> ResNetBackbone:
    return _build_resnet(56, pretrained=pretrained)
