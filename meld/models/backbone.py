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
        try:
            from torchvision.models import resnet18
            from torchvision.models import ResNet18_Weights
        except Exception:
            return
        try:
            reference = resnet18(weights=ResNet18_Weights.DEFAULT)
        except Exception:
            return
        with torch.no_grad():
            conv = reference.conv1.weight[:16, :, 2:5, 2:5]
            self.conv_1_3x3.weight.copy_(conv)
            self.bn_1.weight.copy_(reference.bn1.weight[:16])
            self.bn_1.bias.copy_(reference.bn1.bias[:16])
            self.bn_1.running_mean.copy_(reference.bn1.running_mean[:16])
            self.bn_1.running_var.copy_(reference.bn1.running_var[:16])


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
