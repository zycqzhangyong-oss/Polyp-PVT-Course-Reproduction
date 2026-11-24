"""SC-Net-inspired MDEB, reconstructed from the supplied manuscript, not author code.

Follow Eq.18/Algorithm 1/Table II: concatenate half-width frequency outputs.
Wavelet-inspired filters are learned depthwise 2x2 filters, NOT fixed Haar DWT.
The same-shape residual block replaces (does not follow) Polyp-PVT's CIM.
Compatible with Python 3.8 and PyTorch 1.7.1.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def cbr(cin, cout, kernel=3, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel // 2,
                                   groups=groups, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=False))


class Spatial(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = cbr(channels, channels)
        self.attention = nn.Conv2d(2, 1, 7, padding=3, bias=False)

    def forward(self, x):
        z = self.conv(x)
        pooled = torch.cat((z.mean(1, keepdim=True), z.max(1, keepdim=True)[0]), 1)
        return z * self.attention(pooled).sigmoid()


class Frequency(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.low = cbr(channels, channels, groups=channels)
        self.high = cbr(channels, channels, groups=channels)
        self.low_project = cbr(channels, channels // 2)
        self.high_project = cbr(channels, channels // 2)

    def forward(self, x):
        low = self.low(x)
        high = self.high(x - low)
        return torch.cat((self.low_project(low), self.high_project(high)), 1)


class WaveletInspired(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.filters = nn.ModuleList([nn.Conv2d(channels, channels, 2, stride=2,
                                               groups=channels, bias=False)
                                      for _ in range(4)])
        self.project = cbr(4 * channels, channels, 1)

    def forward(self, x):
        height, width = x.shape[-2:]
        # Official training sizes have even X1 dimensions. Padding handles odds.
        padded = F.pad(x, (0, width % 2, 0, height % 2), mode='replicate')
        z = self.project(torch.cat([layer(padded) for layer in self.filters], 1))
        return F.interpolate(z, size=(height, width), mode='bilinear', align_corners=False)


class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, math.ceil(channels / reduction))
        self.shared = nn.Sequential(nn.Conv2d(channels, hidden, 1, bias=False),
                                    nn.ReLU(inplace=False),
                                    nn.Conv2d(hidden, channels, 1, bias=False))

    def forward(self, x):
        weights = self.shared(F.adaptive_avg_pool2d(x, 1))
        weights = weights + self.shared(F.adaptive_max_pool2d(x, 1))
        return x * weights.sigmoid()


def initialize(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class MultiDomainBlock(nn.Module):
    def __init__(self, channels=64, branches=('spatial', 'frequency', 'wavelet')):
        super().__init__()
        if channels < 2 or channels % 2:
            raise ValueError('An even positive channel count is required.')
        factories = {'spatial': Spatial, 'frequency': Frequency, 'wavelet': WaveletInspired}
        if not branches or len(set(branches)) != len(branches) or set(branches) - set(factories):
            raise ValueError('Invalid branches.')
        self.channels = channels
        self.branches = nn.ModuleDict({name: factories[name](channels) for name in branches})
        self.attention = ChannelAttention(len(branches) * channels)
        self.gate = (nn.Sequential(cbr(len(branches) * channels, channels, 1),
                                   nn.Conv2d(channels, len(branches), 1))
                     if len(branches) > 1 else None)
        self.project = cbr(len(branches) * channels, channels, 1)
        self.apply(initialize)

    def fuse(self, x):
        attended = self.attention(torch.cat([branch(x) for branch in self.branches.values()], 1))
        gate = (self.gate(attended).softmax(dim=1) if self.gate is not None else
                attended.new_ones((x.shape[0], 1, x.shape[2], x.shape[3])))
        fused = self.project(attended)
        for i, branch in enumerate(attended.split(self.channels, dim=1)):
            fused = fused + gate[:, i:i + 1] * branch
        return fused, gate

    def forward(self, x):
        return x + self.fuse(x)[0]
