"""The selector network.

Deliberately small. The label is positive on about 6% of the 6% of pixels that lie on a
contour, so capacity is not the binding constraint - precision in the top few percent is -
and a 1.93M network already shows overfitting past epoch 17 on 795 frames.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def block(a: int, b: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
        nn.Conv2d(b, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """Four levels, one output logit per pixel."""

    def __init__(self, in_ch: int = 5, base: int = 32):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8]
        self.e1, self.e2, self.e3 = block(in_ch, c[0]), block(c[0], c[1]), block(c[1], c[2])
        self.mid = block(c[2], c[3])
        self.u3 = nn.ConvTranspose2d(c[3], c[2], 2, 2)
        self.d3 = block(c[2] * 2, c[2])
        self.u2 = nn.ConvTranspose2d(c[2], c[1], 2, 2)
        self.d2 = block(c[1] * 2, c[1])
        self.u1 = nn.ConvTranspose2d(c[1], c[0], 2, 2)
        self.d1 = block(c[0] * 2, c[0])
        self.out = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        m = self.mid(F.max_pool2d(e3, 2))
        d3 = self.d3(torch.cat([self.u3(m), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)
