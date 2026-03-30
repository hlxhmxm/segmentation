"""Compatibility helpers for running MMSeg on a lightweight Windows setup."""

from __future__ import annotations

import sys
import types

import torch.nn as nn


def install_mmcv_ops_stubs() -> None:
    """Provide import-time stubs when ``mmcv-full`` is unavailable.

    The configs we run here do not rely on MMCV CUDA ops, but MMSeg imports a
    few optional modules eagerly. These stubs let the package import cleanly
    under ``mmcv-lite`` while still failing loudly if an unsupported op is
    actually used at runtime.
    """

    if 'mmcv.ops' in sys.modules:
        return

    ops = types.ModuleType('mmcv.ops')

    def _unavailable(*args, **kwargs):
        raise NotImplementedError(
            'This experiment is running with mmcv-lite, so mmcv CUDA ops are '
            'not available on this machine.')

    class _UnavailableModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            _unavailable(*args, **kwargs)

    ops.point_sample = _unavailable
    ops.sigmoid_focal_loss = _unavailable
    ops.DeformConv2d = _UnavailableModule
    ops.ModulatedDeformConv2d = _UnavailableModule
    ops.CrissCrossAttention = _UnavailableModule
    ops.PSAMask = _UnavailableModule

    sys.modules['mmcv.ops'] = ops
