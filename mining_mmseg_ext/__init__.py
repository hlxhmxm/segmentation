"""Local MMSeg extensions for the mining road-segmentation experiments."""

from .compat import install_mmcv_ops_stubs

install_mmcv_ops_stubs()

from .backbones import MobileNetV2_BiSeNetAdapted  # noqa: E402,F401
from .datasets import automine1d  # noqa: E402,F401
from .transforms import RandomCLAHE  # noqa: E402,F401

__all__ = ['MobileNetV2_BiSeNetAdapted', 'RandomCLAHE', 'automine1d']
