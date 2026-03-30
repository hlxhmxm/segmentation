"""Custom transforms referenced by the paper configs."""

from __future__ import annotations

import numpy as np

from mmseg.datasets.transforms import CLAHE
from mmseg.registry import TRANSFORMS


@TRANSFORMS.register_module()
class RandomCLAHE(CLAHE):
    """Apply CLAHE with a configurable probability."""

    def __init__(self, prob: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.prob = prob

    def transform(self, results):
        if np.random.rand() < self.prob:
            return super().transform(results)
        return results
