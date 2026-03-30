"""Dataset registrations required by the paper configs."""

from mmseg.datasets.basesegdataset import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class automine1d(BaseSegDataset):
    METAINFO = {
        'classes': ['background', 'road'],
        'palette': [[0, 0, 0], [0, 128, 0]],
    }

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.png', seg_map_suffix='.png', **kwargs)
