from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from mmengine.config import Config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mining_mmseg_ext  # noqa: E402,F401
from mmseg.apis import inference_model, init_model  # noqa: E402
from mmseg.utils import register_all_modules  # noqa: E402


def replace_syncbn(obj):
    if isinstance(obj, dict):
        for key, value in list(obj.items()):
            if key == 'type' and value == 'SyncBN':
                obj[key] = 'BN'
            else:
                replace_syncbn(value)
    elif isinstance(obj, list):
        for item in obj:
            replace_syncbn(item)


def load_model():
    register_all_modules(init_default_scope=True)
    config_path = PROJECT_ROOT / 'mmsegmentation' / 'configs' / 'Resnet' / 'Resnet_lr0.001_Clahe_photo.py'
    checkpoint_path = PROJECT_ROOT / 'work_dirs' / 'resnet100' / 'best_mIoU_iter_100.pth'

    cfg = Config.fromfile(str(config_path))
    cfg.custom_imports = dict(
        imports=['mining_mmseg_ext'], allow_failed_imports=False)
    replace_syncbn(cfg._cfg_dict)

    if 'backbone' in cfg.model and 'backbone_cfg' in cfg.model.backbone:
        cfg.model.backbone.backbone_cfg.init_cfg = None

    model = init_model(cfg, str(checkpoint_path), device='cuda:0')
    model.cfg = cfg
    return model


def mask_to_color(mask: np.ndarray) -> np.ndarray:
    color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    color[mask > 0] = np.array([0, 200, 0], dtype=np.uint8)
    return color


def overlay_mask(image_bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    color = mask_to_color(mask)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1.0 - alpha, color, alpha, 0.0)


def make_panel(ax, image, title: str):
    ax.imshow(image, cmap='gray' if image.ndim == 2 else None)
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def main():
    model = load_model()

    samples = [
        (
            'Clean Validation',
            PROJECT_ROOT / 'data' / 'automine1d' / 'images' / 'val' / 'd46844c5-1661922943.png',
            PROJECT_ROOT / 'data' / 'automine1d' / 'annotations' / 'val' / 'd46844c5-1661922943.png',
        ),
        (
            'Lens Soiling',
            PROJECT_ROOT / 'data' / 'automine1d_distortion' / 'lens_soiling' / 'images' / '1ff0a5a2-1662344207.000000.png',
            PROJECT_ROOT / 'data' / 'automine1d_distortion' / 'lens_soiling' / 'annotations' / '1ff0a5a2-1662344207.000000.png',
        ),
        (
            'Sun Glare',
            PROJECT_ROOT / 'data' / 'automine1d_distortion' / 'sun_glare' / 'images' / '01a82c20-1663920811.000000.png',
            PROJECT_ROOT / 'data' / 'automine1d_distortion' / 'sun_glare' / 'annotations' / '01a82c20-1663920811.000000.png',
        ),
    ]

    output_dir = PROJECT_ROOT / 'results' / 'visualizations' / 'resnet100'
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(len(samples), 4, figsize=(16, 11))

    for row, (label, image_path, gt_path) in enumerate(samples):
        image_bgr = cv2.imread(str(image_path))
        gt_mask = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
        pred = inference_model(model, str(image_path))
        pred_mask = pred.pred_sem_seg.data[0].detach().cpu().numpy().astype(np.uint8)

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        gt_overlay = overlay_mask(image_bgr, gt_mask > 0)
        pred_overlay = overlay_mask(image_bgr, pred_mask > 0)
        pred_color = mask_to_color(pred_mask > 0)

        make_panel(axes[row, 0], image_rgb, f'{label}\nOriginal')
        make_panel(axes[row, 1], gt_overlay, 'Ground Truth Overlay')
        make_panel(axes[row, 2], pred_overlay, 'Prediction Overlay')
        make_panel(axes[row, 3], pred_color, 'Prediction Mask')

        single_fig, single_axes = plt.subplots(1, 4, figsize=(16, 4))
        make_panel(single_axes[0], image_rgb, f'{label}\nOriginal')
        make_panel(single_axes[1], gt_overlay, 'Ground Truth Overlay')
        make_panel(single_axes[2], pred_overlay, 'Prediction Overlay')
        make_panel(single_axes[3], pred_color, 'Prediction Mask')
        single_fig.tight_layout()
        single_fig.savefig(output_dir / f'{label.lower().replace(" ", "_")}.png', dpi=150)
        plt.close(single_fig)

    fig.tight_layout()
    fig.savefig(output_dir / 'summary.png', dpi=150)
    plt.close(fig)

    print(output_dir / 'summary.png')


if __name__ == '__main__':
    main()
