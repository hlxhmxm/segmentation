from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import Runner


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mining_mmseg_ext  # noqa: E402,F401
from mmseg.utils import register_all_modules  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train or evaluate the mining road-segmentation configs.')
    parser.add_argument('--config', required=True, help='Path to a paper config.')
    parser.add_argument('--mode', choices=['train', 'test'], required=True)
    parser.add_argument(
        '--domain',
        choices=['val', 'lens_soiling', 'sun_glare'],
        default='val',
        help='Evaluation split to use in test mode.')
    parser.add_argument('--checkpoint', help='Checkpoint for test mode.')
    parser.add_argument('--work-dir', default='work_dirs/local_run')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--max-iters', type=int)
    parser.add_argument('--val-interval', type=int)
    parser.add_argument('--checkpoint-interval', type=int)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


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


def replace_missing_pretrained(obj):
    if isinstance(obj, dict):
        checkpoint = obj.get('checkpoint')
        if isinstance(checkpoint, str):
            lowered = checkpoint.lower()
            if 'mobilenet_v2' in lowered:
                obj['checkpoint'] = 'torchvision://mobilenet_v2'
            elif lowered.startswith('/checkpoint/') or lowered.startswith(
                    'checkpoint/'):
                obj['checkpoint'] = None
        for value in obj.values():
            replace_missing_pretrained(value)
    elif isinstance(obj, list):
        for item in obj:
            replace_missing_pretrained(item)


def build_test_pipeline():
    return [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(2048, 1024), keep_ratio=True),
        dict(type='LoadAnnotations'),
        dict(type='PackSegInputs'),
    ]


def patch_dataloader(dataset_cfg, data_root: Path, img_path: str,
                     seg_map_path: str, pipeline):
    dataset_cfg.type = 'automine1d'
    dataset_cfg.data_root = str(data_root)
    dataset_cfg.data_prefix = dict(img_path=img_path, seg_map_path=seg_map_path)
    dataset_cfg.pipeline = copy.deepcopy(pipeline)


def patch_cfg(cfg: Config, args: argparse.Namespace) -> Config:
    clean_root = PROJECT_ROOT / 'data' / 'automine1d'
    lens_root = PROJECT_ROOT / 'data' / 'automine1d_distortion' / 'lens_soiling'
    sun_root = PROJECT_ROOT / 'data' / 'automine1d_distortion' / 'sun_glare'

    cfg.custom_imports = dict(
        imports=['mining_mmseg_ext'], allow_failed_imports=False)
    cfg.default_scope = 'mmseg'
    cfg.launcher = 'none'
    cfg.work_dir = str((PROJECT_ROOT / args.work_dir).resolve())
    cfg.randomness = dict(seed=args.seed)
    cfg.resume = False
    cfg.load_from = None if args.mode == 'train' else args.checkpoint

    replace_syncbn(cfg._cfg_dict)
    replace_missing_pretrained(cfg._cfg_dict)

    if hasattr(cfg, 'env_cfg'):
        cfg.env_cfg.cudnn_benchmark = False

    patch_dataloader(
        cfg.train_dataloader.dataset,
        clean_root,
        'images/train',
        'annotations/train',
        cfg.train_dataloader.dataset.pipeline,
    )
    cfg.train_dataloader.batch_size = args.batch_size
    cfg.train_dataloader.num_workers = args.num_workers
    cfg.train_dataloader.persistent_workers = False

    val_pipeline = build_test_pipeline()
    patch_dataloader(
        cfg.val_dataloader.dataset,
        clean_root,
        'images/val',
        'annotations/val',
        val_pipeline,
    )
    cfg.val_dataloader.batch_size = 1
    cfg.val_dataloader.num_workers = args.num_workers
    cfg.val_dataloader.persistent_workers = False

    if args.mode == 'test':
        if args.domain == 'val':
            test_root = clean_root
            img_path = 'images/val'
            seg_path = 'annotations/val'
        elif args.domain == 'lens_soiling':
            test_root = lens_root
            img_path = 'images'
            seg_path = 'annotations'
        else:
            test_root = sun_root
            img_path = 'images'
            seg_path = 'annotations'
        patch_dataloader(
            cfg.test_dataloader.dataset,
            test_root,
            img_path,
            seg_path,
            val_pipeline,
        )
    else:
        cfg.test_dataloader = copy.deepcopy(cfg.val_dataloader)

    cfg.test_dataloader.batch_size = 1
    cfg.test_dataloader.num_workers = args.num_workers
    cfg.test_dataloader.persistent_workers = False
    cfg.test_evaluator = dict(
        type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])
    cfg.val_evaluator = dict(
        type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'])

    if args.max_iters is not None:
        cfg.train_cfg.max_iters = args.max_iters
    if args.val_interval is not None:
        cfg.train_cfg.val_interval = args.val_interval
    if args.checkpoint_interval is not None:
        cfg.default_hooks.checkpoint.interval = args.checkpoint_interval

    cfg.default_hooks.logger.interval = min(
        getattr(cfg.default_hooks.logger, 'interval', 100), 10)

    if args.device.startswith('cuda') and not os.environ.get('CUDA_VISIBLE_DEVICES'):
        os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

    return cfg


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)

    # MMEngine's Windows compiler probe can fail on non-UTF8 output. Falling
    # back to a lightweight environment summary keeps training/test runnable.
    import mmengine.runner.runner as runner_module  # noqa: WPS433

    original_collect_env = runner_module.collect_env

    def safe_collect_env():
        try:
            return original_collect_env()
        except UnicodeDecodeError:
            return {
                'PyTorch': torch.__version__,
                'CUDA available': str(torch.cuda.is_available()),
                'CUDA version': str(torch.version.cuda),
                'Platform': sys.platform,
            }

    runner_module.collect_env = safe_collect_env

    cfg = Config.fromfile(str((PROJECT_ROOT / args.config).resolve()))
    cfg = patch_cfg(cfg, args)

    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    cfg.dump(work_dir / 'resolved_config.py')

    runner = Runner.from_cfg(cfg)

    if args.batch_size == 1:
        def freeze_batch_norm(module: nn.Module) -> None:
            for child in module.modules():
                if isinstance(child, nn.modules.batchnorm._BatchNorm):
                    child.eval()
                    if child.weight is not None:
                        child.weight.requires_grad_(False)
                    if child.bias is not None:
                        child.bias.requires_grad_(False)

        original_train = runner.model.train

        def patched_train(mode: bool = True):
            model = original_train(mode)
            freeze_batch_norm(runner.model)
            return model

        runner.model.train = patched_train
        freeze_batch_norm(runner.model)

    if args.mode == 'train':
        runner.train()
    else:
        if not args.checkpoint:
            raise ValueError('--checkpoint is required in test mode.')
        runner.test()


if __name__ == '__main__':
    main()
