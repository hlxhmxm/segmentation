# Local Reproduction Notes (Windows / GTX 1650 / 4GB VRAM)

## What Was Added

- `mining_mmseg_ext/`
  - Registers the custom `automine1d` dataset.
  - Registers `RandomCLAHE`.
  - Registers `MobileNetV2_BiSeNetAdapted`.
  - Installs lightweight `mmcv.ops` stubs so the project can run with `mmcv-lite`.
- `scripts/run_reproduction.py`
  - Loads a paper config.
  - Fixes broken dataset paths for this repository.
  - Replaces `SyncBN` with `BN`.
  - Redirects missing MobileNet pretrained weights to `torchvision://mobilenet_v2`.
  - Freezes BatchNorm automatically when `batch_size=1`.
  - Supports `train` and `test` modes on `val`, `lens_soiling`, and `sun_glare`.

## Environment Used

- Python `3.11`
- PyTorch `2.1.2+cu121`
- torchvision `0.16.2+cu121`
- mmengine `0.10.4`
- mmcv-lite `2.0.1`
- mmsegmentation `1.2.2`
- numpy `1.26.4`
- opencv-python `4.10.0.84`

## Commands Run

### Smoke test

```powershell
.\.venv\Scripts\python scripts\run_reproduction.py `
  --config mmsegmentation\configs\Resnet\Resnet_lr0.001_Clahe_photo.py `
  --mode train `
  --batch-size 1 `
  --num-workers 0 `
  --max-iters 1 `
  --val-interval 1 `
  --checkpoint-interval 1 `
  --work-dir work_dirs\smoke_resnet
```

### Small-stage training

```powershell
.\.venv\Scripts\python scripts\run_reproduction.py `
  --config mmsegmentation\configs\Resnet\Resnet_lr0.001_Clahe_photo.py `
  --mode train `
  --batch-size 1 `
  --num-workers 0 `
  --max-iters 100 `
  --val-interval 50 `
  --checkpoint-interval 50 `
  --work-dir work_dirs\resnet100
```

### Evaluation

```powershell
.\.venv\Scripts\python scripts\run_reproduction.py `
  --config mmsegmentation\configs\Resnet\Resnet_lr0.001_Clahe_photo.py `
  --mode test `
  --domain val `
  --checkpoint work_dirs\resnet100\best_mIoU_iter_100.pth `
  --batch-size 1 `
  --num-workers 0 `
  --work-dir work_dirs\resnet100_eval_val
```

```powershell
.\.venv\Scripts\python scripts\run_reproduction.py `
  --config mmsegmentation\configs\Resnet\Resnet_lr0.001_Clahe_photo.py `
  --mode test `
  --domain lens_soiling `
  --checkpoint work_dirs\resnet100\best_mIoU_iter_100.pth `
  --batch-size 1 `
  --num-workers 0 `
  --work-dir work_dirs\resnet100_eval_lens
```

```powershell
.\.venv\Scripts\python scripts\run_reproduction.py `
  --config mmsegmentation\configs\Resnet\Resnet_lr0.001_Clahe_photo.py `
  --mode test `
  --domain sun_glare `
  --checkpoint work_dirs\resnet100\best_mIoU_iter_100.pth `
  --batch-size 1 `
  --num-workers 0 `
  --work-dir work_dirs\resnet100_eval_sun
```

## Results Observed

### After 1 iteration

- Validation `mIoU = 41.53`

### After 20 iterations

- Validation `mIoU = 51.26`
- Lens soiling `mIoU = 63.38`
- Sun glare `mIoU = 54.05`

### After 100 iterations

- Validation `mIoU = 60.61`
- Lens soiling `mIoU = 66.47`
- Sun glare `mIoU = 57.13`

## Interpretation

- The repository is now executable on this machine.
- The training curve already shows that the clean validation split improves as training proceeds.
- A mild degradation gap has appeared for `sun_glare` at 100 iterations.
- `lens_soiling` is currently scoring higher than the clean validation split, which does **not** match the paper and is likely caused by:
  - the repository / FigShare release not matching the paper's full target-domain test set,
  - the very small target-domain sample count available here,
  - the current run being only `100 / 40000` iterations,
  - the need to freeze BatchNorm for `batch_size=1` on this 4GB GPU.

## Important Caveats

- The published "optimal checkpoint" files on FigShare are placeholders, not real model weights.
- The current public target-domain data is much smaller than the paper describes.
- The paper uses a much stronger hardware setup (`A100 80GB`) and `batch_size=4`.
- This local setup is suitable for staged reproduction, debugging, and trend verification, but not for claiming exact paper-level numbers yet.
