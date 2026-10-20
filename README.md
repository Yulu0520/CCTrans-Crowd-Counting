# CCTrans: Simplifying and Improving Crowd Counting with Transformer(Code reproduction)
* Code reproduction
* Original paper [Link](https://arxiv.org/pdf/2109.14483.pdf)

## Overview
* The results below reproduce ShanghaiTech Part A with the original DM-Count loss (TV mode).
* Training supports both the original TV loss and Gaussian L2 density supervision.
* ShanghaiTech Part A 

| Code      | MAE   | MSE      |
|-----------|-------|-------|
| PAPER     | 54.8  | 86.6  |
| This code | 54.20 | 88.97 |

Our code reaches this result with the standard hyperparameter set in code. Trained with batch-size=8 for around 1500 epoch(as said in the paper). Best validation at around epoch 606
# code framework
* adopt code of DM-Count.
* [link](https://github.com/cvlab-stonybrook/DM-Count)

# Training
Take a look at the arguments accepted by ```train.py```
* update root "data-dir" in ./train.py.
* load pretrained weights of ImageNet-1k in ./Networks/ALTGVT.py.
* pretrained weights [link](https://drive.google.com/file/d/1um39wxIaicmOquP2fr_SiZdxNCUou8w-/view)
* [new] Added [wandb](https://wandb.ai/) integration. If you want to log with wandb, set ```--wandb 1``` in ```train.py``` after having logged in to wandb (```wandb login``` in console)
* launch with ```python train.py```

## Density loss selection

Selecting `--density-loss l2` replaces TV with MSE against an adaptive Gaussian
density target, while retaining L1 counting loss and point-based OT loss:

`loss = L1(predicted_count, head_count) + wot * OT + wl2 * MSE(predicted_density, gaussian_target)`

To use the weights in CCTrans equation (6), run:

```bash
python train.py --data-dir /path/to/SHA --dataset sha --density-loss l2 --wot 0.01 --wl2 1.0
```

To isolate the TV-to-L2 replacement, keep the existing OT weight (`--wot 0.1`).
The default OT weight remains 0.1; selecting L2 does not change it automatically.
To train with the original TV objective and default weights:

```bash
python train.py --data-dir /path/to/SHA --dataset sha --density-loss tv --wot 0.1 --wtv 0.01
```

Gaussian targets are generated after resize/crop, at image resolution. Each head's
sigma is `--gaussian-beta` (default 0.3) times its mean distance to up to three other
heads in the crop. A single head uses `--gaussian-sigma` (default 15 image pixels).
Sigma is at least one pixel, including for duplicate points. Kernels extend to four
sigmas and are renormalized after clipping at image boundaries, so each head
contributes one count. Empty crops produce zero targets. Targets are downsampled
by 8x8 block sums and flipped with the image; their sum remains the crop head count.
These kernel, boundary and sparse-point rules are explicit implementation choices;
the CCTrans loss section does not specify all these details.

MSE averages over batch, channel and spatial dimensions, using unnormalized
density maps without TV's count normalization or count multiplier. `--wl2` only
affects L2 mode; `--wtv` only affects TV mode. L1 counting and OT still use the
original head coordinates/counts. All training datasets support both modes.
The paper additionally uses Bayesian loss for SHB and QNRF; this implementation
retains OT for those datasets, so it does not reproduce those dataset-specific recipes.

Progress, logs and W&B report the selected loss. Checkpoint directories include
the mode, L2 weight and Gaussian settings; adapter checkpoints also store `train_config`.
When resuming, CLI settings control the loss; saved settings are not automatically
restored. Starting a new run is recommended when comparing objectives.

Run focused target, transform, gradient and CPU training-batch checks with:

```bash
python -m unittest discover -s tests -v
```

## Independent day/night DoRA

The baseline is the existing Twins-L CCTrans counting model. Two parallel
adapters named **dora-day** and **dora-night** modify Q/V weights in all 24
backbone attention blocks. Local attention uses packed QKV; global attention
uses separate Q and packed KV. K, projection biases, MLPs, convolutions and the
regression head are shared baseline parameters. The adapters have independent
low-rank A/B matrices and output-vector magnitudes. Exactly one route runs per
forward; both adapters use the same raw baseline, rather than being chained.

Start from a **trained CCTrans counting baseline** (`.pth` weights or an epoch
`.tar`). ImageNet backbone weights alone are not a trained counting baseline.
Existing full training remains available with `--stage baseline` (the default).
The current TV default and loss weights remain unchanged; specify L2 explicitly
if desired. DoRA does not change crop, flip or loss calculations.

```bash
# Optional: train a new full baseline using the existing training options.
python train.py --stage baseline --data-dir /path/to/SHA/mix \
  --density-loss l2 --wot 0.01 --wl2 1.0

# Train only the day adapter.
python train.py --stage dora-day --dataset sha \
  --baseline-checkpoint /path/to/trained_CCTrans.pth \
  --dora-day-train-dir /path/to/SHA/day/train \
  --dora-day-val-dir /path/to/SHA/day/val \
  --dora-day-rank 4 --dora-day-alpha 4 --dora-day-lr 2e-4 \
  --dora-day-epochs 100 --density-loss l2 --wot 0.01 --wl2 1.0

# Train only the night adapter against exactly the same baseline checkpoint.
python train.py --stage dora-night --dataset sha \
  --baseline-checkpoint /path/to/trained_CCTrans.pth \
  --dora-night-train-dir /path/to/SHA/night/train \
  --dora-night-val-dir /path/to/SHA/night/val \
  --dora-night-rank 4 --dora-night-alpha 4 --dora-night-lr 2e-4 \
  --dora-night-epochs 100 --density-loss l2 --wot 0.01 --wl2 1.0
```

To configure experiments directly in source, edit the `default=` values in
`train.py` → `add_training_arguments()`. The **Day adapter** and **Night adapter**
sections are separate: each defines its own rank, alpha, learning rate, weight
decay, epochs, validation schedule, data paths and checkpoint. Set `--stage`'s
`default=` and the baseline/data paths there to launch with just `python train.py`.
Command-line options still override these defaults. For adapter stages, edit the
branch's learning rate and schedule rather than the generic baseline options.

Each branch has these independent settings (replace `NAME` with `dora-day` or
`dora-night`):

| Option | Default | Purpose |
| --- | --- | --- |
| `--NAME-rank` | 4 | Low-rank dimension |
| `--NAME-alpha` | 4.0 | Update scaling, alpha/rank |
| `--NAME-lr` | 2e-4 | Active adapter learning rate |
| `--NAME-weight-decay` | 1e-4 | Active adapter weight decay |
| `--NAME-epochs` | 10 | Total stage epoch count |
| `--NAME-val-epoch` | 1 | Validation interval |
| `--NAME-val-start` | 0 | First eligible validation epoch index |
| `--NAME-train-dir` | required | Training split directory |
| `--NAME-val-dir` | required | Validation split directory |
| `--NAME-checkpoint` | empty | Existing adapter weights for fresh fine-tuning |

Adapter epochs are displayed with the existing zero-based convention: 100 epochs
runs indices 0 through 99. Validation starts are also zero-based. Adapter schedules
and learning rates override the generic `--max-epoch`, `--val-*`, `--lr`, and
`--weight-decay` options. For `custom`, each split directory is a list-file root
containing `train.list` or `valid.list`, respectively, using the existing format.

During adapter training, the optimizer receives only that adapter's parameters.
The shared baseline and the other adapter are frozen. The baseline stays in eval
mode so BatchNorm running statistics, dropout and stochastic depth cannot change
its behavior during adaptation. The active adapter starts with a zero direction
update and baseline weight magnitudes, preserving the initial baseline output
within floating-point tolerance.

Outputs are separated under:

```text
<save-dir>/<experiment>/baseline/<timestamp>/
  best_model_mae-5.00_epoch-12.pth
  baseline_last_epoch-15.ckpt
<save-dir>/<experiment>/dora-day/<timestamp>/
  dora-day_best_mae-4.50_epoch-20.pth
  dora-day_last_epoch-25.ckpt
<save-dir>/<experiment>/dora-night/<timestamp>/
  dora-night_best_mae-6.00_epoch-15.pth
  dora-night_last_epoch-18.ckpt
```

Each run also stores its `config.json` and training log. **Each stage/run retains
one best `.pth` and one latest `.ckpt`**, with best weights selected by the lowest validation MAE. A new best
is saved successfully before the previous best from that run is deleted; equal
or worse MAE does not trigger a save. MAE is shown to two decimals in the filename,
while selection uses the full-precision value. Filename epochs are **1-based**
(`epoch-1` is the first training epoch); internal epoch metadata and the existing
progress display remain zero-based. Existing files from older runs are not removed.

`--save-dir` defaults to `ckpts`. Baseline `.pth` files contain the full model;
day/night `.pth` files contain only the selected adapter plus its settings and
baseline identity. A separate `<stage>_last_epoch-N.ckpt` is written at the end
of every epoch (after validation if it runs). It contains current weights,
optimizer state, completed epoch, best metrics, training configuration, the best
weights path and Python/NumPy/PyTorch/CUDA random states. Baseline CKPTs contain
the full model; adapter CKPTs contain only the active branch and baseline identity.
The new CKPT is saved successfully before deleting the previous CKPT from that
run. New runs do not create epoch `.tar` or `last_checkpoint.tar` files.
Before the first eligible validation there is no best `.pth`, but the latest
`.ckpt` is still available. Keep the trained baseline checkpoint: adapter weights
and adapter CKPTs both need it. DoRA CKPTs do not contain the other adapter.

Adapter files record their name, rank/alpha, exact attention target list,
loss/training configuration and the baseline file's SHA-256 and path. Loading
against a different baseline, branch, rank/alpha or target layout fails explicitly.

To start new fine-tuning from the best adapter, add
`--dora-day-checkpoint /path/to/dora-day_best_mae-4.50_epoch-20.pth` (or the night
equivalent) to the corresponding training command. Optimizer state, epoch and
best metrics start fresh. Reuse its original rank/alpha; changing rank requires
a new adapter. Specify the desired loss options explicitly.

To resume from the latest completed epoch, set `--resume` in `train.py` to the
`.ckpt` path, or pass it on the command line. For example, add this to the original
day training command:

```bash
--resume /path/to/dora-day_last_epoch-25.ckpt --dora-day-epochs 100
```

This continues from the 26th epoch, up to 100 total day-stage epochs. Use the same
stage, rank/alpha, data and loss configuration. CLI learning rate and weight decay
can override the optimizer's saved settings. Best metrics and random states are
restored, and setup rejects a CKPT belonging to a different stage. A DoRA resume
file supplies its saved baseline path if `--baseline-checkpoint` is omitted;
provide that option if the baseline file moved. Old optimizer-containing `.tar`
checkpoints remain supported, but they may not include random states.

Random states are restored after model/optimizer setup, before the next epoch's
data iteration. Reproduction across different hardware or worker configurations
is not guaranteed; restoring a saved CUDA random state requires the same number
of visible GPUs. Each launch creates a new timestamp directory and leaves source
checkpoints untouched. The CKPT records the prior best `.pth` path, so that best
remains available in the source run if a resumed run has not improved it yet.

### Testing and visualization with DoRA

Both `test_image_patch.py` and `vis_densityMap.py` support:

- `--adapter-route auto|baseline|dora-day|dora-night`
- `--baseline-checkpoint PATH` to override the saved baseline location after moving files
- `--dora-day-checkpoint PATH` / `--dora-night-checkpoint PATH` to load either or both adapters

To configure testing directly in source, edit the `default=` values together near
the top of `test_image_patch.py`: `--model-path` for the baseline weights,
`--dora-day-checkpoint`, `--dora-night-checkpoint`, `--data-path`, and
`--adapter-route`. Choose `dora-day` or `dora-night` explicitly to evaluate that
adapter, then run `python test_image_patch.py`. The optional
`--baseline-checkpoint` is only needed as an override when the main model path
itself points to an adapter file. Command-line arguments override these defaults.

`auto` selects the adapter named in the main model checkpoint, or the baseline
for a plain full-model checkpoint. A requested day/night route requires that
trained adapter to be loaded; it never silently uses a fresh random adapter.
Inference reads rank/alpha from adapter metadata.

```bash
# Main model path is an adapter: auto selects dora-day.
python test_image_patch.py --dataset sha --data-path /path/to/SHA \
  --model-path /path/to/dora-day/dora-day_best_mae-5.00_epoch-12.pth \
  --baseline-checkpoint /path/to/trained_CCTrans.pth

# Load both independently and select night explicitly.
python test_image_patch.py --dataset sha --data-path /path/to/SHA \
  --model-path /path/to/trained_CCTrans.pth \
  --dora-day-checkpoint /path/to/dora-day/dora-day_best_mae-5.00_epoch-12.pth \
  --dora-night-checkpoint /path/to/dora-night/dora-night_best_mae-6.00_epoch-15.pth \
  --adapter-route dora-night
```

For visualization, use the same adapter options with the existing `--weight_path`
argument in place of `--model-path`. Visualization retains a fixed route for the
entire run; image-level routing below is implemented in `test_image_patch.py`.

### Mixed-set day/night routing

Edit the following defaults directly in `test_image_patch.py`, or pass them on
the command line:

| Setting | Behavior |
| --- | --- |
| `--router-mode fixed` | Existing behavior: `--adapter-route` selects one route for the run |
| `--router-mode filename` | Select day/night DoRA separately for each original image |
| `--router-mode classifier` | Reserved for a future image router; currently raises a clear error |
| `--day-prefixes IMG_ day_` | Day filename prefixes, case-insensitive |
| `--night-prefixes night_IMG_ night_` | Night filename prefixes, case-insensitive |
| `--unknown-route error` | Stop and identify an unrecognized filename (default) |
| `--unknown-route baseline` | Use the baseline for an unrecognized filename |

In filename mode, both trained adapter files are required. `--adapter-route` is
only used in fixed mode. A name matching both day and night rules always raises
an error, including when unknown-name fallback is enabled. Prefixes classify
the basename, not the parent directories. Both adapters are loaded once, checked
against the exact baseline, and switched in memory. A route is chosen **once per
original image before patching**; all its patches use the same branch.

For example, with `/path/to/SHA/mix/val/*.jpg` and matching `.npy` annotations:

```bash
python test_image_patch.py --dataset sha \
  --data-path /path/to/SHA/mix --split val \
  --model-path /path/to/baseline.pth \
  --dora-day-checkpoint /path/to/dora-day_best_mae-4.50_epoch-20.pth \
  --dora-night-checkpoint /path/to/dora-night_best_mae-6.00_epoch-15.pth \
  --router-mode filename --output-dir test_results
```

`--split val` selects the validation directory under `--data-path`; `--split train`
can evaluate the training split without performing training. `--split ''` reads
the supplied directory directly. Omitting the option keeps the original layouts:
SHA/SHB `test_data`, QNRF `test`, NWPU `val`, and the existing custom test list root.
The ShanghaiTech loader also supports its original images/ground-truth layout.
`--device cpu` and `--num-workers 0` allow CPU checks without loader subprocesses.

Each test run writes a new timestamp directory under `--output-dir`:

- `routing.csv`: image identifier, filename-derived domain, actual route,
  router source, confidence (currently blank), GT count, predicted count and
  signed error (`GT - prediction`).
- `metrics.csv`: count, MAE and RMSE for all/day/night/unknown images. Empty
  groups have blank metrics. Day/night groups use filename labels in both fixed
  and filename modes, enabling comparisons on the same mixed data.
- `ALGVT_sha_test.txt`: legacy-style per-image counts, now inside the results
  directory so separate runs do not overwrite one another.

The future router extension point is `make_router()` in
`utils/inference_router.py`. A router implements `predict(image, filename)` and
returns `RouteDecision(route, domain, source, confidence)`. `image` is the complete
normalized dataset tensor before patches are generated. A future classifier can
consume its pixels and return a selected branch and confidence without changing
the density prediction/stitching code. No classifier is trained or loaded yet.

# Testing
* python test_image_patch.py
* Due to crop training with size of 256x256, the validation image is divided into several patches with size of 256x256, and the overlapping area is averaged.
* Download the pretrained model from Baidu-Disk(提取码: se59) [link](https://pan.baidu.com/s/16qY_cFIUAUaDRsdr5vNsWQ)

# Visualization
* python vis_densityMap.py
* save to ./vis/part_A_final

# Environment
	See requirements.txt
	


