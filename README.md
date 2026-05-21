# UniReg

**UniReg** is a universal medical image registration framework with dynamic deformation generation. This clean release focuses on the UniReg main model and two lightweight baselines: **RPNet** and **CorrMLP**.

## Highlights

- **UniReg with dynamic deformation generation** for multi-task CT/MR registration.
- Public UniReg model names: `unireg_rpn`, `unireg_iirpn`, and `unireg_mlp`.
- Baseline models: `rpnet`, `iirpnet`, and `corrmlp`.
- Two-stage training: Brain MR pretraining followed by 6-task joint training.
- SAMCoarse/SAME `coarse_phi` pre-alignment support.
- Evaluation with Dice, TRE, Jacobian folding ratio, SDlogJ, and runtime reporting.

## Repository structure

```text
UniReg/
  train.py                  # training entry
  evaluate.py               # evaluation/runtime entry
  model_zoo/                # released checkpoints
  configs/
    datasets/               # dataset profiles
    train/                  # training configs
    eval/                   # evaluation configs
    baselines/              # baseline configs
  unireg/                     # core implementation
  tools/                    # utility and sanity-check scripts
  docs/                     # documentation
```

## Installation

```bash
pip install -r requirements.txt
```

A CUDA-enabled PyTorch environment is recommended for 3D registration experiments.

## Data root

Set `DATA_ROOT` to the root directory of the prepared datasets:

```bash
export DATA_ROOT=/path/to/RegDataUnified
```

Dataset profiles are defined in:

```text
configs/datasets/datasets.yaml
```

## Quick start

### Stage 1: Brain MR pretraining

```bash
python train.py --config configs/train/single_task/lumir_rpnet.yaml --gpu 0
```

### Stage 2: 6-task UniReg-RPN joint training

```bash
python train.py \
  --config configs/train/multi_task/unireg_rpn_6task_from_brain.yaml \
  --init_checkpoint ./logs/lumir_rpnet/best.pth \
  --gpu 0
```

### Evaluation with the released checkpoint

```bash
python evaluate.py \
  --config configs/eval/eval_brain_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --mode eval \
  --gpu 0
```

## Model zoo

| Model | Setting | Checkpoint |
|---|---|---|
| UniReg-RPN | Brain MR pretraining + 6-task joint training | `model_zoo/unireg_rpn_6task.pth` |

The released `unireg_rpn_6task.pth` checkpoint was originally trained with the internal architecture name `dyn_rpnet`. In this clean release, the same architecture is exposed as `unireg_rpn`. The checkpoint has been converted to a release format without optimizer states and is fully compatible with the released model.

Verify checkpoint compatibility with:

```bash
python tools/check_checkpoint_compat.py \
  --config configs/eval/eval_brain_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --device cpu
```

Expected output:

```text
Matched parameter tensors: 56/56 (100.00%)
Checkpoint compatibility check passed.
```

## Documentation

- `docs/installation.md`: environment setup.
- `docs/data_preparation.md`: dataset organization and `DATA_ROOT` usage.
- `docs/training.md`: two-stage training commands.
- `docs/evaluation.md`: evaluation and runtime testing.
- `docs/config.md`: config organization.
- `docs/model_zoo.md`: released checkpoints.
- `docs/reproduce_results.md`: commands for reproducing results.
- `docs/prealignment.md`: SAMCoarse/SAME pre-alignment.

## Sanity checks

```bash
python tools/check_clean_repo.py
python tools/check_config_load.py
python tools/check_checkpoint_compat.py \
  --config configs/eval/eval_brain_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --device cpu
```

## Citation

If you find UniReg helpful for your research and applications, please cite our paper:

```bibtex
@article{li2026unireg,
  title   = {UniReg: A Universal Model for Controllable CT Medical Image Registration},
  author  = {Li, Zi and Zhang, Jianpeng and Ma, Tai and Mok, Tony C. W. and Zhou, Yan-Jie and Chen, Zeli and Ye, Xianghua and Lu, Le and Chen, Cheng and Jin, Dakai},
  year    = {2026}
}
```
