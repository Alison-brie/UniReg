# Reproduce Results

This document summarizes the recommended commands for reproducing UniReg-RPN results with the cleaned codebase.

## 1. Prepare data

Set the dataset root before training or evaluation:

```bash
export DATA_ROOT=/path/to/RegDataUnified
```

The dataset profiles are defined in:

```text
configs/datasets/datasets.yaml
```

## 2. Stage-1 Brain MR pretraining

```bash
python train.py --config configs/train/single_task/lumir_rpnet.yaml --gpu 0
```

The default checkpoint path is:

```text
./logs/lumir_rpnet/best.pth
```

## 3. Stage-2 6-task UniReg-RPN joint training

```bash
python train.py \
  --config configs/train/multi_task/unireg_rpn_6task_from_brain.yaml \
  --init_checkpoint ./logs/lumir_rpnet/best.pth \
  --gpu 0
```

The default checkpoint path is:

```text
./logs/unireg_rpn_6task_from_brain/best.pth
```

## 4. Task-specific evaluation

### Brain MR

```bash
python evaluate.py \
  --config configs/eval/eval_brain_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --mode eval \
  --gpu 0
```

### Abdomen CT

```bash
python evaluate.py \
  --config configs/eval/eval_abdomen_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --mode eval \
  --gpu 0
```

### Chest CT

```bash
python evaluate.py \
  --config configs/eval/eval_chest_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --mode eval \
  --gpu 0
```

### HeadNeck CT

```bash
python evaluate.py \
  --config configs/eval/eval_headneck_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --mode eval \
  --gpu 0
```

### Liver CT

```bash
python evaluate.py \
  --config configs/eval/eval_liver_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --mode eval \
  --gpu 0
```

### Cardiac MR / ACDC

```bash
python evaluate.py \
  --config configs/eval/eval_acdc_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --mode eval \
  --gpu 0
```

### Abdominal-DIR-QA TRE evaluation

```bash
python evaluate.py \
  --config configs/eval/eval_dirqa_unireg_rpn.yaml \
  --checkpoint model_zoo/unireg_rpn_6task.pth \
  --mode eval \
  --gpu 0
```

