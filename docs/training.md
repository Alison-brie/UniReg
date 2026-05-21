# Training

Stage 1 trains the brain MR RPNet checkpoint:

```bash
python train.py --config configs/train/single_task/lumir_rpnet.yaml --gpu 0
```

Stage 2 trains UniReg-RPN with multi-task replay:

```bash
python train.py --config configs/train/multi_task/unireg_rpn_6task_from_brain.yaml --init_checkpoint ./logs/lumir_rpnet/best.pth --gpu 0
```
