# Evaluation

Use `evaluate.py` for all testing and runtime reporting. Example:

```bash
python evaluate.py --config configs/eval/eval_brain_unireg_rpn.yaml --checkpoint ./logs/unireg_rpn_6task_from_brain/best.pth --mode eval --gpu 0 --output_dir ./logs/unireg_rpn_6task_from_brain/
```


```bash
python evaluate.py --config configs/eval/eval_liver_unireg_rpn.yaml --checkpoint model_zoo/unireg_rpn_6task.pth --mode eval --gpu 0
```
