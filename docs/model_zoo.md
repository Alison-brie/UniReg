# Model Zoo

This repository exposes the following public architecture names.

| Name | Description |
|---|---|
| `unireg_rpn`   | UniReg with dynamic deformation generation on the RPN/C2F backbone |
| `unireg_iirpn` | UniReg with dynamic deformation generation on the iterative RPN backbone |
| `unireg_mlp`   | UniReg with dynamic deformation generation on the CorrMLP backbone |
| `rpnet`   | RPNet baseline |
| `iirpnet` | IIRPNet baseline |
| `corrmlp` | CorrMLP baseline |

## Released checkpoint

| Model | Training setting | Checkpoint |
|---|---|---|
| UniReg-RPN | Two-stage training: Brain MR pretraining followed by 6-task joint training | `model_zoo/unireg_rpn_6task.pth` |

The released UniReg-RPN checkpoint was originally trained with the previous internal architecture name `dyn_rpnet`. In the released UniReg codebase, the identical architecture is exposed as `unireg_rpn`. The checkpoint has been converted to a release format without optimizer states and is fully compatible with the released model.


## Preparing a release checkpoint

To convert a training checkpoint into a compact release checkpoint without optimizer states:

```bash
python tools/prepare_release_checkpoint.py \
  --input ./logs/unireg_rpn_6task_from_brain/best.pth \
  --output model_zoo/unireg_rpn_6task.pth \
  --arch unireg_rpn \
  --name "UniReg-RPN 6-task"
```

The output checkpoint contains model tensors and lightweight metadata only.
