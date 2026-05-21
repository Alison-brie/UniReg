# Iterative Inference with IIRPN

UniReg-RPN is trained as a residual pyramid registration network under the standard one-pass training setting. During inference, however, the same trained UniReg-RPN checkpoint can be evaluated in either a single-pass manner or an iterative inference manner.

This design follows the inference strategy introduced in **IIRP-Net: Iterative Inference Residual Pyramid Network for Enhanced Image Registration** (CVPR 2024). In IIRP-Net, the model is not necessarily retrained with an iterative objective. Instead, iterative inference repeatedly applies a trained residual registration network to progressively refine the moving image toward the fixed image.

## Single-pass RPN inference

In the default RPN setting, the model predicts a deformation field once.