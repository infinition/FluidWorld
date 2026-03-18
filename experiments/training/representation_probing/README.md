# Phase 2b -- Linear Probing

Evaluates the quality of learned representations by training a linear classifier on
frozen FluidWorld encoder features. Tests whether the PDE dynamics produce linearly
separable features for downstream tasks.

## Status

Planned. Requires a trained Phase 1 checkpoint.

## Prerequisites

- Trained FluidWorld checkpoint from Phase 1
- Labeled dataset (Moving MNIST `.npz` or ImageFolder format)

## Run

```bash
python experiments/training/representation_probing/linear_probe.py --checkpoint checkpoints/phase1_pixel/best.pt --data-dir data/ucf101_64 --epochs 50 --lr 1e-3
```

To generate Moving MNIST labeled data:

```bash
python experiments/training/representation_probing/generate_mnist_npz.py --out data/mnist_labeled.npz
```

## Targets

- Moving MNIST: > 80% accuracy
- UCF-101 (101 classes): > 40% accuracy (at ~800K params, 600x smaller than ViT-L)

## TensorBoard tags

`Probe/Train_Acc`, `Probe/Val_Acc`, `Probe/Loss`
