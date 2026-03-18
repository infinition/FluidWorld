# Phase 2a -- Parameter-Matched Baselines

Three-way ablation study. Trains Transformer and ConvLSTM baselines with the same
parameter budget (~800K), same data, same encoder/decoder, and same losses as FluidWorld.
Only the latent dynamics engine differs.

## Status

Done. All three models trained to step 8000 on UCF-101 64x64.

## Results (step 8000)

| Model | Params | Recon Loss | Pred Loss | Speed |
|-------|--------|------------|-----------|-------|
| FluidWorld (PDE) | 801K | 0.001 | 0.003 | ~1 it/s |
| Transformer | 801K | 0.002 | 0.004 | ~5.2 it/s |
| ConvLSTM | 802K | 0.001 | 0.003 | ~7.8 it/s |

Single-step scalars are comparable, but multi-step rollouts diverge significantly:
PDE maintains spatial coherence to horizon h=3+, while both baselines degrade at h=2.

## Run

Transformer baseline:

```bash
python experiments/training/baseline_comparison/train_transformer.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000
```

ConvLSTM baseline:

```bash
python experiments/training/baseline_comparison/train_convlstm.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000
```

## Outputs

- Checkpoints: `checkpoints/phase2_transformer/`, `checkpoints/phase2_convlstm/`
- TensorBoard: `runs/phase2_transformer/`, `runs/phase2_convlstm/`

## TensorBoard overlay

```bash
tensorboard --logdir runs/phase1_pixel:PDE,runs/phase2_transformer:Transformer,runs/phase2_convlstm:ConvLSTM
```
