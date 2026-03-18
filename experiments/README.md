# Experiments

All experiments use the same FluidWorld codebase. Install the package first:

```bash
pip install -e .
```

## Data preparation

Download UCF-101 from https://www.crcv.ucf.edu/data/UCF101.php and preprocess:

```bash
python scripts/prepare_ucf101.py --source-dir path/to/UCF-101 --out-dir data/ucf101_64 --size 64 --max-frames 150
```

## TensorBoard

All training scripts log to `runs/`. Monitor with:

```bash
tensorboard --logdir runs/
```

To overlay multiple runs (PDE vs Transformer vs ConvLSTM):

```bash
tensorboard --logdir runs/phase1_pixel:PDE,runs/phase2_transformer:Transformer,runs/phase2_convlstm:ConvLSTM
```

---

## Recommended training command (v2, FluidWorld-Delta)

```bash
python experiments/training/pixel_prediction/train_pixel.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000 --no-fatigue --var-weight 0.1 --var-target 0.3
```

Key flags:
- `--no-fatigue`: disables SynapticFatigue (causes feature collapse, Dead_Dims -> 30K)
- `--var-weight 0.1 --var-target 0.3`: forces feature variance to stay healthy
- `--no-deltanet`: disables DeltaNet temporal correction (ablation)
- `--no-titans`: disables Titans persistent memory (ablation)
- `--deep-supervision` (optional): applies loss at intermediate encoder layers

## Ablation variants

PDE only (no DeltaNet, no Titans):

```bash
python experiments/training/pixel_prediction/train_pixel.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000 --no-fatigue --var-weight 0.1 --var-target 0.3 --no-deltanet --no-titans
```

PDE + DeltaNet only (no Titans):

```bash
python experiments/training/pixel_prediction/train_pixel.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000 --no-fatigue --var-weight 0.1 --var-target 0.3 --no-titans
```

PDE + Titans only (no DeltaNet):

```bash
python experiments/training/pixel_prediction/train_pixel.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000 --no-fatigue --var-weight 0.1 --var-target 0.3 --no-deltanet
```

Transformer baseline:

```bash
python experiments/training/baseline_comparison/train_transformer.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000
```

ConvLSTM baseline:

```bash
python experiments/training/baseline_comparison/train_convlstm.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000
```

## Resume from checkpoint

```bash
python experiments/training/pixel_prediction/train_pixel.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000 --no-fatigue --var-weight 0.1 --var-target 0.3 --resume checkpoints/phase1_pixel/model_step_8000.pt
```

## PCA feature visualization

Visualize encoder dense features (V-JEPA-style PCA projection to RGB):

```bash
python tools/visualize_pca_features.py --checkpoint checkpoints/phase1_pixel/model_step_8000.pt --data-dir data/ucf101_64 --n-samples 16
```

Output: `paper/figures/pca/pca_grid.png`

Compare features across training stages by pointing to different checkpoints.

---

## Training Pipeline

Sequential experiments forming the core development pipeline.
Each step builds on the previous one.

| # | Directory | Status | Description |
|---|-----------|--------|-------------|
| 1 | [training/proprioception](training/proprioception/) | Planned | MLP baseline on proprioceptive state prediction |
| 2 | [training/pixel_prediction](training/pixel_prediction/) | Done | PDE + DeltaNet + Titans on UCF-101 (64x64) |
| 3 | [training/baseline_comparison](training/baseline_comparison/) | Done | Transformer + ConvLSTM at same param budget (~800K) |
| 4 | [training/representation_probing](training/representation_probing/) | Planned | Linear probe on frozen encoder features |
| 5 | [training/rollout_evaluation](training/rollout_evaluation/) | Planned | Multi-step autoregressive rollout stability |
| 6 | [training/gradient_planning](training/gradient_planning/) | Planned | Action optimization via backprop through PDE |
| 7 | [training/robot_deployment](training/robot_deployment/) | Planned | SO-101 real robot with LeRobot integration |

## Analysis and Ablations

Independent experiments that probe, analyze, or ablate specific properties
of the trained model. Can be run in any order once a checkpoint is available.

| Directory | Priority | Description |
|-----------|----------|-------------|
| [analysis/temporal_probe](analysis/temporal_probe/) | P0 | Does the model encode velocity and direction? |
| [analysis/surprise_signal](analysis/surprise_signal/) | P0 | Does prediction error spike at unexpected events? |
| [analysis/perturbation_analysis](analysis/perturbation_analysis/) | P1 | Are causal effects spatially localized in latent space? |
| [analysis/multiscale_rollout](analysis/multiscale_rollout/) | P1 | How does imagination quality degrade with horizon? |
| [analysis/hebbian_ablation](analysis/hebbian_ablation/) | P2 | Impact of Hebbian plasticity on learned dynamics |
| [analysis/curriculum_training](analysis/curriculum_training/) | P2 | Does structured training order improve convergence? |

## Tools

| Tool | Description |
|------|-------------|
| `tools/visualize_pca_features.py` | PCA visualization of encoder dense features (V-JEPA-style) |

---

## Roadmap

Near-term:
- Quantitative rollout metrics (SSIM, LPIPS) for PDE vs baselines
- DeltaNet vs PDE-only ablation (compare rollout h3-h5 stability)
- PCA feature visualization comparison across architectures
- Linear probing on UCF-101 action classes

Mid-term:
- Gradient planning on controlled environments
- Additional datasets (KITTI, RoboNet)
- Hebbian ablation study
- Deep supervision impact study

Long-term:
- Real robot deployment with SO-101
- Integration with LeRobot action chunking
- Scaling experiments at higher resolutions (128x128, 256x256)
