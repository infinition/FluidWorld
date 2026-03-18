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

## Training Pipeline

Sequential experiments forming the core development pipeline.
Each step builds on the previous one.

| # | Directory | Status | Description |
|---|-----------|--------|-------------|
| 1 | [training/proprioception](training/proprioception/) | Planned | MLP baseline on proprioceptive state prediction |
| 2 | [training/pixel_prediction](training/pixel_prediction/) | Done | PDE world model trained on UCF-101 (64x64) |
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

---

## Roadmap

Near-term:
- Quantitative rollout metrics (SSIM, LPIPS) for the 3-way comparison
- Linear probing on UCF-101 action classes
- Temporal dynamics probing on UCF-101 data

Mid-term:
- Gradient planning on controlled environments
- Additional datasets (KITTI, RoboNet)
- Hebbian ablation study

Long-term:
- Real robot deployment with SO-101
- Integration with LeRobot action chunking
- Scaling experiments at higher resolutions (128x128, 256x256)
