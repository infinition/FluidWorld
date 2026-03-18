<img width="2453" height="1472" alt="Gemini_Generated_Image_t0ym24t0ym24t0ym" src="https://github.com/user-attachments/assets/649b28ad-69d4-4037-ab49-0cc2637bb302" />


# FluidWorld

A world model that replaces attention and recurrence with **reaction-diffusion PDEs**.
Latent states evolve through Laplacian diffusion and learned reaction terms on a 2D
spatial grid, giving the model a built-in inductive bias for spatial coherence,
adaptive computation depth, and O(N) scaling in the number of spatial tokens.

**Paper**: [FluidWorld: PDE-Based World Modeling via Reaction-Diffusion Dynamics](paper/fluidworld.tex)

---

## Key Results

Three architectures, same parameter budget (~800K), same data (UCF-101 64x64),
same encoder/decoder, same losses. Only the latent dynamics engine differs.

| | FluidWorld (PDE) | Transformer | ConvLSTM |
|---|---|---|---|
| **Parameters** | 801K | 801K | 802K |
| **Recon Loss** | 0.001 | 0.002 | 0.001 |
| **Pred Loss** | 0.003 | 0.004 | 0.003 |
| **Rollout coherence** | h=3+ | h=1 | h=1 |
| **Spatial complexity** | O(N) | O(N^2) | O(N*k^2) |
| **Training speed** | ~1 it/s | ~5.2 it/s | ~7.8 it/s |

Single-step scalar metrics are comparable across all three models. The critical
difference appears in multi-step rollouts: the PDE model maintains spatial coherence
to horizon h=3 and beyond, while both baselines degrade visibly at h=2. The Laplacian
diffusion operator acts as an implicit spatial regularizer that prevents exponential
error accumulation during autoregressive generation.

---

## Architecture

```
Input frame (3, 64, 64)
        |
  PatchEmbed (patch=4)
        |
  (128, 16, 16) latent grid
        |
  FluidWorldLayer2D x2          <-- reaction-diffusion PDE
  |  Laplacian diffusion             (adaptive step count)
  |  Learned reaction terms
  |  Bio mechanisms (fatigue, Hebbian)
        |
  BeliefField                    <-- temporal state (PDE evolve)
        |
  PixelDecoder
        |
  Output frame (3, 64, 64)
```

Each `FluidWorldLayer2D` integrates the PDE:

```
du/dt = D * Laplacian(u) + R(u)
```

where D is a learned diffusion coefficient and R is a nonlinear reaction network.
The number of integration steps adapts per-sample based on a learned halting signal.

---

## Quick Start

### Install

```bash
git clone https://github.com/infinition/FluidWorld.git
cd FluidWorld
pip install -e .
```

### Prepare data

Download [UCF-101](https://www.crcv.ucf.edu/data/UCF101.php) and preprocess:

```bash
python scripts/prepare_ucf101.py --source-dir path/to/UCF-101 --out-dir data/ucf101_64 --size 64 --max-frames 150
```

### Train

FluidWorld (PDE):

```bash
python experiments/training/pixel_prediction/train_pixel.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000
```

Transformer baseline:

```bash
python experiments/training/baseline_comparison/train_transformer.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000
```

ConvLSTM baseline:

```bash
python experiments/training/baseline_comparison/train_convlstm.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000
```

### Monitor

```bash
tensorboard --logdir runs/
```

To overlay all three runs:

```bash
tensorboard --logdir runs/phase1_pixel:PDE,runs/phase2_transformer:Transformer,runs/phase2_convlstm:ConvLSTM
```

### Test

```bash
python -m pytest tests/ -v
```

---

## Repository Structure

```
FluidWorld/
├── fluidworld/              Core package
│   └── core/
│       ├── world_model_v2.py          PDE world model
│       ├── fluid_world_layer.py       Reaction-diffusion layer
│       ├── belief_field.py            Temporal state (PDE-evolved)
│       ├── bio_mechanisms.py          Fatigue, Hebbian plasticity
│       ├── transformer_world_model.py Transformer baseline
│       ├── convlstm_world_model.py    ConvLSTM baseline
│       ├── decoder.py                 Pixel decoder
│       ├── video_dataset.py           Data loading
│       └── ...
├── experiments/
│   ├── training/
│   │   ├── pixel_prediction/    PDE world model on UCF-101
│   │   ├── baseline_comparison/ Transformer + ConvLSTM (same params)
│   │   ├── rollout_evaluation/  Multi-step autoregressive rollout
│   │   ├── gradient_planning/   Action optimization through PDE
│   │   └── ...
│   └── analysis/
│       ├── temporal_probe/      Velocity/direction encoding
│       ├── surprise_signal/     Prediction error at events
│       ├── hebbian_ablation/    Plasticity mechanism impact
│       └── ...
├── scripts/                 Data preparation
├── tools/                   Visualization and inspection
├── tests/                   Smoke tests
└── paper/                   LaTeX source and figures
```

See [experiments/README.md](experiments/README.md) for detailed protocols and
the full roadmap.

---

## Experiment Pipeline

The experiments follow a sequential pipeline. Each step builds on the previous one.

```
1. training/proprioception        MLP baseline on robot proprio (planned, needs robot data)
       |
2. training/pixel_prediction      PDE world model on UCF-101 video        [DONE]
       |
3. training/baseline_comparison   Transformer + ConvLSTM at same params   [DONE]
       |
4. training/representation_probing   Linear probe on frozen features
       |
5. training/rollout_evaluation       Multi-step autoregressive stability
       |
6. training/gradient_planning        Action optimization via backprop through PDE
       |
7. training/robot_deployment         Closed-loop control on SO-101 robot
```

Independent analysis experiments (can be run after step 2 with a trained checkpoint):

```
analysis/temporal_probe         Does the model encode velocity and direction?
analysis/surprise_signal        Does prediction error spike at unexpected events?
analysis/perturbation_analysis  Are causal effects spatially localized?
analysis/multiscale_rollout     How does imagination quality degrade with horizon?
analysis/hebbian_ablation       Impact of Hebbian plasticity on dynamics
analysis/curriculum_training    Does structured training order help convergence?
```

---

## Generating Paper Figures

Regenerate all figures from TensorBoard logs:

```bash
python paper/generate_figures.py
```

Output: `paper/figures/*.pdf` (used by `paper/fluidworld.tex`).

---

## Citation

```bibtex
@article{polly2026fluidworld,
    title={FluidWorld: PDE-Based World Modeling via Reaction-Diffusion Dynamics},
    author={Polly, Fabien},
    year={2026},
    note={Preprint}
}
```

---

## License

MIT. See [LICENSE](LICENSE).
