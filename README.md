<img width="2453" height="1472" alt="Gemini_Generated_Image_t0ym24t0ym24t0ym" src="https://github.com/user-attachments/assets/649b28ad-69d4-4037-ab49-0cc2637bb302" />

# FluidWorld

A world model built on **reaction-diffusion PDEs**, **DeltaNet temporal correction**,
and **Titans persistent memory**. Latent states evolve through Laplacian diffusion
and learned reaction terms on a 2D spatial grid, while DeltaNet provides content-based
error correction and Titans maintains persistent scene knowledge across rollouts.

O(N) scaling, no KV-cache, 867K parameters.

**Paper**: [FluidWorld: PDE-Based World Modeling via Reaction-Diffusion Dynamics](paper/fluidworld.tex)

---

## Architecture History

### v1 -- PDE Only (published)

Three architectures, same parameter budget (~800K), same data (UCF-101 64x64),
same encoder/decoder, same losses. Only the latent dynamics engine differs.

| | FluidWorld (PDE) | Transformer | ConvLSTM |
|---|---|---|---|
| **Parameters** | 801K | 801K | 802K |
| **Recon Loss** | 0.001 | 0.002 | 0.001 |
| **Pred Loss** | 0.003 | 0.004 | 0.003 |
| **Rollout coherence** | h=3+ | h=1 | h=1 |
| **Spatial complexity** | O(N) | O(N^2) | O(N*k^2) |

### v2 -- FluidWorld-Delta (current)

The BeliefField (temporal state) is augmented with two new mechanisms:

| Component | Role | Params | Complexity |
|-----------|------|--------|------------|
| **PDE** (Laplacian diffusion + reaction) | Spatial coherence | 133K | O(N) |
| **DeltaNet** (delta rule temporal correction) | Content-based error correction | 66K | O(N*d^2) train, O(d^2) inference |
| **Titans** (persistent memory) | Scene structure persistence | 82K | O(d) |

Total model: **867K parameters** (+8% over v1). No KV-cache, constant inference memory.

The three components have non-overlapping roles:
- **PDE** answers "how does the world change?" (diffusion, motion)
- **DeltaNet** answers "where did my prediction go wrong?" (error correction)
- **Titans** answers "what is the world?" (persistent object/scene memory)

---

## Architecture

```
Input frame (3, 64, 64)
        |
  PatchEmbed (patch=4)
        |
  (128, 16, 16) latent grid
        |
  FluidWorldLayer2D x3          <-- spatial PDE
  |  Multi-scale Laplacian           (adaptive step count)
  |  Learned reaction terms
  |  Bio mechanisms (Hebbian)
        |
  BeliefField                    <-- temporal dynamics
  |  write(state, observation)       Gated state update
  |  evolve(state, stimulus)         PDE + DeltaNet + Titans
  |  read(state)                     Spatial/vector readout
        |
  PixelDecoder
        |
  Output frame (3, 64, 64)
```

The `evolve()` step integrates the dual-ODE:

```
du/dt = Laplacian_diffusion(u)    -- spatial coherence (PDE)
      + DeltaNet(u)               -- temporal error correction
      + Reaction(u)               -- semantic nonlinearity
      + alpha * Titans_memory     -- persistent scene context
      + stimulus                  -- external action input
```

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

FluidWorld-Delta (PDE + DeltaNet + Titans):

```bash
python experiments/training/pixel_prediction/train_pixel.py --data-dir data/ucf101_64 --epochs 200 --batch-size 16 --bptt-steps 4 --max-steps 6 --lr 3e-4 --max-batches-per-epoch 2000 --no-fatigue --var-weight 0.1 --var-target 0.3
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

### Visualize dense features (PCA)

After training, visualize encoder feature quality:

```bash
python tools/visualize_pca_features.py --checkpoint checkpoints/phase1_pixel/model_step_8000.pt --data-dir data/ucf101_64 --n-samples 16
```

Output: `paper/figures/pca/pca_grid.png` (top row: original frames, bottom row: PCA of encoder features mapped to RGB).

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
│       ├── belief_field.py            BeliefField (PDE + DeltaNet + Titans)
│       ├── bio_mechanisms.py          Hebbian plasticity, lateral inhibition
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
├── tools/                   Visualization (PCA features, etc.)
├── tests/                   Smoke tests
└── paper/                   LaTeX source and figures
```

See [experiments/README.md](experiments/README.md) for detailed protocols and
the full roadmap.

---

## Experiment Pipeline

The experiments follow a sequential pipeline. Each step builds on the previous one.

```
1. training/proprioception        MLP baseline on robot proprio (planned)
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

## Changelog

### v2 (FluidWorld-Delta) -- March 2026

- **DeltaNet temporal correction** in BeliefField: content-based error-driven
  state update using the delta rule (linear attention with error correction).
  Replaces blind PDE-only temporal dynamics with learned correction.
- **Titans persistent memory**: fast-weight associative memory that stores scene
  structure and updates online at inference. Replaces MemoryPump.
- **SynapticFatigue disabled by default**: was causing feature collapse
  (Dead_Dims reaching 30K/32K). Use `--no-fatigue` flag.
- **PCA feature visualization**: `tools/visualize_pca_features.py` for
  V-JEPA-style dense feature quality inspection.
- **Standalone**: all PDE modules (diffusion, FluidLayer2D, PatchEmbed) are now
  bundled in `fluidworld/core/`. No external FluidVLA dependency required.
- **Variance regularization**: `--var-weight` / `--var-target` flags to prevent
  feature collapse without relying on SynapticFatigue.

### v1 (PDE Only) -- February 2026

- Initial release: reaction-diffusion PDE world model.
- 3-way ablation (PDE vs Transformer vs ConvLSTM) at matched ~800K params.
- Bio mechanisms: SynapticFatigue, LateralInhibition, HebbianDiffusion.
- BeliefField with PDE-based temporal evolution.
- Multi-step rollout evaluation up to h=5.

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
