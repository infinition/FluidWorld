<img width="2453" height="1472" alt="Gemini_Generated_Image_t0ym24t0ym24t0ym" src="https://github.com/user-attachments/assets/649b28ad-69d4-4037-ab49-0cc2637bb302" />

# FluidWorld

### A world model that replaces attention with physics. And self-repairs its own predictions.

FluidWorld is a proof-of-concept world model built on reaction-diffusion PDEs. No attention. No KV-cache. The Laplacian operator *is* the computation. And it produces something no Transformer or ConvLSTM has demonstrated: **autopoietic self-repair**, where the model spontaneously corrects its own prediction errors through the physics of diffusion.

Trained with a latent prediction objective, the PDE substrate maintains cosine similarity of 0.827 across 19 autoregressive steps. It predicts in abstract representation space, not pixels, and encodes dynamics non-linearly.

800K parameters. One consumer GPU. Results that challenge the assumption that attention is necessary for world modeling.

> **Paper:** [Reaction-Diffusion Dynamics as a Predictive Substrate for World Models](paper/fluidworld.tex)

---

## Why This Matters

### 1. The model heals itself

During autoregressive rollout, prediction quality degrades. Every world model has this problem. But FluidWorld does something no attention-based model does: after degrading, it **recovers**.

Over 500 rollouts on Moving MNIST, SSIM drops from 0.778 to 0.287 by step 6, then climbs back to 0.508 by step 9. Two thirds of all rollouts show this recovery. The statistics leave no room for doubt:

| | |
|---|---|
| Recovery rate | **66.8%** (334 / 500 rollouts) |
| p-value | **1.67 x 10^-49** |
| Cohen's d | **0.739** (medium-to-large effect) |
| Mechanism | Laplacian diffusion dissipates high-frequency errors |

Transformers and ConvLSTMs only decay. They never recover. The difference is structural: the Laplacian is a low-pass filter that smooths prediction errors at every integration step. Blurry intermediate predictions are not a defect. They are the self-repair mechanism.

### 2. Destroy half the internal state. It comes back.

Corrupt 50% of the BeliefField (the model's persistent latent state) during rollout. Inject noise, zero out channels, shuffle spatial positions. The model recovers in 3 to 7 steps. No explicit repair mechanism. No retraining. The Laplacian fills corrupted regions via diffusion from intact neighbors, and RMSNorm re-normalizes amplitude.

Push corruption to 90%. The model still works. No cliff effect. Graceful degradation all the way up. This kind of robustness has implications for deployed world models operating under sensor noise or partial observability.

### 3. Stable latent prediction across 19 autoregressive steps

Training with a JEPA-style latent objective (target encoder + VICReg, inspired by Y. LeCun's JEPA framework) produces abstract representations that barely degrade over time:

| Rollout step | Cosine similarity |
|---|---|
| Step 1 | 0.833 |
| Step 10 | 0.827 |
| Step 19 | 0.827 |

For comparison, Pixel and Random models sit near zero. The PDE dynamics produce stable internal predictions without any pixel-level supervision.

An MLP probe confirms the representations encode velocity non-linearly (R^2 jumps from 0.29 to 0.60), while Pixel and Random both *drop* under the same probe. The JEPA-style PDE is the only model whose dynamics encoding *improves* with a non-linear readout.

### 4. O(N) vs O(N^2): the gap grows with resolution

| Resolution | Attention ops | PDE diffusion ops | Ratio |
|---|---|---|---|
| 16x16 (256 tokens) | 65K | 256 | 256x |
| 64x64 (4K tokens) | 16.7M | 4K | 4,096x |
| 128x128 (16K tokens) | 268M | 16K | **16,384x** |

At current resolution the difference is negligible. At the resolutions real-world robotics and autonomous driving require, PDE diffusion is orders of magnitude cheaper than attention.

### 5. Three-way ablation: same parameters, same data, different substrate

All three models: ~800K parameters, identical encoder, identical decoder, identical losses, identical data (UCF-101 64x64). The *only* difference is the predictive engine.

| | FluidWorld (PDE) | Transformer | ConvLSTM |
|---|---|---|---|
| Recon Loss | **0.001** | 0.002 | 0.001 |
| Pred Loss | 0.003 | 0.004 | 0.003 |
| Spatial Std | **1.16** | 1.05 | 1.12 |
| Effective Rank | **~20K** | ~16.5K | ~19K |
| Rollout coherence | **h=3** | h=1 | h=1 |
| Spatial complexity | **O(N)** | O(N^2) | O(N) |

Single-step metrics are intentionally comparable. That was the point: isolate what happens when you chain predictions. The PDE maintains spatial structure 1 to 2 steps longer than both baselines. The ConvLSTM has spatial bias too (convolutions), yet it fails just as fast as the Transformer. Spatial bias alone is not enough. Continuous diffusion with global reach is what matters.

---

## Architecture

```
Input frame (3, 64, 64)
       |
 PatchEmbed (stride 4)
       |
 (128, 16, 16) latent grid
       |
 FluidLayer2D x3               Reaction-diffusion PDE
 |  Multi-scale Laplacian          dilations {1, 4, 16}
 |  Learned reaction MLP           position-wise nonlinearity
 |  Bio mechanisms                 Hebbian, lateral inhibition, fatigue
       |
 BeliefField                    Persistent temporal state
 |  write(observation)             GRU-gated integration
 |  evolve(state)                  PDE + DeltaNet + Titans
 |  read(state)                    Spatial readout
       |
 PixelDecoder
       |
 Output frame (3, 64, 64)
```

One equation, two roles. The same reaction-diffusion PDE governs both spatial encoding (in the encoder layers) and temporal prediction (in the BeliefField). The core update:

```
u(t+1) = u(t) + dt * [D * laplacian(u) + Reaction(u) + Memory_terms]
```

What falls out naturally: O(N) computation, adaptive early stopping, spatial coherence via continuous diffusion, and self-repair as a free byproduct of the Laplacian low-pass filter.

The BeliefField adds biologically-inspired mechanisms: lateral inhibition (retinal processing), synaptic fatigue (prevents channel collapse), and Hebbian diffusion (co-activated pathways strengthen). The system operates at the edge of chaos (Lyapunov exponent 0.0033), bounded by RMSNorm homeostasis.

---

## The Fluid Architecture Lineage

FluidWorld is the third iteration of a research program replacing attention with reaction-diffusion PDEs across modalities:

- **FluidLM (2024)** : Language modeling with 1D Laplacian on token sequences
- **FluidVLA (2025)** : Vision and robotic control with 2D Laplacian (40ms inference on RTX 4070). Extended to 3D for medical imaging (CT/MRI segmentation)
- **FluidWorld (2026)** : World modeling. The PDE is now both encoder and temporal predictor

Same core equation. Same O(N) complexity. Three modalities.

---

## Experiments

15 numbered Jupyter notebooks, ordered for sequential execution. Training notebooks (01 to 03) need a GPU. Analysis notebooks (04 to 15) load saved checkpoints.

| # | Notebook | What it does |
|---|----------|-------------|
| 01 | `01_train_moving_mnist` | Train pixel-prediction model (30 epochs) |
| 02 | `02_train_jepa_mnist` | Train JEPA-style latent prediction (30 epochs) |
| 03 | `03_train_edgefreq_fair` | Edge/Freq ablation from scratch (fair comparison) |
| 04 | `04_test_symmetry_breaking` | Spontaneous pattern formation from uniform field |
| 05 | `05_test_transition_phase` | Phase diagram: 225 (D, dt) configurations |
| 06 | `06_test_resilience` | Perturbation recovery across corruption types |
| 07 | `07_test_memory_titans` | Persistent memory: occlusion and recovery |
| 08 | `08_test_energy_conservation` | Energy dynamics, RMSNorm, numerical stability |
| 09 | `09_quantify_autopoiesis` | Autopoietic recovery (N=500, p < 10^-49) |
| 10 | `10_perturbation_demo` | Visual: corrupt at step 5, watch self-repair |
| 11 | `11_ablation_edgefreq` | Edge/Freq vs Laplacian-only (60 vs 30 epochs) |
| 12 | `12_eval_edgefreq_fair` | Fair comparison: both 30 epochs. Edge/Freq collapses. |
| 13 | `13_linear_probes` | Position, velocity, direction probes |
| 14 | `14_quick_test` | Pixel vs JEPA vs Random: full comparison |
| 15 | `15_demo_interactive_rollout` | Interactive rollout visualization |

---

## Reproducing the Results

Clone the repository and run the numbered notebooks in `experiments/` sequentially. Training notebooks (01 to 03) require a GPU (tested on RTX 4070 Ti, ~3.5h for 30 epochs). Analysis notebooks (04 to 15) load saved checkpoints and run on CPU or GPU.

Full setup instructions and dataset preparation are documented inside each notebook.

---

## Repository

- `fluidworld/core/` : model source code (PDE, BeliefField, baselines, decoder)
- `experiments/` : 15 numbered notebooks + saved analysis data
- `paper/` : LaTeX source, all figures, bibliography

## Citation

```bibtex
@article{polly2026fluidworld,
    title={FluidWorld: Reaction-Diffusion Dynamics as a Predictive Substrate for World Models},
    author={Polly, Fabien},
    year={2026},
    note={Preprint}
}
```

## License

[CC-BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Free for research. Attribution required. No commercial use without permission. Derivatives must share alike.

See [LICENSE](LICENSE).
