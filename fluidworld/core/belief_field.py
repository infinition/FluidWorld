"""
belief_field.py -- FluidWorld-Delta: PDE + DeltaNet + Titans persistent memory.

Architecture:
  - PDE (Laplacian diffusion + reaction): spatial coherence (unchanged)
  - DeltaNet temporal correction: content-based error-driven state update
  - Titans persistent memory: fast-weight memory that adapts at inference
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._fluidvla_imports import Laplacian2D, ReactionMLP, RMSNorm
from .bio_mechanisms import HebbianDiffusion


# ---------------------------------------------------------------------------
# DeltaNet: Linear attention with delta rule error correction
# ---------------------------------------------------------------------------

class DeltaNetTemporal(nn.Module):
    """Delta Rule temporal correction for 2D spatial features.

    Instead of blindly interpolating states (GRU), DeltaNet:
      1. Predicts what the state should be (key-value association)
      2. Compares with the actual state (error = target - prediction)
      3. Corrects the internal memory proportionally to the error

    This is the temporal equivalent of "I thought the ball would be here,
    but it's actually there, so I update my model accordingly."

    Operates on flattened spatial tokens: (B, H*W, C) -> (B, H*W, C)
    """

    def __init__(self, channels: int, n_heads: int = 4):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.n_heads = n_heads
        self.head_dim = channels // n_heads

        # Q, K, V projections
        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)

        # Beta: per-head learning rate for delta rule (how much to correct)
        self.beta_proj = nn.Linear(channels, n_heads, bias=False)

        # Output projection
        self.out_proj = nn.Linear(channels, channels, bias=False)
        self.norm = RMSNorm(channels)

        # Persistent state: S = key-value association matrix
        # Shape per head: (head_dim, head_dim), stored externally
        self._S = None

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=0.1)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=0.1)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize the key-value association matrix S to zeros."""
        self._S = torch.zeros(
            batch_size, self.n_heads, self.head_dim, self.head_dim,
            device=device, dtype=dtype,
        )

    def detach_state(self):
        if self._S is not None:
            self._S = self._S.detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply delta rule correction to spatial features.

        Fully parallel: no Python loop. S updates once per frame
        by aggregating over all spatial positions.

        Args:
            x: (B, N, C) flattened spatial features

        Returns:
            (B, N, C) corrected features
        """
        B, N, C = x.shape
        H, D = self.n_heads, self.head_dim

        q = self.q_proj(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        k = self.k_proj(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)
        v = self.v_proj(x).view(B, N, H, D).transpose(1, 2)  # (B, H, N, D)

        # Normalize keys
        k = F.elu(k) + 1.0
        k = k / (k.norm(dim=-1, keepdim=True) + 1e-6)

        # Per-position learning rate
        beta = torch.sigmoid(self.beta_proj(x))  # (B, N, H)
        beta = beta.permute(0, 2, 1).unsqueeze(-1)  # (B, H, N, 1)

        if self._S is None:
            self.init_state(B, x.device, x.dtype)

        S = self._S * 0.95  # decay

        # -- READ: query memory for all positions in parallel --
        y = torch.einsum("bhij,bhnj->bhni", S, q)  # (B, H, N, D)

        # -- WRITE: update S with aggregated delta from all positions --
        pred = torch.einsum("bhij,bhnj->bhni", S, k)  # (B, H, N, D)
        error = v - pred  # (B, H, N, D)
        # Aggregate: mean over spatial positions, weighted by beta
        weighted_error = (beta * error).mean(dim=2)  # (B, H, D)
        weighted_keys = k.mean(dim=2)  # (B, H, D)
        delta_S = torch.einsum("bhi,bhj->bhij", weighted_error, weighted_keys)
        S = (S + delta_S).clamp(-10.0, 10.0)

        self._S = S.detach()

        y = y.transpose(1, 2).contiguous().view(B, N, C)
        return self.out_proj(self.norm(y))


# ---------------------------------------------------------------------------
# Titans: Persistent memory with online weight adaptation
# ---------------------------------------------------------------------------

class TitansMemory(nn.Module):
    """Titans-style persistent memory that learns at inference time.

    Maintains a fast-weight memory matrix M that stores scene structure.
    Unlike MemoryPump (which is a simple gate-value accumulator),
    Titans actively writes and retrieves from an associative memory
    that updates its weights based on surprise (prediction error).

    This gives FluidWorld persistent knowledge of "what the world is"
    (object templates, background structure) while the PDE handles
    "how the world changes" (motion, diffusion).
    """

    def __init__(self, channels: int, memory_slots: int = 32):
        super().__init__()
        self.channels = channels
        self.memory_slots = memory_slots

        # Write: project input to key-value for memory storage
        self.write_key = nn.Linear(channels, channels, bias=False)
        self.write_value = nn.Linear(channels, channels, bias=False)

        # Read: project input to query for memory retrieval
        self.read_query = nn.Linear(channels, channels, bias=False)

        # Surprise gate: how much to update memory (high surprise = high update)
        self.surprise_gate = nn.Sequential(
            nn.Linear(channels * 2, channels),
            nn.Sigmoid(),
        )

        # Memory learning rate (learned, per-channel)
        self.eta = nn.Parameter(torch.full((channels,), 0.1))

        # Persistent memory state
        self._M = None  # (B, memory_slots, channels) -- the memory bank
        self._M_keys = None  # (B, memory_slots, channels) -- keys for retrieval

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.write_key.weight, gain=0.1)
        nn.init.xavier_uniform_(self.write_value.weight, gain=0.1)
        nn.init.xavier_uniform_(self.read_query.weight, gain=0.1)

    def init_state(self, batch_size: int, device: torch.device, dtype: torch.dtype):
        """Initialize memory bank to small random values."""
        self._M = torch.randn(
            batch_size, self.memory_slots, self.channels,
            device=device, dtype=dtype,
        ) * 0.01
        self._M_keys = torch.randn(
            batch_size, self.memory_slots, self.channels,
            device=device, dtype=dtype,
        ) * 0.01

    def detach_state(self):
        if self._M is not None:
            self._M = self._M.detach()
            self._M_keys = self._M_keys.detach()

    def forward(
        self,
        h_global: torch.Tensor,
        pooled_features: torch.Tensor,
    ) -> torch.Tensor:
        """Read from and write to persistent memory.

        Args:
            h_global: (B, C) current global state
            pooled_features: (B, C) spatially-pooled current features

        Returns:
            (B, C) memory-augmented global state
        """
        B, C = h_global.shape

        if self._M is None:
            self.init_state(B, h_global.device, h_global.dtype)

        # -- READ: retrieve relevant memories --
        query = self.read_query(h_global)  # (B, C)
        # Attention over memory slots
        attn_logits = torch.einsum("bc,bmc->bm", query, self._M_keys)  # (B, M)
        attn_weights = F.softmax(attn_logits / (C ** 0.5), dim=-1)  # (B, M)
        retrieved = torch.einsum("bm,bmc->bc", attn_weights, self._M)  # (B, C)

        # -- SURPRISE: how unexpected is the current observation? --
        # High surprise = memory doesn't predict current features well
        surprise_input = torch.cat([retrieved, pooled_features], dim=-1)  # (B, 2C)
        surprise = self.surprise_gate(surprise_input)  # (B, C)

        # -- WRITE: update memory with current observation --
        write_k = self.write_key(pooled_features)  # (B, C)
        write_v = self.write_value(pooled_features)  # (B, C)

        # Find most similar slot and update it (soft addressing)
        write_attn = torch.einsum("bc,bmc->bm", write_k, self._M_keys)  # (B, M)
        write_weights = F.softmax(write_attn / (C ** 0.5), dim=-1)  # (B, M)

        # Update memory: M += eta * surprise * (value - retrieved) @ weights
        eta = torch.sigmoid(self.eta)  # (C,)
        update = (eta * surprise * (write_v - retrieved)).unsqueeze(1)  # (B, 1, C)
        self._M = self._M + write_weights.unsqueeze(-1) * update
        self._M_keys = self._M_keys + write_weights.unsqueeze(-1) * (write_k.unsqueeze(1) - self._M_keys) * 0.01

        # Detach to prevent backprop through memory (online learning)
        self._M = self._M.detach()
        self._M_keys = self._M_keys.detach()

        # -- OUTPUT: combine current state with retrieved memory --
        return h_global + retrieved


# ---------------------------------------------------------------------------
# BeliefField with DeltaNet + Titans
# ---------------------------------------------------------------------------

class BeliefField(nn.Module):
    """
    Persistent 2D latent field with DeltaNet temporal correction and Titans memory.

    Architecture:
        write(state, obs)  -> gated state update
        evolve(state, stim) -> PDE dynamics + DeltaNet correction + Titans memory
        read(state) -> extract representation

    The PDE provides spatial coherence (physics prior).
    DeltaNet provides content-based temporal correction (delta rule).
    Titans provides persistent scene memory (online learning).
    """

    def __init__(
        self,
        channels: int,
        stimulus_dim: int = 6,
        spatial_hw: int = 16,
        decay: float = 0.95,
        n_evolve_steps: int = 3,
        dilations: list = [1, 4],
        use_memory_pump: bool = True,
        use_hebbian: bool = True,
        hebbian_lr: float = 0.01,
        hebbian_decay: float = 0.99,
        use_deltanet: bool = True,
        use_titans: bool = True,
    ):
        super().__init__()

        self.channels = channels
        self.spatial_hw = spatial_hw
        self.n_evolve_steps = n_evolve_steps
        self.use_memory_pump = use_memory_pump
        self.use_hebbian = use_hebbian
        self.use_deltanet = use_deltanet
        self.use_titans = use_titans

        # -- Write mechanism --
        self.gate_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.value_proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.Tanh(),
        )

        # -- Internal PDE dynamics (spatial coherence) --
        self.diffusion = Laplacian2D(
            channels=channels,
            dilations=dilations,
            signed_diffusion=False,
            diffusion_scale=0.25,
        )
        self.reaction = ReactionMLP(channels)
        self.norm = RMSNorm(channels)

        # -- DeltaNet temporal correction --
        if use_deltanet:
            self.deltanet = DeltaNetTemporal(channels, n_heads=4)

        # -- Titans persistent memory (replaces MemoryPump) --
        if use_titans:
            self.titans = TitansMemory(channels, memory_slots=32)
            self.alpha_pump = nn.Parameter(torch.tensor(0.1))
        self._h_global = None

        # -- Hebbian Diffusion --
        if use_hebbian:
            self.hebbian = HebbianDiffusion(
                channels=channels,
                spatial_hw=spatial_hw,
                hebbian_lr=hebbian_lr,
                hebbian_decay=hebbian_decay,
            )

        # -- External stimulus injection --
        self.stimulus_proj = nn.Sequential(
            nn.Linear(stimulus_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

        # -- Read mechanism --
        self.read_proj = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
        )

        # -- Learned parameters --
        self.log_decay = nn.Parameter(torch.log(torch.tensor(decay)))
        self.log_dt = nn.Parameter(torch.log(torch.tensor(0.1)))

        self._init_small()

    def _init_small(self):
        with torch.no_grad():
            self.stimulus_proj[-1].weight.mul_(0.01)
            self.stimulus_proj[-1].bias.zero_()

    @property
    def decay(self) -> torch.Tensor:
        return self.log_decay.exp().clamp(0.5, 0.99)

    @property
    def dt(self) -> torch.Tensor:
        return self.log_dt.exp().clamp(0.01, 0.3)

    def init_state(
        self,
        batch_size: int,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        # Initialize all persistent states
        self._h_global = torch.zeros(batch_size, self.channels, device=device, dtype=dtype)
        if self.use_deltanet:
            self.deltanet.init_state(batch_size, device, dtype)
        if self.use_titans:
            self.titans.init_state(batch_size, device, dtype)

        return torch.zeros(
            batch_size, self.channels,
            self.spatial_hw, self.spatial_hw,
            device=device, dtype=dtype,
        )

    def detach_hidden(self):
        if self._h_global is not None:
            self._h_global = self._h_global.detach()
        if self.use_deltanet:
            self.deltanet.detach_state()
        if self.use_titans:
            self.titans.detach_state()

    def write(
        self,
        state: torch.Tensor,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        B, C, H_b, W_b = state.shape

        if observation.dim() == 4:
            obs = F.adaptive_avg_pool2d(observation, (H_b, W_b))
        elif observation.dim() == 2:
            obs = observation.unsqueeze(-1).unsqueeze(-1).expand(B, C, H_b, W_b)
        else:
            raise ValueError(
                f"observation must be (B, C, H, W) or (B, C), got shape {observation.shape}"
            )

        obs_flat = obs.permute(0, 2, 3, 1).reshape(B * H_b * W_b, C)
        gate = self.gate_proj(obs_flat).reshape(B, H_b, W_b, C).permute(0, 3, 1, 2)
        value = self.value_proj(obs_flat).reshape(B, H_b, W_b, C).permute(0, 3, 1, 2)

        decay = self.decay
        new_state = decay * state + gate * value
        return new_state

    def evolve(
        self,
        state: torch.Tensor,
        stimulus: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Evolve the field via PDE + DeltaNet correction + Titans memory.

        The PDE provides spatial coherence (diffusion + reaction).
        DeltaNet corrects temporal prediction errors via the delta rule.
        Titans provides persistent scene memory across rollouts.
        """
        B, C, H_b, W_b = state.shape
        n = H_b * W_b
        u = state

        # Precompute stimulus field
        if stimulus is not None:
            stim = self.stimulus_proj(stimulus)
            stim_field = stim.unsqueeze(-1).unsqueeze(-1).expand(B, C, H_b, W_b)
            stim_flat = stim_field.permute(0, 2, 3, 1).reshape(B, n, C)
        else:
            stim_flat = None

        dt = self.dt

        # Detach persistent states for TBPTT
        if self._h_global is not None:
            self._h_global = self._h_global.detach()

        for step in range(self.n_evolve_steps):
            # -- Spatial PDE (unchanged) --
            diff = self.diffusion(u)

            if self.use_hebbian:
                diff = self.hebbian.update_and_modulate(u, diff)

            u_flat = u.permute(0, 2, 3, 1).reshape(B, n, C)
            diff_flat = diff.permute(0, 2, 3, 1).reshape(B, n, C)

            react = self.reaction(u_flat)

            # Assembly: PDE terms
            du = diff_flat + react
            if stim_flat is not None:
                du = du + stim_flat

            # -- Titans persistent memory --
            if self.use_titans and self._h_global is not None:
                pooled = u_flat.mean(dim=1)  # (B, C)
                self._h_global = self.titans(self._h_global, pooled)
                alpha_pump = F.softplus(self.alpha_pump)
                h_broadcast = self._h_global.unsqueeze(1).expand(-1, n, -1)
                du = du + alpha_pump * h_broadcast

            # -- DeltaNet temporal correction --
            # The delta rule corrects prediction errors in the latent field.
            # PDE says "the state should diffuse this way", DeltaNet says
            # "but based on what I've seen before, it should actually go THIS way"
            if self.use_deltanet:
                delta_correction = self.deltanet(u_flat)
                du = du + delta_correction

            # Euler integration
            u_flat = u_flat + dt * du

            # Periodic normalization
            if (step + 1) % 2 == 0:
                u_flat = self.norm(u_flat)

            u = u_flat.reshape(B, H_b, W_b, C).permute(0, 3, 1, 2).contiguous()

        return u

    def read(self, state: torch.Tensor) -> torch.Tensor:
        pooled = state.mean(dim=(-2, -1))
        return self.read_proj(pooled)

    def read_spatial(self, state: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        H, W = target_hw
        if (H, W) == (state.shape[2], state.shape[3]):
            return state
        return F.interpolate(
            state, size=(H, W), mode="bilinear", align_corners=False
        )

    def forward(
        self,
        state: torch.Tensor,
        observation: Optional[torch.Tensor] = None,
        stimulus: Optional[torch.Tensor] = None,
        detach_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if observation is not None:
            state = self.write(state, observation)
        state = self.evolve(state, stimulus=stimulus)
        representation = self.read(state)
        if detach_state:
            state = state.detach()

        return state, representation
