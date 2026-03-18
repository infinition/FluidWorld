"""
test_smoke.py -- Smoke tests for FluidWorld

Verifies that all modules instantiate correctly and that forward
passes produce the expected tensor shapes.

Usage :
    python -m pytest tests/test_smoke.py -v
    python tests/test_smoke.py
"""

import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Path setup ──
_here = Path(__file__).resolve().parent
_project = _here.parent
sys.path.insert(0, str(_project))

# ── Test constants ──
B = 2       # batch size
C = 64      # channels (d_model)
H = 8       # latent height
W = 8       # latent width
ACTION_DIM = 6
PROPRIO_DIM = 6


def test_action_force():
    """ActionForce: action -> spatial force field."""
    from fluidworld.core.action_force import ActionForce

    af = ActionForce(action_dim=ACTION_DIM, channels=C, force_spatial_size=4)
    action = torch.randn(B, ACTION_DIM)

    force = af(action, (H, W))

    assert force.shape == (B, C, H, W), f"Expected shape {(B, C, H, W)}, got {force.shape}"
    # Force should be small at initialization
    assert force.abs().max() < 1.0, "Initial force too large"
    print("  [OK] ActionForce")


def test_fluid_world_layer_no_action():
    """FluidWorldLayer2D without action = perception mode."""
    from fluidworld.core.fluid_world_layer import FluidWorldLayer2D

    layer = FluidWorldLayer2D(
        channels=C,
        action_dim=ACTION_DIM,
        dilations=[1, 4],
        max_steps=4,
        min_steps=1,
    )
    u = torch.randn(B, C, H, W)

    u_out, info = layer(u, action=None)

    assert u_out.shape == (B, C, H, W), f"Shape: {u_out.shape}"
    assert not info["action_injected"], "Action should not be injected"
    assert info["steps_used"] >= 1
    print("  [OK] FluidWorldLayer2D (no action)")


def test_fluid_world_layer_with_action():
    """FluidWorldLayer2D with action = imagination mode."""
    from fluidworld.core.fluid_world_layer import FluidWorldLayer2D

    layer = FluidWorldLayer2D(
        channels=C,
        action_dim=ACTION_DIM,
        dilations=[1, 4],
        max_steps=4,
        min_steps=1,
    )
    u = torch.randn(B, C, H, W)
    action = torch.randn(B, ACTION_DIM)

    u_out, info = layer(u, action=action)

    assert u_out.shape == (B, C, H, W), f"Shape: {u_out.shape}"
    assert info["action_injected"], "Action should be injected"
    print("  [OK] FluidWorldLayer2D (with action)")


def test_fluid_world_layer_backward():
    """Verify that gradients flow through the world layer."""
    from fluidworld.core.fluid_world_layer import FluidWorldLayer2D

    layer = FluidWorldLayer2D(
        channels=C,
        action_dim=ACTION_DIM,
        dilations=[1, 4],
        max_steps=3,
        min_steps=1,
    )
    u = torch.randn(B, C, H, W, requires_grad=True)
    action = torch.randn(B, ACTION_DIM, requires_grad=True)

    u_out, info = layer(u, action=action)
    loss = u_out.mean()
    loss.backward()

    assert u.grad is not None, "Missing gradient on u"
    assert action.grad is not None, "Missing gradient on action"
    print("  [OK] FluidWorldLayer2D backward")


def test_target_encoder():
    """EMATargetEncoder : clone + update EMA."""
    from fluidworld.core.target_encoder import EMATargetEncoder

    # Create a small network as online encoder
    online = torch.nn.Sequential(
        torch.nn.Linear(10, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 10),
    )

    target = EMATargetEncoder(online, momentum=0.99)

    # Weights should be identical at the start
    for p_t, p_o in zip(target.encoder.parameters(), online.parameters()):
        assert torch.allclose(p_t, p_o), "Initial weights differ"

    # Modify online weights
    with torch.no_grad():
        for p in online.parameters():
            p.add_(torch.randn_like(p) * 0.1)

    # Update EMA
    target.update(online)

    # Weights should have moved (but not completely)
    for p_t, p_o in zip(target.encoder.parameters(), online.parameters()):
        assert not torch.allclose(p_t, p_o), "Target weights should have moved"

    print("  [OK] EMATargetEncoder")


def test_cosine_momentum_schedule():
    """Cosine momentum schedule."""
    from fluidworld.core.target_encoder import cosine_momentum_schedule

    # At start: base_momentum
    m0 = cosine_momentum_schedule(0.996, 1.0, 0, 1000)
    assert abs(m0 - 0.996) < 1e-6, f"Momentum initial: {m0}"

    # At end: final_momentum
    m_end = cosine_momentum_schedule(0.996, 1.0, 1000, 1000)
    assert abs(m_end - 1.0) < 1e-6, f"Momentum final: {m_end}"

    # Monotonically increasing
    momenta = [cosine_momentum_schedule(0.996, 1.0, i, 100) for i in range(101)]
    for i in range(len(momenta) - 1):
        assert momenta[i] <= momenta[i + 1] + 1e-9, "Not monotone"

    print("  [OK] cosine_momentum_schedule")


def test_vicreg_losses():
    """VICReg losses: variance + covariance."""
    from fluidworld.core.vicreg import variance_loss, covariance_loss, vicreg_loss

    z = torch.randn(32, C)  # batch of features

    # Variance loss
    v_loss = variance_loss(z)
    assert v_loss.shape == (), "Must be a scalar"
    assert v_loss >= 0, "Must be >= 0"

    # Covariance loss
    c_loss = covariance_loss(z)
    assert c_loss.shape == (), "Must be a scalar"
    assert c_loss >= 0, "Must be >= 0"

    # Full VICReg (v5: takes only z_pred)
    z_pred = torch.randn(32, C)
    result = vicreg_loss(z_pred)
    assert "var_loss" in result
    assert "cov_loss" in result
    assert "vicreg_total" in result
    assert result["vicreg_total"] >= 0

    # Collapse test: if z is constant, variance_loss should be large
    z_collapsed = torch.ones(32, C) * 0.5
    v_collapsed = variance_loss(z_collapsed)
    assert v_collapsed > 0.5, f"Should detect collapse: {v_collapsed}"

    print("  [OK] VICReg losses")


def test_proprio_model():
    """ProprioWorldModel: single-step prediction."""
    from fluidworld.core.proprio_model import ProprioWorldModel

    model = ProprioWorldModel(
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=64,
    )

    proprio = torch.randn(B, PROPRIO_DIM)
    action = torch.randn(B, ACTION_DIM)

    pred = model(proprio, action)

    assert pred.shape == (B, PROPRIO_DIM), f"Shape: {pred.shape}"

    # At initialization (small init), prediction should be close to proprio
    delta = (pred - proprio).abs().max()
    assert delta < 0.5, f"Initial delta too large: {delta}"

    # Backward
    loss = F.mse_loss(pred, torch.randn(B, PROPRIO_DIM))
    loss.backward()

    print("  [OK] ProprioWorldModel")


def test_multi_step_proprio():
    """MultiStepProprioModel: multi-step prediction."""
    from fluidworld.core.proprio_model import MultiStepProprioModel

    T = 5
    model = MultiStepProprioModel(
        proprio_dim=PROPRIO_DIM,
        action_dim=ACTION_DIM,
        hidden_dim=64,
    )

    proprio_init = torch.randn(B, PROPRIO_DIM)
    actions = torch.randn(B, T, ACTION_DIM)

    preds = model(proprio_init, actions)
    assert preds.shape == (B, T, PROPRIO_DIM), f"Shape: {preds.shape}"

    # compute_loss
    targets = torch.randn(B, T, PROPRIO_DIM)
    result = model.compute_loss(proprio_init, actions, targets)

    assert "loss" in result
    assert "per_step_mse" in result
    assert result["per_step_mse"].shape == (T,)
    result["loss"].backward()

    print("  [OK] MultiStepProprioModel")


def test_compute_equilibrium_loss():
    """Equilibrium loss."""
    from fluidworld.core.fluid_world_layer import compute_equilibrium_loss

    info_list = [
        {"diff_turbulence": torch.tensor(0.1)},
        {"diff_turbulence": torch.tensor(0.05)},
    ]
    eq_loss = compute_equilibrium_loss(info_list)
    assert eq_loss.shape == (), "Must be a scalar"
    assert abs(eq_loss.item() - 0.075) < 1e-6, f"Unexpected value: {eq_loss}"

    # Empty case
    eq_empty = compute_equilibrium_loss([])
    assert eq_empty.item() == 0.0

    print("  [OK] compute_equilibrium_loss")


def run_all():
    """Run all tests."""
    print("=" * 50)
    print("FluidWorld -- Smoke Tests")
    print("=" * 50)

    tests = [
        test_action_force,
        test_fluid_world_layer_no_action,
        test_fluid_world_layer_with_action,
        test_fluid_world_layer_backward,
        test_target_encoder,
        test_cosine_momentum_schedule,
        test_vicreg_losses,
        test_proprio_model,
        test_multi_step_proprio,
        test_compute_equilibrium_loss,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
            failed += 1

    print("=" * 50)
    print(f"Results: {passed} OK, {failed} FAIL out of {len(tests)} tests")
    if failed > 0:
        sys.exit(1)
    print("All tests pass!")


if __name__ == "__main__":
    run_all()
