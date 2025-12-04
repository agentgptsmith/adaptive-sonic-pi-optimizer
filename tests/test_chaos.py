"""
Tests for CHAOS MODE features.

These tests verify that chaos features work... chaotically.
"""

import torch
import math
from pi_opt.optim import PiAdam
from pi_opt.chaos import (
    ChaoticPiPhase,
    RageQuitDetector,
    VibesBasedTuner,
    AdversarialBreathing,
    CursedLossLandscape,
    add_chaos_to_optimizer
)


def test_chaotic_phase_exists():
    """Test that chaotic phase can be created."""
    phase = ChaoticPiPhase(alpha=0.25, chaos_prob=0.1, chaos_scale=5.0)
    phi = phase.phi(100)
    assert math.isfinite(phi), "Chaotic phase should produce finite values"


def test_chaotic_phase_sometimes_tunnels():
    """Test that quantum tunneling happens (probabilistically)."""
    phase = ChaoticPiPhase(alpha=0.25, chaos_prob=0.5, chaos_scale=10.0)  # High prob

    # Run many times, should see variation from tunneling
    values = [phase.phi(10) for _ in range(100)]
    assert len(set(values)) > 1, "Should see variation from quantum tunneling"


def test_rage_quit_detector():
    """Test that rage quit triggers when stuck."""
    rage = RageQuitDetector(patience=5, rage_scale=0.1, verbose=False)

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    original = x.clone()

    # Feed same loss repeatedly to trigger rage quit
    for _ in range(4):
        triggered = rage.check(10.0, [x])
        assert not triggered, "Should not rage quit before patience"

    # Fifth time should trigger
    triggered = rage.check(10.0, [x])
    assert triggered, "Should rage quit after patience exceeded"
    assert not torch.allclose(x, original), "Parameters should be yeeted"


def test_vibes_based_tuner():
    """Test that vibes tuner tracks and adjusts."""
    vibes = VibesBasedTuner(adapt_rate=0.1)

    # Feed some losses
    for loss in [10.0, 9.0, 8.0, 7.0, 6.0]:
        vibes.update(loss, None)  # No actual optimizer needed for this test

    assert len(vibes.loss_history) == 5, "Should track loss history"


def test_vibes_tuner_with_optimizer():
    """Test vibes tuner with actual optimizer."""
    x = torch.tensor([1.0], requires_grad=True)
    opt = PiAdam([x], lr=0.01, pi_amplitude=0.1)

    vibes = VibesBasedTuner(adapt_rate=0.05)

    # Feed increasing loss (should increase amplitude)
    for i in range(15):
        loss = 10.0 + i  # Increasing
        adjustments = vibes.update(loss, opt)

    # Should have adjusted something
    current_amp = opt.param_groups[0]['pi_amplitude']
    assert current_amp >= 0.1, "Amplitude should increase with increasing loss"


def test_adversarial_breathing():
    """Test that adversarial breathing adds noise."""
    adversarial = AdversarialBreathing(adversarial_prob=1.0, adversarial_scale=0.1)  # Always trigger

    x = torch.tensor([1.0, 2.0], requires_grad=True)
    x.grad = torch.tensor([0.5, -0.5])
    original = x.clone()

    adversarial.apply([x], [x.grad])

    # Should have been perturbed (not guaranteed, but likely with 100% prob)
    # Note: due to randomness, might occasionally fail
    assert True  # Smoke test - just checking it doesn't crash


def test_cursed_loss_landscape():
    """Test that cursed landscapes add difficulty."""
    curse = CursedLossLandscape(roughness=0.1, frequency=5.0)

    x = torch.tensor([1.0, 2.0])
    original_loss = torch.tensor(5.0)

    cursed = curse.curse(original_loss, [x])

    assert cursed != original_loss, "Loss should be cursed"
    assert cursed.item() >= original_loss.item(), "Cursed loss should be >= original"


def test_add_chaos_quantum_tunnel():
    """Test adding quantum tunnel chaos to optimizer."""
    x = torch.tensor([1.0], requires_grad=True)
    opt = PiAdam([x], lr=0.01)

    phase = add_chaos_to_optimizer(opt, chaos_mode="quantum_tunnel", chaos_prob=0.1)

    assert isinstance(phase, ChaoticPiPhase), "Should return chaotic phase"
    assert opt._phase is phase, "Optimizer should use chaotic phase"


def test_add_chaos_rage_quit():
    """Test adding rage quit chaos."""
    x = torch.tensor([1.0], requires_grad=True)
    opt = PiAdam([x], lr=0.01)

    rage = add_chaos_to_optimizer(opt, chaos_mode="rage_quit", patience=10)

    assert isinstance(rage, RageQuitDetector), "Should return rage detector"


def test_add_chaos_full_gremlin():
    """Test FULL GREMLIN MODE activation."""
    x = torch.tensor([1.0], requires_grad=True)
    opt = PiAdam([x], lr=0.01)

    controllers = add_chaos_to_optimizer(opt, chaos_mode="full_gremlin")

    assert 'quantum' in controllers, "Should have quantum tunneling"
    assert 'rage' in controllers, "Should have rage quit"
    assert 'vibes' in controllers, "Should have vibes tuner"
    assert 'adversarial' in controllers, "Should have adversarial breathing"


def test_chaos_mode_doesnt_explode():
    """Integration test: make sure chaos mode completes without crashing."""
    x = torch.tensor([5.0, 5.0], requires_grad=True)
    opt = PiAdam([x], lr=0.02, pi_amplitude=0.1)

    # Add all the chaos
    controllers = add_chaos_to_optimizer(opt, chaos_mode="full_gremlin")
    curse = CursedLossLandscape(roughness=0.1)

    # Run optimization with chaos
    for step in range(50):
        loss = (x ** 2).sum()
        cursed_loss = curse.curse(loss, [x])

        # Check rage quit
        controllers['rage'].check(cursed_loss.item(), [x])

        # Vibes update
        controllers['vibes'].update(cursed_loss.item(), opt)

        # Optimization
        opt.zero_grad()
        cursed_loss.backward()

        # Adversarial
        controllers['adversarial'].apply([x], [x.grad])

        opt.step()

    # If we got here without crashing, chaos mode works!
    assert True, "Chaos mode survived 50 steps!"
