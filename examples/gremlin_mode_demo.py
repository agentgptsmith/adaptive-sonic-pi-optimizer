"""
🌀 GREMLIN MODE DEMONSTRATION 🌀

This script showcases MAXIMUM CHAOS optimization strategies.
Will it work? Maybe. Will it be entertaining? Absolutely.

Features demonstrated:
- Quantum tunneling through local minima
- Rage quit detection (stuck? YEET!)
- Vibes-based hyperparameter tuning
- Adversarial breathing
- Cursed loss landscapes

Run this if you dare.
"""

import torch
import math
from pi_opt.optim import PiAdam
from pi_opt.chaos import (
    add_chaos_to_optimizer,
    CursedLossLandscape,
    ChaoticPiPhase
)


def rastrigin(x: torch.Tensor, A: float = 10.0) -> torch.Tensor:
    """The final boss of optimization benchmarks."""
    n = x.numel()
    return A * n + torch.sum(x**2 - A * torch.cos(2 * math.pi * x))


def main():
    print("🔥" * 30)
    print("     WELCOME TO GREMLIN MODE")
    print("  Where bugs are features and")
    print("    exploits are gameplay")
    print("🔥" * 30)
    print()

    # Test problem: 5D Rastrigin (notoriously evil)
    dim = 5
    x = torch.randn(dim, requires_grad=True) * 5.0
    print(f"Starting position: {x.data.tolist()}")
    print(f"Starting loss: {rastrigin(x).item():.4f}")
    print()

    # Create optimizer
    opt = PiAdam(
        [x],
        lr=0.02,
        pi_alpha=0.3,
        pi_lambdas=[0.5, 0.2, 0.05],
        pi_amplitude=0.15
    )

    # INJECT MAXIMUM CHAOS
    print("💉 Injecting FULL GREMLIN MODE...")
    chaos_controllers = add_chaos_to_optimizer(opt, chaos_mode="full_gremlin")

    # Optional: Make loss landscape HARDER (truly cursed)
    use_cursed_landscape = True
    if use_cursed_landscape:
        curse = CursedLossLandscape(roughness=0.5, frequency=5.0)
        print("😈 Cursed loss landscape: ACTIVATED")
    print()

    # Optimization loop with chaos
    print("Starting chaos optimization...")
    print("-" * 60)

    best_loss = float('inf')
    chaos_events = {
        'quantum_tunnels': 0,
        'rage_quits': 0,
        'vibes_adjustments': 0
    }

    for step in range(500):
        # Compute loss (with optional cursing)
        loss = rastrigin(x)
        if use_cursed_landscape:
            loss = curse.curse(loss, [x])

        # Track best
        if loss.item() < best_loss:
            best_loss = loss.item()

        # Check for rage quit
        params_before = x.data.clone()
        if chaos_controllers['rage'].check(loss.item(), [x]):
            chaos_events['rage_quits'] += 1
            print(f"  💢 Step {step:3d}: RAGE QUIT (yeeted parameters)")

        # Vibes-based tuning
        adjustments = chaos_controllers['vibes'].update(loss.item(), opt)
        if adjustments:
            chaos_events['vibes_adjustments'] += 1
            if step % 100 == 0:
                print(f"  🎵 Step {step:3d}: Vibes adjustment {adjustments}")

        # Standard optimization step
        opt.zero_grad()
        loss.backward()

        # Adversarial breathing
        if x.grad is not None:
            chaos_controllers['adversarial'].apply([x], [x.grad])

        opt.step()

        # Progress updates
        if step % 100 == 0:
            print(f"  Step {step:3d}: loss={loss.item():.4f}, best={best_loss:.4f}")

    # Final results
    print("-" * 60)
    print("🎉 CHAOS OPTIMIZATION COMPLETE")
    print()
    print(f"Final position: {x.data.tolist()}")
    print(f"Final loss: {rastrigin(x).item():.6f}")
    print(f"Best loss achieved: {best_loss:.6f}")
    print()
    print("Chaos events:")
    print(f"  🌀 Quantum tunnels: {chaos_events['quantum_tunnels']}")
    print(f"  💢 Rage quits: {chaos_events['rage_quits']}")
    print(f"  🎵 Vibes adjustments: {chaos_events['vibes_adjustments']}")
    print()

    # Judgment
    if best_loss < 0.1:
        print("✨ GREMLIN MODE: SUCCESS!")
        print("   The chaos has blessed you.")
    elif best_loss < 1.0:
        print("🎯 GREMLIN MODE: DECENT!")
        print("   Chaos worked... kinda.")
    elif best_loss < 10.0:
        print("🤷 GREMLIN MODE: IT TRIED!")
        print("   At least it's entertaining.")
    else:
        print("💀 GREMLIN MODE: PURE CHAOS!")
        print("   But hey, we learned something... maybe?")
    print()
    print("🔥" * 30)


if __name__ == "__main__":
    main()
