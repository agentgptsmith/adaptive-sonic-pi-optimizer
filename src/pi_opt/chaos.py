"""
CHAOS MODE: Experimental gremlin-tier features for π-recursive optimizers.

⚠️  WARNING: These are deliberately unhinged optimization strategies.
    Bugs are features. Exploits are gameplay. Use at your own risk.

Features:
- Quantum tunneling (random jumps to escape minima)
- Adversarial breathing (gradient-aware chaos injection)
- Rage quit detection (stuck? YEET some parameters)
- Vibes-based auto-tuning (loss goes up? breathe harder)
- Meta-optimization (optimizer optimizes itself)
"""

import torch
import math
import random
from typing import Optional, Callable


class ChaoticPiPhase:
    """
    π-recursive phase with QUANTUM TUNNELING.

    Sometimes just... teleports the phase. Why? Because fuck local minima.
    """

    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 1.0,
        lambdas: list = None,
        chaos_prob: float = 0.01,  # Probability of quantum tunneling
        chaos_scale: float = 10.0   # How far to tunnel
    ):
        self.alpha = alpha
        self.beta = beta
        self.lambdas = lambdas or [0.4, 0.15]
        self.chaos_prob = chaos_prob
        self.chaos_scale = chaos_scale
        self._last_phi = 0.0

    def phi(self, t: int) -> float:
        """Compute phase with occasional QUANTUM TUNNELING."""
        # Normal π-recursive phase
        u = t + self.beta
        drift = self.alpha * (math.log(u) / math.log(math.pi) if u > 0 else 0)
        harm = sum(lam * math.cos((math.pi ** (i+1)) * u)
                   for i, lam in enumerate(self.lambdas))
        phi = 2.0 * math.pi * (drift + harm)

        # QUANTUM TUNNELING: Random phase jumps
        if random.random() < self.chaos_prob:
            tunnel_offset = random.gauss(0, self.chaos_scale)
            phi += tunnel_offset
            # print(f"🌀 QUANTUM TUNNEL at step {t}: +{tunnel_offset:.2f} rad")

        self._last_phi = phi
        return phi


class RageQuitDetector:
    """
    Detects when optimizer is stuck and says "fuck this" and randomizes.

    If loss hasn't improved in N steps, randomly perturb parameters.
    This is either genius or insane. Probably both.
    """

    def __init__(
        self,
        patience: int = 100,
        rage_scale: float = 0.1,
        verbose: bool = True
    ):
        self.patience = patience
        self.rage_scale = rage_scale
        self.verbose = verbose
        self.best_loss = float('inf')
        self.steps_without_improvement = 0

    def check(self, loss: float, params: list) -> bool:
        """Check if we should rage quit. Returns True if YEETED."""
        if loss < self.best_loss:
            self.best_loss = loss
            self.steps_without_improvement = 0
            return False

        self.steps_without_improvement += 1

        if self.steps_without_improvement >= self.patience:
            # RAGE QUIT: Randomly perturb parameters
            if self.verbose:
                print(f"💢 RAGE QUIT! Stuck for {self.patience} steps. YEETING parameters...")

            with torch.no_grad():
                for p in params:
                    if p.requires_grad:
                        noise = torch.randn_like(p) * self.rage_scale * p.std()
                        p.add_(noise)

            self.steps_without_improvement = 0
            self.best_loss = float('inf')  # Reset
            return True

        return False


class VibesBasedTuner:
    """
    Tunes π-parameters based on ~vibes~ (loss trajectory).

    Loss going up? Breathe harder.
    Loss going down? Chill out.
    Loss oscillating? You're doing great sweetie.
    """

    def __init__(self, adapt_rate: float = 0.01):
        self.adapt_rate = adapt_rate
        self.loss_history = []

    def update(self, loss: float, optimizer) -> dict:
        """Update π-parameters based on vibes."""
        self.loss_history.append(loss)

        if len(self.loss_history) < 10:
            return {}

        # Analyze recent vibes
        recent = self.loss_history[-10:]
        trend = (recent[-1] - recent[0]) / (recent[0] + 1e-8)
        volatility = torch.tensor(recent).std().item() / (torch.tensor(recent).mean().item() + 1e-8)

        adjustments = {}

        # If loss increasing, BREATHE HARDER (increase amplitude)
        if trend > 0.01:
            if hasattr(optimizer, 'param_groups'):
                for group in optimizer.param_groups:
                    if 'pi_amplitude' in group:
                        old_amp = group['pi_amplitude']
                        group['pi_amplitude'] = min(0.5, old_amp * (1 + self.adapt_rate))
                        adjustments['pi_amplitude'] = group['pi_amplitude']

        # If loss decreasing, chill (decrease amplitude)
        elif trend < -0.01:
            if hasattr(optimizer, 'param_groups'):
                for group in optimizer.param_groups:
                    if 'pi_amplitude' in group:
                        old_amp = group['pi_amplitude']
                        group['pi_amplitude'] = max(0.01, old_amp * (1 - self.adapt_rate))
                        adjustments['pi_amplitude'] = group['pi_amplitude']

        # If super volatile, increase drift (more exploration)
        if volatility > 0.1:
            if hasattr(optimizer, '_phase'):
                optimizer._phase.alpha = min(0.5, optimizer._phase.alpha * 1.01)
                adjustments['pi_alpha'] = optimizer._phase.alpha

        return adjustments


class AdversarialBreathing:
    """
    Use GRADIENT DIRECTION to make breathing ADVERSARIAL.

    Instead of random breathing, breathe in directions that maximize exploration.
    Deliberately go the "wrong" way sometimes to escape basins.
    """

    def __init__(self, adversarial_prob: float = 0.05, adversarial_scale: float = 0.1):
        self.adversarial_prob = adversarial_prob
        self.adversarial_scale = adversarial_scale

    def apply(self, params: list, grads: list):
        """Occasionally add adversarial noise in gradient direction."""
        if random.random() < self.adversarial_prob:
            with torch.no_grad():
                for p, g in zip(params, grads):
                    if g is not None and p.requires_grad:
                        # Go OPPOSITE to gradient (deliberately wrong)
                        adversarial_step = g.sign() * random.uniform(0, self.adversarial_scale) * p.abs().mean()
                        p.add_(adversarial_step)


def add_chaos_to_optimizer(
    optimizer,
    chaos_mode: str = "quantum_tunnel",
    **chaos_kwargs
):
    """
    Inject CHAOS into an existing optimizer.

    Args:
        optimizer: PiAdam or PiSGD instance
        chaos_mode: One of "quantum_tunnel", "rage_quit", "vibes", "adversarial", "full_gremlin"
        **chaos_kwargs: Additional chaos parameters

    Returns:
        Chaos controller object
    """
    if chaos_mode == "quantum_tunnel":
        # Replace phase with chaotic phase
        if hasattr(optimizer, '_phase'):
            old_phase = optimizer._phase
            optimizer._phase = ChaoticPiPhase(
                alpha=old_phase.alpha,
                beta=old_phase.beta,
                lambdas=old_phase.lambdas,
                **chaos_kwargs
            )
        return optimizer._phase

    elif chaos_mode == "rage_quit":
        return RageQuitDetector(**chaos_kwargs)

    elif chaos_mode == "vibes":
        return VibesBasedTuner(**chaos_kwargs)

    elif chaos_mode == "adversarial":
        return AdversarialBreathing(**chaos_kwargs)

    elif chaos_mode == "full_gremlin":
        # ALL THE CHAOS
        controllers = {
            'quantum': ChaoticPiPhase(
                alpha=optimizer._phase.alpha if hasattr(optimizer, '_phase') else 0.25,
                beta=1.0,
                lambdas=[0.4, 0.15],
                chaos_prob=0.02,
                chaos_scale=20.0
            ),
            'rage': RageQuitDetector(patience=50, rage_scale=0.2),
            'vibes': VibesBasedTuner(adapt_rate=0.02),
            'adversarial': AdversarialBreathing(adversarial_prob=0.1, adversarial_scale=0.2)
        }
        if hasattr(optimizer, '_phase'):
            optimizer._phase = controllers['quantum']
        return controllers

    else:
        raise ValueError(f"Unknown chaos mode: {chaos_mode}")


class CursedLossLandscape:
    """
    Make your loss landscape WORSE on purpose to force better exploration.

    Adds random hills and valleys to smooth landscapes.
    Theory: Harder optimization = better final solution? Maybe?

    This is either 4D chess or complete madness.
    """

    def __init__(self, roughness: float = 0.1, frequency: float = 10.0):
        self.roughness = roughness
        self.frequency = frequency

    def curse(self, loss: torch.Tensor, params: list) -> torch.Tensor:
        """Add chaotic perturbation to loss."""
        # Create deterministic chaos based on parameters
        chaos = 0.0
        for i, p in enumerate(params):
            if p.requires_grad:
                param_hash = torch.sin(p.sum() * self.frequency + i).item()
                chaos += param_hash

        cursed_loss = loss + self.roughness * abs(chaos)
        return cursed_loss


# Example usage in a cursed training loop
if __name__ == "__main__":
    print("🌀 CHAOS MODE DEMONSTRATION 🌀")
    print("=" * 60)

    # Example: Optimize with FULL GREMLIN MODE
    import torch
    from pi_opt.optim import PiAdam

    # Simple test function
    x = torch.tensor([5.0, 5.0], requires_grad=True)

    opt = PiAdam([x], lr=0.01, pi_amplitude=0.1)

    # INJECT ALL THE CHAOS
    chaos_controllers = add_chaos_to_optimizer(opt, chaos_mode="full_gremlin")

    print("Optimizing with FULL GREMLIN MODE...")
    print("(Quantum tunneling + Rage quit + Vibes-based tuning + Adversarial breathing)")
    print()

    for step in range(200):
        loss = (x ** 2).sum()  # Simple quadratic

        # Check for rage quit
        if chaos_controllers['rage'].check(loss.item(), [x]):
            print(f"  Step {step}: RAGE QUIT TRIGGERED")

        # Vibes-based tuning
        adjustments = chaos_controllers['vibes'].update(loss.item(), opt)
        if adjustments and step % 50 == 0:
            print(f"  Step {step}: Vibes adjustment: {adjustments}")

        opt.zero_grad()
        loss.backward()

        # Adversarial breathing
        chaos_controllers['adversarial'].apply([x], [x.grad])

        opt.step()

        if step % 50 == 0:
            print(f"  Step {step}: loss={loss.item():.6f}, x=[{x[0].item():.3f}, {x[1].item():.3f}]")

    print("\nChaos optimization complete! 🎉")
