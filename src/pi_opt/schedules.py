"""
π-recursive phase modulation and learning rate schedules.

This module implements phase computation and schedule modulation based on
π-recursive patterns, enabling adaptive "breathing" in learning rates to help
optimization escape shallow local minima.
"""

import math
from typing import Iterable, List, Optional


class PiPhase:
    """
    Computes π-recursive phase values for modulating optimizer schedules.

    The phase combines:
    - A logarithmic drift term (base-π logarithm)
    - Harmonic oscillations with π^i frequency scaling

    This creates a complex, non-repeating pattern that helps optimizers
    explore the loss landscape more effectively.

    Args:
        alpha: Scaling factor for the logarithmic drift component
        beta: Time offset to avoid log(0)
        lambdas: Amplitudes for harmonic components (one per frequency level)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        beta: float = 1.0,
        lambdas: Optional[Iterable[float]] = None
    ) -> None:
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambdas: List[float] = list(lambdas) if lambdas is not None else [0.4, 0.15]

    @staticmethod
    def _log_base_pi(x: float) -> float:
        """Compute logarithm base π, with safeguard for x ≤ 0."""
        if x <= 0.0:
            x = 1e-12
        return math.log(x) / math.log(math.pi)

    def phi(self, t: int) -> float:
        """
        Compute the phase value at timestep t.

        Args:
            t: Current training step (non-negative integer)

        Returns:
            Phase value in radians
        """
        u = t + self.beta
        drift = self.alpha * self._log_base_pi(u)
        harm = 0.0
        for i, lam in enumerate(self.lambdas, start=1):
            harm += lam * math.cos((math.pi ** i) * u)
        return 2.0 * math.pi * (drift + harm)

def pi_schedule(
    t: int,
    base: float,
    amplitude: float = 0.1,
    phase: Optional[PiPhase] = None,
    anneal_b: float = 1e-4,
    mode: str = "cos"
) -> float:
    """
    Compute a π-modulated learning rate at timestep t.

    The schedule oscillates around the base value with π-recursive phase
    modulation, creating a "breathing" pattern that helps avoid local minima.
    Optional annealing gradually reduces the effective learning rate over time.

    Args:
        t: Current training step (non-negative integer)
        base: Base learning rate value
        amplitude: Modulation amplitude (fraction of base rate)
        phase: PiPhase instance (creates default if None)
        anneal_b: Annealing coefficient (0 = no annealing)
        mode: Modulation mode ("cos" or "sin")

    Returns:
        Modulated learning rate value

    Raises:
        ValueError: If base <= 0, amplitude < 0, or anneal_b < 0
    """
    if base <= 0:
        raise ValueError(f"base must be positive, got {base}")
    if amplitude < 0:
        raise ValueError(f"amplitude must be non-negative, got {amplitude}")
    if anneal_b < 0:
        raise ValueError(f"anneal_b must be non-negative, got {anneal_b}")
    if mode not in ("cos", "sin"):
        raise ValueError(f"mode must be 'cos' or 'sin', got {mode}")

    phase = phase or PiPhase()
    phi = phase.phi(t)
    carrier = math.cos(phi) if mode == "cos" else math.sin(phi)
    val = base * (1.0 + amplitude * carrier)

    if anneal_b > 0:
        val /= math.sqrt(1.0 + anneal_b * max(t, 0))

    return val
