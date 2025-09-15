
import math
from typing import Iterable, Optional

class PiPhase:
    def __init__(self, alpha: float = 0.25, beta: float = 1.0, lambdas: Optional[Iterable[float]] = None):
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambdas = list(lambdas) if lambdas is not None else [0.4, 0.15]

    @staticmethod
    def _log_base_pi(x: float) -> float:
        if x <= 0.0: x = 1e-12
        return math.log(x) / math.log(math.pi)

    def phi(self, t: int) -> float:
        u = t + self.beta
        drift = self.alpha * self._log_base_pi(u)
        harm = 0.0
        for i, lam in enumerate(self.lambdas, start=1):
            harm += lam * math.cos((math.pi ** i) * u)
        return 2.0 * math.pi * (drift + harm)

def pi_schedule(t: int, base: float, amplitude: float = 0.1, phase: Optional[PiPhase] = None, anneal_b: float = 1e-4, mode: str = "cos") -> float:
    phase = phase or PiPhase()
    phi = phase.phi(t)
    carrier = math.cos(phi) if mode == "cos" else math.sin(phi)
    val = base * (1.0 + amplitude * carrier)
    if anneal_b > 0:
        val /= math.sqrt(1.0 + anneal_b * max(t, 0))
    return val
