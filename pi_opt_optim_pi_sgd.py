
import math
from typing import Iterable, Optional, Tuple
import torch
from torch.optim import Optimizer
from ..schedules import PiPhase, pi_schedule

class PiSGD(Optimizer):
    def __init__(self, params: Iterable, lr: float = 1e-2, momentum: float = 0.9, weight_decay: float = 0.0, nesterov: bool = False,
                 pi_alpha: float = 0.25, pi_beta: float = 1.0, pi_lambdas: Optional[Iterable[float]] = None, pi_amplitude: float = 0.1,
                 anneal_b: float = 1e-4, momentum_amplitude: float = 0.05, m_bounds: Tuple[float, float] = (0.5, 0.99), maximize: bool = False):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov, pi_alpha=pi_alpha, pi_beta=pi_beta,
                        pi_lambdas=list(pi_lambdas) if pi_lambdas is not None else [0.4, 0.15], pi_amplitude=pi_amplitude,
                        anneal_b=anneal_b, momentum_amplitude=momentum_amplitude, m_bounds=m_bounds, maximize=maximize)
        super().__init__(params, defaults)
        self._phase = PiPhase(alpha=pi_alpha, beta=pi_beta, lambdas=defaults["pi_lambdas"])
        self._t = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._t += 1

        for group in self.param_groups:
            lr0 = group["lr"]
            mom0 = group["momentum"]
            wd = group["weight_decay"]
            nesterov = group["nesterov"]
            maximize = group.get("maximize", False)

            lr_t = pi_schedule(t=self._t, base=lr0, amplitude=group["pi_amplitude"], phase=self._phase, anneal_b=group["anneal_b"], mode="cos")
            phi = self._phase.phi(self._t)
            m_amp = group["momentum_amplitude"]
            mmin, mmax = group["m_bounds"]
            mom_t = max(mmin, min(mmax, mom0 + m_amp * math.sin(phi)))

            for p in group["params"]:
                if p.grad is None: continue
                d_p = -p.grad if maximize else p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["momentum_buffer"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                if wd != 0: d_p = d_p.add(p, alpha=wd)

                buf = state["momentum_buffer"]
                buf.mul_(mom_t).add_(d_p)

                update = d_p.add(buf, alpha=mom_t) if nesterov else buf
                p.add_(update, alpha=-lr_t)

        return loss
