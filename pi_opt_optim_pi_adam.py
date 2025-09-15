
import math
from typing import Iterable, Optional, Tuple
import torch
from torch.optim import Optimizer
from ..schedules import PiPhase, pi_schedule

class PiAdam(Optimizer):
    def __init__(self, params: Iterable, lr: float = 3e-4, betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0,
                 pi_alpha: float = 0.25, pi_beta: float = 1.0, pi_lambdas: Optional[Iterable[float]] = None, pi_amplitude: float = 0.1,
                 anneal_b: float = 1e-4, momentum_amplitude: float = 0.05, m_bounds: Tuple[float, float] = (0.7, 0.99),
                 maximize: bool = False, foreach: Optional[bool] = None, capturable: bool = False, differentiable: bool = False, fused: Optional[bool] = None):
        if lr <= 0.0: raise ValueError(f"Invalid lr: {lr}")
        if eps <= 0.0: raise ValueError(f"Invalid eps: {eps}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, pi_alpha=pi_alpha, pi_beta=pi_beta,
                        pi_lambdas=list(pi_lambdas) if pi_lambdas is not None else [0.4, 0.15],
                        pi_amplitude=pi_amplitude, anneal_b=anneal_b, momentum_amplitude=momentum_amplitude,
                        m_bounds=m_bounds, maximize=maximize, foreach=foreach, capturable=capturable, differentiable=differentiable, fused=fused)
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
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            maximize = group.get("maximize", False)

            lr_t = pi_schedule(t=self._t, base=lr0, amplitude=group["pi_amplitude"], phase=self._phase, anneal_b=group["anneal_b"], mode="cos")
            phi = self._phase.phi(self._t)
            m_amp = group["momentum_amplitude"]
            mmin, mmax = group["m_bounds"]
            beta1_t = max(mmin, min(mmax, beta1 + m_amp * math.sin(phi)))

            for p in group["params"]:
                if p.grad is None: continue
                grad = -p.grad if maximize else p.grad

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1

                if wd != 0: p.mul_(1 - lr_t * wd)

                exp_avg.mul_(beta1_t).add_(grad, alpha=(1 - beta1_t))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))

                denom = exp_avg_sq.sqrt().add_(eps)
                p.addcdiv_(exp_avg, denom, value=-lr_t)

        return loss
