from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch
from torch import nn

OPTIMIZER_EXPERIMENTS_COMMIT = "689568d71ebe92093e5f5bf433127a5184ef0c35"
OPTIMIZER_EXPERIMENTS_SOURCE = (
    "https://github.com/catid/optimizer_experiments/"
    "workers/codex_noradam_confidence/experiments/run_wikitext_llm50m.py"
)


def _matrix_view(x: torch.Tensor) -> torch.Tensor:
    effective_shape = tuple(int(dim) for dim in x.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        raise ValueError("matrix update requires at least two non-singleton dimensions")
    if len(effective_shape) == 2:
        return x.reshape(effective_shape)
    return x.reshape(effective_shape[0], -1)


def _is_matrix_like(param: torch.Tensor, *, min_matrix_dim: int = 2) -> bool:
    if not param.is_floating_point():
        return False
    effective_shape = tuple(int(dim) for dim in param.shape if int(dim) > 1)
    if len(effective_shape) < 2:
        return False
    return min(effective_shape[0], math.prod(effective_shape[1:])) >= min_matrix_dim


class PaceOptimizer:
    """PACE pullback wrapper for an existing optimizer.

    Vendored and reduced from `optimizer_experiments` commit
    `689568d71ebe92093e5f5bf433127a5184ef0c35`.  This wrapper is intentionally
    optimizer-agnostic so `pace_muon` can preserve this project's official
    `SingleDeviceMuonWithAuxAdam` baseline and add only the PACE pullback/EMA
    evaluation policy around it.
    """

    def __init__(
        self,
        base_optimizer: torch.optim.Optimizer,
        *,
        pullback_c: float,
        kappa: float,
        precond: str = "adam",
        beta2: float = 0.999,
        eps: float = 1e-8,
        update_freq: int = 1,
        min_decay: float = 1e-4,
    ) -> None:
        if pullback_c < 0.0:
            raise ValueError("pullback_c must be non-negative")
        if not 0.0 < kappa <= 1.0:
            raise ValueError("kappa must be in (0, 1]")
        precond = str(precond).lower()
        if precond not in {"adam", "scalar", "row"}:
            raise ValueError("precond must be one of: adam, scalar, row")
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("beta2 must be in [0, 1)")
        if eps <= 0.0:
            raise ValueError("eps must be positive")
        if update_freq < 1:
            raise ValueError("update_freq must be >= 1")
        if min_decay < 0.0:
            raise ValueError("min_decay must be non-negative")

        self.base_optimizer = base_optimizer
        self.param_groups = base_optimizer.param_groups
        self.state: dict[nn.Parameter, dict[str, Any]] = {}
        self.pullback_c = float(pullback_c)
        self.kappa = float(kappa)
        self.precond = precond
        self.beta2 = float(beta2)
        self.eps = float(eps)
        self.update_freq = int(update_freq)
        self.min_decay = float(min_decay)
        self.step_index = 0
        self._swapped = False

    def _decay_t(self, step: int) -> float:
        return max((1.0 + float(step)) ** (-self.kappa), self.min_decay)

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self, closure: Any | None = None) -> Any:
        if self._swapped:
            raise RuntimeError("PaceOptimizer.step() called while weights are swapped to EMA")

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self.step_index += 1
        step = self.step_index
        decay = self._decay_t(step)

        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state.setdefault(parameter, {})
                ema = state.get("ema")
                if ema is None or ema.shape != parameter.shape:
                    ema = state["ema"] = parameter.detach().to(torch.float32, copy=True)
                pre = state.get("pre")
                if pre is None or pre.shape != parameter.shape:
                    pre = state["pre"] = torch.empty_like(parameter, dtype=torch.float32)
                pre.copy_(parameter.detach())

                if self.pullback_c <= 0.0:
                    continue
                if self.precond == "adam":
                    second = state.get("v")
                    if second is None or second.shape != parameter.shape:
                        second = state["v"] = torch.zeros_like(parameter, dtype=torch.float32)
                    grad = parameter.grad.detach().to(torch.float32)
                    second.mul_(self.beta2).addcmul_(grad, grad, value=1.0 - self.beta2)
                elif self.precond == "row" and _is_matrix_like(parameter):
                    grad_matrix = _matrix_view(parameter.grad.detach()).to(torch.float32)
                    rows = grad_matrix.shape[0]
                    second_row = state.get("v_row")
                    if second_row is None or second_row.shape != (rows,):
                        second_row = state["v_row"] = torch.zeros(
                            rows,
                            device=parameter.device,
                            dtype=torch.float32,
                        )
                    second_row.mul_(self.beta2).add_(
                        grad_matrix.square().mean(dim=1),
                        alpha=1.0 - self.beta2,
                    )

        base_loss = self.base_optimizer.step()
        if loss is None:
            loss = base_loss

        for group in self.param_groups:
            lr = float(group.get("lr", 0.0))
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state.get(parameter)
                if not state or "ema" not in state:
                    continue
                ema = state["ema"]
                pre = state["pre"]
                if self.pullback_c > 0.0:
                    if self.precond == "adam":
                        bias2 = max(1.0 - self.beta2**step, 1e-16)
                        gain = (state["v"] / bias2).sqrt().add_(self.eps).reciprocal_()
                        gain.mul_(lr * self.pullback_c * decay).clamp_(max=1.0)
                        diff = (ema - pre).mul_(gain)
                    elif self.precond == "row" and "v_row" in state:
                        bias2 = max(1.0 - self.beta2**step, 1e-16)
                        gain_row = (state["v_row"] / bias2).sqrt().add_(self.eps).reciprocal_()
                        gain_row.mul_(lr * self.pullback_c * decay).clamp_(max=1.0)
                        diff = _matrix_view(ema - pre).mul_(gain_row.unsqueeze(1)).reshape_as(parameter)
                    else:
                        diff = (ema - pre).mul_(min(lr * self.pullback_c * decay, 1.0))
                    parameter.add_(diff.to(parameter.dtype))
                if step % self.update_freq == 0:
                    ema.mul_(1.0 - decay).add_(
                        parameter.detach().to(torch.float32),
                        alpha=decay,
                    )
        return loss

    @torch.no_grad()
    def swap_to_ema(self) -> None:
        if self._swapped:
            return
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or "ema" not in state:
                    continue
                backup = state.get("live_backup")
                if backup is None or backup.shape != parameter.shape:
                    backup = state["live_backup"] = torch.empty_like(parameter)
                backup.copy_(parameter.detach())
                parameter.copy_(state["ema"].to(parameter.dtype))
        self._swapped = True

    @torch.no_grad()
    def swap_to_live(self) -> None:
        if not self._swapped:
            return
        for group in self.param_groups:
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if not state or "live_backup" not in state:
                    continue
                parameter.copy_(state["live_backup"])
        self._swapped = False

    @contextmanager
    def use_ema_weights(self) -> Iterator[None]:
        self.swap_to_ema()
        try:
            yield
        finally:
            self.swap_to_live()

    def _flat_params(self) -> list[nn.Parameter]:
        return [parameter for group in self.param_groups for parameter in group["params"]]

    @staticmethod
    def _clone_state_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.detach().clone()
        if isinstance(value, dict):
            return {key: PaceOptimizer._clone_state_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [PaceOptimizer._clone_state_value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(PaceOptimizer._clone_state_value(item) for item in value)
        return value

    def state_dict(self) -> dict[str, Any]:
        param_to_index = {parameter: index for index, parameter in enumerate(self._flat_params())}
        return {
            "base_optimizer": self.base_optimizer.state_dict(),
            "state": {
                param_to_index[parameter]: self._clone_state_value(state)
                for parameter, state in self.state.items()
                if parameter in param_to_index
            },
            "step_index": self.step_index,
            "pullback_c": self.pullback_c,
            "kappa": self.kappa,
            "precond": self.precond,
            "beta2": self.beta2,
            "eps": self.eps,
            "update_freq": self.update_freq,
            "min_decay": self.min_decay,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.base_optimizer.load_state_dict(state_dict["base_optimizer"])
        self.param_groups = self.base_optimizer.param_groups
        flat_params = self._flat_params()
        packed_state = state_dict.get("state", {})
        self.state = {}
        for index, state in packed_state.items():
            int_index = int(index)
            if int_index < len(flat_params):
                self.state[flat_params[int_index]] = self._clone_state_value(state)
        self.step_index = int(state_dict.get("step_index", 0))
        self.pullback_c = float(state_dict.get("pullback_c", self.pullback_c))
        self.kappa = float(state_dict.get("kappa", self.kappa))
        self.precond = str(state_dict.get("precond", self.precond))
        self.beta2 = float(state_dict.get("beta2", self.beta2))
        self.eps = float(state_dict.get("eps", self.eps))
        self.update_freq = int(state_dict.get("update_freq", self.update_freq))
        self.min_decay = float(state_dict.get("min_decay", self.min_decay))
        self._swapped = False


__all__ = [
    "OPTIMIZER_EXPERIMENTS_COMMIT",
    "OPTIMIZER_EXPERIMENTS_SOURCE",
    "PaceOptimizer",
]
