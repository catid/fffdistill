"""Shape-preserving Linear baselines for matched-budget comparisons."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def dense_linear_parameter_count(in_features: int, out_features: int, *, bias: bool = True) -> int:
    """Return the parameter count for ``nn.Linear(in_features, out_features)``."""

    _require_positive_int("in_features", in_features)
    _require_positive_int("out_features", out_features)
    _require_bool("bias", bias)
    return in_features * out_features + (out_features if bias else 0)


def low_rank_linear_parameter_count(
    in_features: int,
    out_features: int,
    rank: int,
    *,
    bias: bool = True,
) -> int:
    """Return the parameter count for a rank-``rank`` factorized Linear."""

    _validate_low_rank_shape(in_features, out_features, rank)
    _require_bool("bias", bias)
    return rank * (in_features + out_features) + (out_features if bias else 0)


def smaller_dense_linear_parameter_count(
    in_features: int,
    out_features: int,
    active_out_features: int,
    *,
    bias: bool = True,
) -> int:
    """Return the trainable parameter count for an output-row-subset dense Linear."""

    _validate_smaller_dense_shape(in_features, out_features, active_out_features)
    _require_bool("bias", bias)
    return active_out_features * (in_features + (1 if bias else 0))


def low_rank_rank_for_parameter_budget(
    in_features: int,
    out_features: int,
    parameter_budget: int,
    *,
    bias: bool = True,
) -> int:
    """Choose the largest low-rank factorization rank that fits ``parameter_budget``."""

    _require_positive_int("in_features", in_features)
    _require_positive_int("out_features", out_features)
    _require_positive_int("parameter_budget", parameter_budget)
    _require_bool("bias", bias)

    bias_parameters = out_features if bias else 0
    rank = (parameter_budget - bias_parameters) // (in_features + out_features)
    rank = min(rank, in_features, out_features)
    if rank < 1:
        minimum = low_rank_linear_parameter_count(in_features, out_features, 1, bias=bias)
        raise ValueError(
            f"parameter_budget={parameter_budget} cannot fit rank 1 low-rank Linear; "
            f"minimum is {minimum}"
        )
    return rank


def smaller_dense_out_features_for_parameter_budget(
    in_features: int,
    out_features: int,
    parameter_budget: int,
    *,
    bias: bool = True,
) -> int:
    """Choose the largest active output-row count that fits ``parameter_budget``."""

    _require_positive_int("in_features", in_features)
    _require_positive_int("out_features", out_features)
    _require_positive_int("parameter_budget", parameter_budget)
    _require_bool("bias", bias)

    parameters_per_output = in_features + (1 if bias else 0)
    active_out_features = min(out_features, parameter_budget // parameters_per_output)
    if active_out_features < 1:
        minimum = smaller_dense_linear_parameter_count(
            in_features,
            out_features,
            1,
            bias=bias,
        )
        raise ValueError(
            f"parameter_budget={parameter_budget} cannot fit 1 dense output row; "
            f"minimum is {minimum}"
        )
    return active_out_features


def linear_module_parameter_count(module: nn.Module) -> int:
    """Return the number of trainable and frozen parameters registered on ``module``."""

    return sum(parameter.numel() for parameter in module.parameters())


def _validate_low_rank_shape(in_features: int, out_features: int, rank: int) -> None:
    _require_positive_int("in_features", in_features)
    _require_positive_int("out_features", out_features)
    _require_positive_int("rank", rank)
    max_rank = min(in_features, out_features)
    if rank > max_rank:
        raise ValueError(f"rank must be <= min(in_features, out_features) ({max_rank})")


def _validate_smaller_dense_shape(
    in_features: int,
    out_features: int,
    active_out_features: int,
) -> None:
    _require_positive_int("in_features", in_features)
    _require_positive_int("out_features", out_features)
    _require_positive_int("active_out_features", active_out_features)
    if active_out_features > out_features:
        raise ValueError("active_out_features must be <= out_features")


class LowRankLinear(nn.Module):
    """A factorized Linear with public shape ``[..., in_features] -> [..., out_features]``."""

    in_features: int
    out_features: int
    rank: int
    input_weight: nn.Parameter
    output_weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _validate_low_rank_shape(in_features, out_features, rank)
        _require_bool("bias", bias)
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        factory_kwargs = {"device": device, "dtype": dtype}
        self.input_weight = nn.Parameter(torch.empty((rank, in_features), **factory_kwargs))
        self.output_weight = nn.Parameter(torch.empty((out_features, rank), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.input_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.output_weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        hidden = F.linear(input, self.input_weight)
        return F.linear(hidden, self.output_weight, self.bias)

    def parameter_count(self) -> int:
        return low_rank_linear_parameter_count(
            self.in_features,
            self.out_features,
            self.rank,
            bias=self.bias is not None,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"rank={self.rank}, bias={self.bias is not None}"
        )


class SmallerDenseLinear(nn.Module):
    """A dense Linear over a prefix of output rows, zero-padded to the full output shape."""

    in_features: int
    out_features: int
    active_out_features: int
    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        active_out_features: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _validate_smaller_dense_shape(in_features, out_features, active_out_features)
        _require_bool("bias", bias)
        self.in_features = in_features
        self.out_features = out_features
        self.active_out_features = active_out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        self.weight = nn.Parameter(torch.empty((active_out_features, in_features), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(active_out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        active_output = F.linear(input, self.weight, self.bias)
        inactive_out_features = self.out_features - self.active_out_features
        if inactive_out_features == 0:
            return active_output
        return F.pad(active_output, (0, inactive_out_features))

    def parameter_count(self) -> int:
        return smaller_dense_linear_parameter_count(
            self.in_features,
            self.out_features,
            self.active_out_features,
            bias=self.bias is not None,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"active_out_features={self.active_out_features}, bias={self.bias is not None}"
        )


def _require_linear(module: nn.Linear) -> None:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(module).__name__}")


def make_matched_low_rank_linear(
    linear: nn.Linear,
    *,
    parameter_budget: int,
) -> LowRankLinear:
    """Create a low-rank replacement using the largest rank within ``parameter_budget``."""

    _require_linear(linear)
    rank = low_rank_rank_for_parameter_budget(
        linear.in_features,
        linear.out_features,
        parameter_budget,
        bias=linear.bias is not None,
    )
    replacement = LowRankLinear(
        linear.in_features,
        linear.out_features,
        rank,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    replacement.train(linear.training)
    return replacement


def make_matched_smaller_dense_linear(
    linear: nn.Linear,
    *,
    parameter_budget: int,
) -> SmallerDenseLinear:
    """Create a smaller dense replacement with as many output rows as fit the budget."""

    _require_linear(linear)
    active_out_features = smaller_dense_out_features_for_parameter_budget(
        linear.in_features,
        linear.out_features,
        parameter_budget,
        bias=linear.bias is not None,
    )
    replacement = SmallerDenseLinear(
        linear.in_features,
        linear.out_features,
        active_out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    replacement.train(linear.training)
    return replacement
