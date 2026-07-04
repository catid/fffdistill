"""Initial torch-only Fast Feedforward linear layer.

The layer is a binary-tree mixture of row-wise linear features. Shared rows
always contribute to the output, route rows choose one path through the tree,
and leaf rows contribute according to the selected or soft-routed leaf.
Route rows may optionally also contribute to the output as an ablation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ActivationName = Literal["silu", "gelu", "relu"]
ForwardMode = Literal["grouped", "naive"]
RouteRowRole = Literal["routing_only", "shared_routing_and_output", "split_routing_output"]
RouteRowsOutputCount = int | Literal["all"]


@dataclass(frozen=True)
class FFFLinearConfig:
    """Configuration for :class:`FFFLinear`."""

    in_features: int
    out_features: int
    depth: int = 2
    shared_rows: int = 0
    route_rows: int = 1
    route_result_rows: int = 0
    leaf_rows: int = 1
    activation: ActivationName = "silu"
    hard_routing: bool = True
    train_temperature: float = 1.0
    region_leak: float = 0.0
    route_rows_contribute: bool = False
    route_row_role: RouteRowRole = "routing_only"
    route_rows_output_count: RouteRowsOutputCount | None = None
    route_rows_output_fraction: float | None = None
    master_leaf: bool = False
    fallback_leaf: bool = False
    bias: bool = True
    forward_mode: ForwardMode = "grouped"

    def normalized(self) -> FFFLinearConfig:
        """Align legacy ``route_rows_contribute`` with explicit role semantics."""

        route_row_role = self.route_row_role
        route_rows_contribute = self.route_rows_contribute
        if route_rows_contribute and route_row_role == "routing_only":
            route_row_role = "shared_routing_and_output"
        elif route_row_role != "routing_only" and not route_rows_contribute:
            route_rows_contribute = True
        return replace(
            self,
            route_row_role=route_row_role,
            route_rows_contribute=route_rows_contribute,
        )

    def validate(self) -> None:
        config = self.normalized()
        _require_positive_int("in_features", self.in_features)
        _require_positive_int("out_features", self.out_features)
        _require_positive_int("depth", self.depth)
        _require_non_negative_int("shared_rows", self.shared_rows)
        if not isinstance(self.route_rows_contribute, bool):
            raise ValueError("route_rows_contribute must be a bool")
        if self.route_rows not in (1, 2):
            raise ValueError("route_rows must be 1 or 2")
        _require_non_negative_int("route_result_rows", config.route_result_rows)
        if config.route_row_role not in (
            "routing_only",
            "shared_routing_and_output",
            "split_routing_output",
        ):
            raise ValueError(
                "route_row_role must be one of: routing_only, "
                "shared_routing_and_output, split_routing_output"
            )
        if config.route_row_role == "split_routing_output":
            if config.route_result_rows <= 0:
                raise ValueError("route_result_rows must be positive for split_routing_output")
        elif config.route_result_rows != 0:
            raise ValueError(
                "route_result_rows must be 0 unless route_row_role is split_routing_output"
            )
        if self.leaf_rows not in (1, 2, 4):
            raise ValueError("leaf_rows must be one of 1, 2, or 4")
        if self.activation not in ("silu", "gelu", "relu"):
            raise ValueError("activation must be one of: silu, gelu, relu")
        if self.train_temperature <= 0.0 or not math.isfinite(self.train_temperature):
            raise ValueError("train_temperature must be a positive finite value")
        if not 0.0 <= self.region_leak <= 1.0:
            raise ValueError("region_leak must be in [0, 1]")
        if self.forward_mode not in ("grouped", "naive"):
            raise ValueError("forward_mode must be 'grouped' or 'naive'")
        _validate_route_output_controls(
            route_row_role=config.route_row_role,
            route_rows_output_count=config.route_rows_output_count,
            route_rows_output_fraction=config.route_rows_output_fraction,
        )


@dataclass(frozen=True)
class FFFRouteInfo:
    """Routing diagnostics for a batch.

    ``leaf_probs`` are the router probabilities before ``region_leak``.
    ``leaf_weights`` are the effective regular-leaf output weights.
    Route output contribution, when enabled, follows the hard/argmax path for
    both hard and soft routing.
    """

    leaf_ids: Tensor
    leaf_probs: Tensor
    leaf_weights: Tensor
    node_ids: Tensor
    route_bits: Tensor
    route_logits: Tensor
    route_values: Tensor
    route_row_ids: Tensor
    route_result_values: Tensor
    route_result_row_ids: Tensor
    diagnostics: dict[str, object]


@dataclass(frozen=True)
class _FlatRouteInfo:
    leaf_ids: Tensor
    leaf_probs: Tensor
    leaf_weights: Tensor
    node_ids: Tensor
    route_bits: Tensor
    route_logits: Tensor
    route_values: Tensor
    route_row_ids: Tensor
    route_result_values: Tensor
    route_result_row_ids: Tensor
    active_rows_per_token: Tensor


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _validate_route_output_controls(
    *,
    route_row_role: RouteRowRole,
    route_rows_output_count: RouteRowsOutputCount | None,
    route_rows_output_fraction: float | None,
) -> None:
    if route_rows_output_count is not None and route_rows_output_count != "all":
        _require_non_negative_int("route_rows_output_count", route_rows_output_count)

    if route_rows_output_fraction is not None:
        if isinstance(route_rows_output_fraction, bool) or not isinstance(
            route_rows_output_fraction,
            int | float,
        ):
            raise ValueError("route_rows_output_fraction must be numeric")
        if not 0.0 <= float(route_rows_output_fraction) <= 1.0:
            raise ValueError("route_rows_output_fraction must be in [0, 1]")

    count_requests_output = route_rows_output_count not in (None, 0)
    fraction_requests_output = (
        route_rows_output_fraction is not None and float(route_rows_output_fraction) > 0.0
    )
    if route_row_role == "routing_only" and (count_requests_output or fraction_requests_output):
        raise ValueError("routing_only conflicts with non-zero route row output controls")


def _active_autocast_dtype(device_type: str) -> torch.dtype | None:
    try:
        enabled = torch.is_autocast_enabled(device_type)
    except TypeError:  # pragma: no cover - compatibility with older torch APIs
        enabled = device_type == "cuda" and torch.is_autocast_enabled()
    if not enabled:
        return None

    try:
        return torch.get_autocast_dtype(device_type)
    except (AttributeError, TypeError):  # pragma: no cover - compatibility fallback
        if device_type == "cuda":
            return torch.get_autocast_gpu_dtype()
        if device_type == "cpu":
            return torch.get_autocast_cpu_dtype()
        return None


class FFFLinear(nn.Module):
    """Binary-tree Fast Feedforward linear approximation.

    Args mirror :class:`FFFLinearConfig`. The public forward is shape preserving:
    ``[..., in_features] -> [..., out_features]``.
    """

    config: FFFLinearConfig
    shared_weight: nn.Parameter | None
    shared_bias: nn.Parameter | None
    shared_output: nn.Parameter | None
    route_output: nn.Parameter | None
    route_result_weight: nn.Parameter | None
    route_result_bias: nn.Parameter | None
    route_result_output: nn.Parameter | None
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int | FFFLinearConfig,
        out_features: int | None = None,
        *,
        depth: int = 2,
        shared_rows: int = 0,
        route_rows: int = 1,
        route_result_rows: int = 0,
        leaf_rows: int = 1,
        activation: ActivationName = "silu",
        hard_routing: bool = True,
        train_temperature: float = 1.0,
        region_leak: float = 0.0,
        route_rows_contribute: bool = False,
        route_row_role: RouteRowRole = "routing_only",
        route_rows_output_count: RouteRowsOutputCount | None = None,
        route_rows_output_fraction: float | None = None,
        master_leaf: bool = False,
        fallback_leaf: bool = False,
        bias: bool = True,
        forward_mode: ForwardMode = "grouped",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if isinstance(in_features, FFFLinearConfig):
            if out_features is not None:
                raise ValueError("out_features must be omitted when passing FFFLinearConfig")
            config = in_features
        else:
            if out_features is None:
                raise ValueError("out_features is required")
            config = FFFLinearConfig(
                in_features=in_features,
                out_features=out_features,
                depth=depth,
                shared_rows=shared_rows,
                route_rows=route_rows,
                route_result_rows=route_result_rows,
                leaf_rows=leaf_rows,
                activation=activation,
                hard_routing=hard_routing,
                train_temperature=train_temperature,
                region_leak=region_leak,
                route_rows_contribute=route_rows_contribute,
                route_row_role=route_row_role,
                route_rows_output_count=route_rows_output_count,
                route_rows_output_fraction=route_rows_output_fraction,
                master_leaf=master_leaf,
                fallback_leaf=fallback_leaf,
                bias=bias,
                forward_mode=forward_mode,
            )
        config = config.normalized()
        config.validate()
        self.config = config

        factory_kwargs = {"device": device, "dtype": dtype}
        if self.shared_rows > 0:
            self.shared_weight = nn.Parameter(
                torch.empty(self.shared_rows, self.in_features, **factory_kwargs)
            )
            self.shared_bias = nn.Parameter(torch.empty(self.shared_rows, **factory_kwargs))
            self.shared_output = nn.Parameter(
                torch.empty(self.shared_rows, self.out_features, **factory_kwargs)
            )
        else:
            self.shared_weight = None
            self.shared_bias = None
            self.shared_output = None

        self.route_weight = nn.Parameter(
            torch.empty(
                self.internal_nodes,
                self.route_rows,
                self.in_features,
                **factory_kwargs,
            )
        )
        self.route_bias = nn.Parameter(
            torch.empty(self.internal_nodes, self.route_rows, **factory_kwargs)
        )
        if self.config.route_row_role == "shared_routing_and_output":
            self.route_output = nn.Parameter(
                torch.empty(
                    self.internal_nodes,
                    self.route_rows,
                    self.out_features,
                    **factory_kwargs,
                )
            )
        else:
            self.route_output = None

        if self.config.route_row_role == "split_routing_output":
            self.route_result_weight = nn.Parameter(
                torch.empty(
                    self.internal_nodes,
                    self.route_result_rows,
                    self.in_features,
                    **factory_kwargs,
                )
            )
            self.route_result_bias = nn.Parameter(
                torch.empty(self.internal_nodes, self.route_result_rows, **factory_kwargs)
            )
            self.route_result_output = nn.Parameter(
                torch.empty(
                    self.internal_nodes,
                    self.route_result_rows,
                    self.out_features,
                    **factory_kwargs,
                )
            )
        else:
            self.route_result_weight = None
            self.route_result_bias = None
            self.route_result_output = None

        self.leaf_weight = nn.Parameter(
            torch.empty(
                self.leaf_banks,
                self.leaf_rows,
                self.in_features,
                **factory_kwargs,
            )
        )
        self.leaf_bias = nn.Parameter(torch.empty(self.leaf_banks, self.leaf_rows, **factory_kwargs))
        self.leaf_output = nn.Parameter(
            torch.empty(
                self.leaf_banks,
                self.leaf_rows,
                self.out_features,
                **factory_kwargs,
            )
        )

        if self.config.bias:
            self.bias = nn.Parameter(torch.empty(self.out_features, **factory_kwargs))
        else:
            self.bias = None

        self.reset_parameters()

    @property
    def in_features(self) -> int:
        return self.config.in_features

    @property
    def out_features(self) -> int:
        return self.config.out_features

    @property
    def depth(self) -> int:
        return self.config.depth

    @property
    def shared_rows(self) -> int:
        return self.config.shared_rows

    @property
    def route_rows(self) -> int:
        return self.config.route_rows

    @property
    def route_result_rows(self) -> int:
        return self.config.route_result_rows

    @property
    def route_row_role(self) -> RouteRowRole:
        return self.config.route_row_role

    @property
    def leaf_rows(self) -> int:
        return self.config.leaf_rows

    @property
    def leaves(self) -> int:
        return 1 << self.depth

    @property
    def num_leaves(self) -> int:
        return self.leaves

    @property
    def internal_nodes(self) -> int:
        return self.leaves - 1

    @property
    def leaf_banks(self) -> int:
        return self.leaves + int(self.config.master_leaf) + int(self.config.fallback_leaf)

    @property
    def master_leaf_index(self) -> int | None:
        return self.leaves if self.config.master_leaf else None

    @property
    def fallback_leaf_index(self) -> int | None:
        if not self.config.fallback_leaf:
            return None
        return self.leaves + int(self.config.master_leaf)

    @property
    def stored_rows(self) -> int:
        return (
            self.shared_rows
            + self.internal_nodes * self.route_rows
            + self.internal_nodes * self.route_result_rows
            + self.leaf_banks * self.leaf_rows
        )

    @property
    def stored_route_output_rows(self) -> int:
        return self.internal_nodes * self._route_output_rows_per_node()

    @property
    def effective_route_output_rows(self) -> int:
        return self.internal_nodes * self._route_output_rows_per_node_selected()

    @property
    def unused_route_output_rows(self) -> int:
        return self.stored_route_output_rows - self.effective_route_output_rows

    @property
    def unused_stored_route_output_rows(self) -> int:
        if self.route_row_role == "split_routing_output":
            return self.unused_route_output_rows
        return 0

    @property
    def effective_stored_rows(self) -> int:
        return self.stored_rows - self.unused_stored_route_output_rows

    @property
    def effective_trainable_rows(self) -> int:
        return self.effective_stored_rows

    @property
    def max_visited_route_rows_per_token(self) -> int:
        return self.depth * self.route_rows

    @property
    def route_output_rows_per_token(self) -> int:
        return self._route_output_count()

    @property
    def max_route_output_rows_per_token(self) -> int:
        return self.depth * self._route_output_rows_per_node()

    def reset_parameters(self) -> None:
        if self.shared_weight is not None:
            self._reset_input_bank(self.shared_weight, self.shared_bias)
            self._reset_output_bank(self.shared_output)
        self._reset_input_bank(self.route_weight, self.route_bias)
        if self.route_output is not None:
            self._reset_output_bank(self.route_output)
        if self.route_result_weight is not None:
            self._reset_input_bank(self.route_result_weight, self.route_result_bias)
            self._reset_output_bank(self.route_result_output)
        self._reset_input_bank(self.leaf_weight, self.leaf_bias)
        self._reset_output_bank(self.leaf_output)
        if self.bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self,
        x: Tensor,
        *,
        implementation: ForwardMode | None = None,
    ) -> Tensor:
        mode = implementation or self.config.forward_mode
        if mode == "grouped":
            return self.forward_grouped(x)
        if mode == "naive":
            return self.forward_naive(x)
        raise ValueError("implementation must be 'grouped' or 'naive'")

    def forward_grouped(self, x: Tensor) -> Tensor:
        flat, leading_shape = self._flatten_input(x)
        route_info = self._route_flat(flat, hard=self.config.hard_routing)
        out = flat.new_zeros(flat.shape[0], self.out_features)

        if self.shared_weight is not None:
            shared_values = self._activation(
                F.linear(flat, self.shared_weight, self.shared_bias)
            )
            out = out + shared_values @ self.shared_output

        out = out + self._regular_leaf_output_grouped(flat, route_info)
        out = out + self._extra_leaf_output_grouped(flat, fallback_weight=self._fallback_weight())
        out = out + self._route_output_grouped(route_info)

        if self.bias is not None:
            out = out + self.bias
        return self._finalize_forward_output(out, leading_shape, x)

    def forward_naive(self, x: Tensor) -> Tensor:
        flat, leading_shape = self._flatten_input(x)
        route_info = self._route_flat(flat, hard=self.config.hard_routing)
        outputs: list[Tensor] = []
        route_output_count = self._route_output_count()
        fallback_weight = self._fallback_weight()

        for token_idx in range(flat.shape[0]):
            token = flat[token_idx]
            out = token.new_zeros(self.out_features)

            if self.shared_weight is not None:
                values = self._activation(self.shared_weight @ token + self.shared_bias)
                out = out + values @ self.shared_output

            for leaf_idx in range(self.leaves):
                weight = route_info.leaf_weights[token_idx, leaf_idx]
                if bool((weight != 0).item()):
                    out = out + weight * self._leaf_bank_output(token, leaf_idx)

            master_idx = self.master_leaf_index
            if master_idx is not None:
                out = out + self._leaf_bank_output(token, master_idx)

            fallback_idx = self.fallback_leaf_index
            if fallback_idx is not None and fallback_weight != 0.0:
                out = out + fallback_weight * self._leaf_bank_output(token, fallback_idx)

            if route_output_count > 0:
                route_values, route_outputs = self._route_output_values_and_vectors(
                    route_info,
                    token_idx,
                )
                for row_offset in range(route_output_count):
                    out = out + route_values[row_offset] * route_outputs[row_offset]

            if self.bias is not None:
                out = out + self.bias
            outputs.append(out)

        out_flat = torch.stack(outputs, dim=0) if outputs else flat.new_empty(0, self.out_features)
        return self._finalize_forward_output(out_flat, leading_shape, x)

    def route(self, x: Tensor, *, hard: bool | None = None) -> FFFRouteInfo:
        flat, leading_shape = self._flatten_input(x)
        route_info = self._route_flat(
            flat,
            hard=self.config.hard_routing if hard is None else hard,
        )
        diagnostics = self._diagnostics_from_route(route_info, leading_shape=leading_shape)
        return FFFRouteInfo(
            leaf_ids=route_info.leaf_ids.reshape(*leading_shape),
            leaf_probs=route_info.leaf_probs.reshape(*leading_shape, self.leaves),
            leaf_weights=route_info.leaf_weights.reshape(*leading_shape, self.leaves),
            node_ids=route_info.node_ids.reshape(*leading_shape, self.depth),
            route_bits=route_info.route_bits.reshape(*leading_shape, self.depth),
            route_logits=route_info.route_logits.reshape(*leading_shape, self.depth, 2),
            route_values=route_info.route_values.reshape(*leading_shape, self.depth, self.route_rows),
            route_row_ids=route_info.route_row_ids.reshape(
                *leading_shape,
                self.depth,
                self.route_rows,
            ),
            route_result_values=route_info.route_result_values.reshape(
                *leading_shape,
                self.depth,
                route_info.route_result_values.shape[-1],
            ),
            route_result_row_ids=route_info.route_result_row_ids.reshape(
                *leading_shape,
                self.depth,
                route_info.route_result_row_ids.shape[-1],
            ),
            diagnostics=diagnostics,
        )

    def diagnostics(self, x: Tensor | None = None) -> dict[str, object]:
        if x is None:
            return {
                "depth": self.depth,
                "leaves": self.leaves,
                "internal_nodes": self.internal_nodes,
                "stored_rows": self.stored_rows,
                "effective_stored_rows": self.effective_stored_rows,
                "effective_trainable_rows": self.effective_trainable_rows,
                "shared_rows": self.shared_rows,
                "route_rows": self.route_rows,
                "route_result_rows": self.route_result_rows,
                "route_rows_contribute": self.config.route_rows_contribute,
                "route_output_contributes": self._route_output_count() > 0,
                "route_row_role": self.route_row_role,
                "route_rows_output_count": self.config.route_rows_output_count,
                "route_rows_output_fraction": self.config.route_rows_output_fraction,
                "leaf_rows": self.leaf_rows,
                "max_visited_route_rows_per_token": self.max_visited_route_rows_per_token,
                "max_route_output_rows_per_token": self.max_route_output_rows_per_token,
                "stored_route_output_rows": self.stored_route_output_rows,
                "effective_route_output_rows": self.effective_route_output_rows,
                "unused_route_output_rows": self.unused_route_output_rows,
                "unused_stored_route_output_rows": self.unused_stored_route_output_rows,
                "route_output_rows_per_node": self._route_output_rows_per_node_selected(),
                "route_output_rows_per_token": self.route_output_rows_per_token,
                "active_rows_per_token": self._static_active_rows_per_token(),
                "grouped_leaf_path": (
                    "selected_leaf" if self._can_use_selected_leaf_grouped_path() else "all_leaves"
                ),
            }
        route_info = self._route_flat(self._flatten_input(x)[0], hard=self.config.hard_routing)
        leading_shape = x.shape[:-1]
        return self._diagnostics_from_route(route_info, leading_shape=leading_shape)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"depth={self.depth}, shared_rows={self.shared_rows}, route_rows={self.route_rows}, "
            f"route_result_rows={self.route_result_rows}, leaf_rows={self.leaf_rows}, "
            f"route_row_role={self.route_row_role}, activation={self.config.activation}, "
            f"hard_routing={self.config.hard_routing}, bias={self.bias is not None}"
        )

    def _reset_input_bank(self, weight: Tensor, bias: Tensor | None) -> None:
        rows = weight.numel() // self.in_features
        nn.init.kaiming_uniform_(weight.reshape(rows, self.in_features), a=math.sqrt(5))
        if bias is not None:
            bound = 1.0 / math.sqrt(self.in_features)
            nn.init.uniform_(bias, -bound, bound)

    def _reset_output_bank(self, weight: Tensor) -> None:
        bound = 1.0 / math.sqrt(max(1, self._static_active_rows_per_token()))
        nn.init.uniform_(weight, -bound, bound)

    def _activation(self, values: Tensor) -> Tensor:
        if self.config.activation == "silu":
            return F.silu(values)
        if self.config.activation == "gelu":
            return F.gelu(values)
        if self.config.activation == "relu":
            return F.relu(values)
        raise RuntimeError("validated activation became invalid")

    def _flatten_input(self, x: Tensor) -> tuple[Tensor, torch.Size]:
        if not isinstance(x, Tensor):
            raise TypeError("x must be a torch.Tensor")
        if not x.is_floating_point():
            raise TypeError("x must be a floating point tensor")
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError(f"x must have shape [..., {self.in_features}]")
        return x.reshape(-1, self.in_features), x.shape[:-1]

    def _finalize_forward_output(
        self,
        out: Tensor,
        leading_shape: torch.Size,
        original_input: Tensor,
    ) -> Tensor:
        target_dtype = _active_autocast_dtype(original_input.device.type)
        if target_dtype is not None and out.dtype != target_dtype:
            out = out.to(dtype=target_dtype)
        return out.reshape(*leading_shape, self.out_features).contiguous()

    def _route_flat(self, flat: Tensor, *, hard: bool) -> _FlatRouteInfo:
        route_preacts = (
            torch.einsum("ni,mri->nmr", flat, self.route_weight) + self.route_bias
        )
        branch_logits = self._branch_logits(route_preacts)
        if hard:
            leaf_ids, node_ids, route_bits, route_logits = self._hard_route_from_logits(
                branch_logits
            )
            leaf_probs = F.one_hot(leaf_ids, num_classes=self.leaves).to(flat.dtype)
        else:
            leaf_probs = self._soft_leaf_probs(branch_logits)
            leaf_ids = leaf_probs.argmax(dim=-1)
            node_ids, route_bits = self._path_from_leaf_ids(leaf_ids)
            batch = torch.arange(flat.shape[0], device=flat.device)
            route_logits = branch_logits[batch[:, None], node_ids]

        batch = torch.arange(flat.shape[0], device=flat.device)
        route_values = self._activation(route_preacts[batch[:, None], node_ids])
        route_row_offsets = torch.arange(self.route_rows, device=flat.device)
        route_row_ids = node_ids.unsqueeze(-1) * self.route_rows + route_row_offsets
        route_result_values = self._route_result_values(flat, node_ids)
        selected_result_rows = route_result_values.shape[-1]
        result_row_offsets = torch.arange(selected_result_rows, device=flat.device)
        route_result_row_ids = node_ids.unsqueeze(-1) * self.route_result_rows + result_row_offsets
        leaf_weights, fallback_weight = self._regular_leaf_weights(leaf_probs)
        active_rows = self._active_rows_per_token(leaf_weights, fallback_weight=fallback_weight)
        return _FlatRouteInfo(
            leaf_ids=leaf_ids,
            leaf_probs=leaf_probs,
            leaf_weights=leaf_weights,
            node_ids=node_ids,
            route_bits=route_bits,
            route_logits=route_logits,
            route_values=route_values,
            route_row_ids=route_row_ids,
            route_result_values=route_result_values,
            route_result_row_ids=route_result_row_ids,
            active_rows_per_token=active_rows,
        )

    def _branch_logits(self, route_preacts: Tensor) -> Tensor:
        if self.route_rows == 1:
            score = route_preacts[..., 0]
            return torch.stack((-score, score), dim=-1)
        return route_preacts

    def _route_result_values(self, flat: Tensor, node_ids: Tensor) -> Tensor:
        if self.config.route_row_role != "split_routing_output":
            return flat.new_empty(flat.shape[0], self.depth, 0)
        rows_per_node = self._route_output_rows_per_node_selected()
        if rows_per_node == 0:
            return flat.new_empty(flat.shape[0], self.depth, 0)
        if self.route_result_weight is None or self.route_result_bias is None:
            raise RuntimeError("route_result rows are required for split_routing_output")
        result_preacts = (
            torch.einsum("ni,mri->nmr", flat, self.route_result_weight[:, :rows_per_node])
            + self.route_result_bias[:, :rows_per_node]
        )
        batch = torch.arange(flat.shape[0], device=flat.device)
        return self._activation(result_preacts[batch[:, None], node_ids])

    def _hard_route_from_logits(
        self,
        branch_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = branch_logits.shape[0]
        node = torch.zeros(batch_size, dtype=torch.long, device=branch_logits.device)
        batch = torch.arange(batch_size, device=branch_logits.device)
        node_ids: list[Tensor] = []
        bits: list[Tensor] = []
        logits: list[Tensor] = []

        for _ in range(self.depth):
            current_logits = branch_logits[batch, node]
            bit = current_logits.argmax(dim=-1)
            node_ids.append(node)
            bits.append(bit)
            logits.append(current_logits)
            node = node * 2 + 1 + bit

        leaf_ids = node - self.internal_nodes
        return (
            leaf_ids,
            torch.stack(node_ids, dim=1),
            torch.stack(bits, dim=1),
            torch.stack(logits, dim=1),
        )

    def _soft_leaf_probs(self, branch_logits: Tensor) -> Tensor:
        temperature = self.config.train_temperature if self.training else 1.0
        branch_probs = torch.softmax(branch_logits / temperature, dim=-1)
        batch_size = branch_logits.shape[0]
        frontier = [(0, branch_logits.new_ones(batch_size))]
        leaf_probs: list[Tensor | None] = [None] * self.leaves

        for depth_idx in range(self.depth):
            next_frontier: list[tuple[int, Tensor]] = []
            for node_idx, prob_here in frontier:
                left_prob = prob_here * branch_probs[:, node_idx, 0]
                right_prob = prob_here * branch_probs[:, node_idx, 1]
                left_child = 2 * node_idx + 1
                right_child = left_child + 1
                if depth_idx == self.depth - 1:
                    leaf_probs[left_child - self.internal_nodes] = left_prob
                    leaf_probs[right_child - self.internal_nodes] = right_prob
                else:
                    next_frontier.append((left_child, left_prob))
                    next_frontier.append((right_child, right_prob))
            frontier = next_frontier

        return torch.stack([prob for prob in leaf_probs if prob is not None], dim=-1)

    def _path_from_leaf_ids(self, leaf_ids: Tensor) -> tuple[Tensor, Tensor]:
        bits_by_depth = [
            (leaf_ids >> shift).bitwise_and(1)
            for shift in range(self.depth - 1, -1, -1)
        ]
        node = torch.zeros_like(leaf_ids)
        node_ids: list[Tensor] = []
        for bit in bits_by_depth:
            node_ids.append(node)
            node = node * 2 + 1 + bit
        return torch.stack(node_ids, dim=1), torch.stack(bits_by_depth, dim=1)

    def _regular_leaf_weights(self, leaf_probs: Tensor) -> tuple[Tensor, float]:
        leak = float(self.config.region_leak)
        if leak == 0.0:
            return leaf_probs, 0.0
        if self.config.fallback_leaf:
            return leaf_probs * (1.0 - leak), leak
        return leaf_probs * (1.0 - leak) + leak / float(self.leaves), 0.0

    def _fallback_weight(self) -> float:
        if not self.config.fallback_leaf:
            return 0.0
        return float(self.config.region_leak)

    def _leaf_bank_output(self, token: Tensor, bank_idx: int) -> Tensor:
        values = self._activation(self.leaf_weight[bank_idx] @ token + self.leaf_bias[bank_idx])
        return values @ self.leaf_output[bank_idx]

    def _regular_leaf_output_grouped(self, flat: Tensor, route_info: _FlatRouteInfo) -> Tensor:
        if self._can_use_selected_leaf_grouped_path():
            return self._selected_leaf_output_grouped(flat, route_info)
        if (
            self.config.hard_routing
            and self.config.region_leak > 0.0
            and not self.config.fallback_leaf
        ):
            selected = self._selected_leaf_raw_output_grouped(flat, route_info)
            uniform = self._uniform_regular_leaf_output_grouped(flat)
            leak = float(self.config.region_leak)
            return selected.mul(1.0 - leak).add(uniform, alpha=leak)

        leaf_values = self._activation(
            torch.einsum("ni,lri->nlr", flat, self.leaf_weight[: self.leaves])
            + self.leaf_bias[: self.leaves]
        )
        leaf_outputs = torch.einsum("nlr,lro->nlo", leaf_values, self.leaf_output[: self.leaves])
        return (leaf_outputs * route_info.leaf_weights.unsqueeze(-1)).sum(dim=1)

    def _can_use_selected_leaf_grouped_path(self) -> bool:
        if not self.config.hard_routing:
            return False
        if self.config.region_leak == 0.0:
            return True
        return self.config.fallback_leaf

    def _selected_leaf_output_grouped(self, flat: Tensor, route_info: _FlatRouteInfo) -> Tensor:
        if flat.shape[0] == 0:
            return flat.new_empty(0, self.out_features)
        if self.config.fallback_leaf and self.config.region_leak == 1.0:
            return flat.new_zeros(flat.shape[0], self.out_features)

        weights = route_info.leaf_weights.gather(1, route_info.leaf_ids.unsqueeze(1)).squeeze(1)
        selected_weight = self.leaf_weight[route_info.leaf_ids]
        selected_bias = self.leaf_bias[route_info.leaf_ids]
        selected_output = self.leaf_output[route_info.leaf_ids]
        values = self._activation(
            torch.bmm(selected_weight, flat.unsqueeze(-1)).squeeze(-1) + selected_bias
        )
        outputs = torch.bmm(values.unsqueeze(1), selected_output).squeeze(1)
        return outputs * weights.unsqueeze(-1)

    def _selected_leaf_raw_output_grouped(self, flat: Tensor, route_info: _FlatRouteInfo) -> Tensor:
        if flat.shape[0] == 0:
            return flat.new_empty(0, self.out_features)
        selected_weight = self.leaf_weight[route_info.leaf_ids]
        selected_bias = self.leaf_bias[route_info.leaf_ids]
        selected_output = self.leaf_output[route_info.leaf_ids]
        values = self._activation(
            torch.bmm(selected_weight, flat.unsqueeze(-1)).squeeze(-1) + selected_bias
        )
        return torch.bmm(values.unsqueeze(1), selected_output).squeeze(1)

    def _uniform_regular_leaf_output_grouped(self, flat: Tensor) -> Tensor:
        if flat.shape[0] == 0:
            return flat.new_empty(0, self.out_features)
        out = flat.new_zeros(flat.shape[0], self.out_features)
        # Keep the leak path memory bounded: avoid materializing [N, leaves, out].
        leaf_chunk = 4
        for start in range(0, self.leaves, leaf_chunk):
            stop = min(start + leaf_chunk, self.leaves)
            values = self._activation(
                torch.einsum("ni,lri->nlr", flat, self.leaf_weight[start:stop])
                + self.leaf_bias[start:stop]
            )
            out = out + torch.einsum("nlr,lro->no", values, self.leaf_output[start:stop])
        return out / float(self.leaves)

    def _extra_leaf_output_grouped(self, flat: Tensor, *, fallback_weight: float) -> Tensor:
        out = flat.new_zeros(flat.shape[0], self.out_features)
        master_idx = self.master_leaf_index
        if master_idx is not None:
            values = self._activation(
                F.linear(flat, self.leaf_weight[master_idx], self.leaf_bias[master_idx])
            )
            out = out + values @ self.leaf_output[master_idx]

        fallback_idx = self.fallback_leaf_index
        if fallback_idx is not None and fallback_weight != 0.0:
            values = self._activation(
                F.linear(flat, self.leaf_weight[fallback_idx], self.leaf_bias[fallback_idx])
            )
            out = out + fallback_weight * (values @ self.leaf_output[fallback_idx])
        return out

    def _route_output_grouped(self, route_info: _FlatRouteInfo) -> Tensor:
        output_count = self._route_output_count()
        if output_count == 0:
            return route_info.route_values.new_zeros(route_info.route_values.shape[0], self.out_features)
        flat_values, flat_outputs = self._route_output_values_and_vectors(route_info)
        return torch.einsum(
            "nr,nro->no",
            flat_values[:, :output_count],
            flat_outputs[:, :output_count],
        )

    def _route_output_values_and_vectors(
        self,
        route_info: _FlatRouteInfo,
        token_idx: int | None = None,
    ) -> tuple[Tensor, Tensor]:
        rows_per_node = self._route_output_rows_per_node_selected()
        if self.config.route_row_role == "shared_routing_and_output":
            if self.route_output is None:
                raise RuntimeError("route_output is required for shared_routing_and_output")
            values = route_info.route_values
            vectors = self.route_output[route_info.node_ids]
        elif self.config.route_row_role == "split_routing_output":
            if self.route_result_output is None:
                raise RuntimeError("route_result_output is required for split_routing_output")
            values = route_info.route_result_values
            vectors = self.route_result_output[:, :rows_per_node][route_info.node_ids]
        else:
            raise RuntimeError("routing_only has no route output rows")

        values = values[:, :, :rows_per_node]
        vectors = vectors[:, :, :rows_per_node]
        if token_idx is not None:
            return (
                values[token_idx].reshape(-1),
                vectors[token_idx].reshape(-1, self.out_features),
            )
        route_rows = values.shape[1] * values.shape[2]
        return (
            values.reshape(values.shape[0], route_rows),
            vectors.reshape(values.shape[0], route_rows, self.out_features),
        )

    def _route_output_rows_per_node(self) -> int:
        if self.config.route_row_role == "shared_routing_and_output":
            return self.route_rows
        if self.config.route_row_role == "split_routing_output":
            return self.route_result_rows
        return 0

    def _route_output_count(self) -> int:
        return self.depth * self._route_output_rows_per_node_selected()

    def _route_output_rows_per_node_selected(self) -> int:
        if self.config.route_row_role == "routing_only":
            return 0

        max_rows = self._route_output_rows_per_node()
        count_candidates: list[int] = []
        if self.config.route_rows_output_count is None:
            if self.config.route_rows_output_fraction is None:
                count_candidates.append(max_rows)
        elif self.config.route_rows_output_count == "all":
            count_candidates.append(max_rows)
        else:
            count_candidates.append(min(int(self.config.route_rows_output_count), max_rows))

        if self.config.route_rows_output_fraction is not None:
            fraction = float(self.config.route_rows_output_fraction)
            count_candidates.append(min(math.ceil(fraction * max_rows), max_rows))

        if not count_candidates:
            return 0
        return min(count_candidates)

    def _active_rows_per_token(self, leaf_weights: Tensor, *, fallback_weight: float) -> Tensor:
        active_regular_leaf_rows = (leaf_weights != 0).sum(dim=-1) * self.leaf_rows
        active = active_regular_leaf_rows + self.shared_rows + self._route_output_count()
        if self.config.master_leaf:
            active = active + self.leaf_rows
        if self.config.fallback_leaf and fallback_weight != 0.0:
            active = active + self.leaf_rows
        return active.to(torch.long)

    def _static_active_rows_per_token(self) -> int:
        route_rows = self._route_output_count()
        if self.config.hard_routing:
            if self.config.fallback_leaf:
                regular_leaf_rows = 0 if self.config.region_leak == 1.0 else self.leaf_rows
            elif self.config.region_leak > 0.0:
                regular_leaf_rows = self.leaves * self.leaf_rows
            else:
                regular_leaf_rows = self.leaf_rows
        else:
            regular_leaf_rows = self.leaves * self.leaf_rows
        master_rows = self.leaf_rows if self.config.master_leaf else 0
        fallback_rows = (
            self.leaf_rows
            if self.config.fallback_leaf and self.config.region_leak != 0.0
            else 0
        )
        return self.shared_rows + regular_leaf_rows + master_rows + fallback_rows + route_rows

    def _diagnostics_from_route(
        self,
        route_info: _FlatRouteInfo,
        *,
        leading_shape: torch.Size,
    ) -> dict[str, object]:
        active = route_info.active_rows_per_token.reshape(*leading_shape)
        if route_info.active_rows_per_token.numel() == 0:
            mean_active = float(self._static_active_rows_per_token())
        else:
            mean_active = float(route_info.active_rows_per_token.float().mean().item())
        return {
            "depth": self.depth,
            "leaves": self.leaves,
            "internal_nodes": self.internal_nodes,
            "stored_rows": self.stored_rows,
            "effective_stored_rows": self.effective_stored_rows,
            "effective_trainable_rows": self.effective_trainable_rows,
            "shared_rows": self.shared_rows,
            "route_rows": self.route_rows,
            "route_result_rows": self.route_result_rows,
            "route_rows_contribute": self.config.route_rows_contribute,
            "route_output_contributes": self._route_output_count() > 0,
            "route_row_role": self.route_row_role,
            "route_rows_output_count": self.config.route_rows_output_count,
            "route_rows_output_fraction": self.config.route_rows_output_fraction,
            "leaf_rows": self.leaf_rows,
            "max_visited_route_rows_per_token": self.max_visited_route_rows_per_token,
            "max_route_output_rows_per_token": self.max_route_output_rows_per_token,
            "stored_route_output_rows": self.stored_route_output_rows,
            "effective_route_output_rows": self.effective_route_output_rows,
            "unused_route_output_rows": self.unused_route_output_rows,
            "unused_stored_route_output_rows": self.unused_stored_route_output_rows,
            "route_output_rows_per_node": self._route_output_rows_per_node_selected(),
            "route_output_rows_per_token": self._route_output_count(),
            "active_rows_per_token": active,
            "mean_active_rows_per_token": mean_active,
            "grouped_leaf_path": (
                "selected_leaf" if self._can_use_selected_leaf_grouped_path() else "all_leaves"
            ),
        }
