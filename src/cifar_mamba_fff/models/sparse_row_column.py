"""Sparse row/column Linear and checkerboard MoE prototypes.

These modules are intentionally small baselines for design and accounting work.
They preserve the public ``nn.Linear`` shape ``[..., in_features] -> [..., out_features]``
while routing each token to sparse row banks, sparse output-column blocks, or both.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ActivationName = Literal["silu", "gelu", "relu"]
CheckerboardRouterType = Literal["linear", "mlp"]


@dataclass(frozen=True)
class SparseRowColumnRouteInfo:
    """Routing selections and diagnostics for sparse row/column prototypes."""

    row_ids: Tensor
    row_scores: Tensor
    column_block_ids: Tensor
    column_scores: Tensor
    column_ids: Tensor
    diagnostics: dict[str, object]


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_bool(name: str, value: bool) -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a bool")


def _activation(name: ActivationName, x: Tensor) -> Tensor:
    if name == "silu":
        return F.silu(x)
    if name == "gelu":
        return F.gelu(x)
    if name == "relu":
        return F.relu(x)
    raise ValueError("activation must be one of: silu, gelu, relu")


def _validate_input(input: Tensor, in_features: int) -> tuple[Tensor, torch.Size]:
    if input.shape[-1] != in_features:
        raise ValueError(
            f"expected input last dimension {in_features}, got {input.shape[-1]}"
        )
    leading_shape = input.shape[:-1]
    return input.reshape(-1, in_features), leading_shape


def _validate_topk(name: str, count: int, limit: int) -> None:
    _require_positive_int(name, count)
    if count > limit:
        raise ValueError(f"{name} must be <= {limit}")


def _validate_column_blocks(
    out_features: int,
    column_blocks: int,
    column_blocks_per_token: int,
) -> None:
    _require_positive_int("out_features", out_features)
    _require_positive_int("column_blocks", column_blocks)
    if out_features % column_blocks != 0:
        raise ValueError("out_features must be divisible by column_blocks for this prototype")
    _validate_topk("column_blocks_per_token", column_blocks_per_token, column_blocks)


def _column_ids_from_block_ids(block_ids: Tensor, block_size: int) -> Tensor:
    offsets = torch.arange(block_size, device=block_ids.device, dtype=block_ids.dtype)
    return block_ids.unsqueeze(-1) * block_size + offsets


def _scatter_columns(flat_output: Tensor, column_ids: Tensor, values: Tensor) -> Tensor:
    active_columns = math.prod(column_ids.shape[1:])
    return flat_output.scatter_add(
        1,
        column_ids.reshape(column_ids.shape[0], active_columns),
        values.reshape(values.shape[0], active_columns),
    )


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


def _forward_output_dtype(input: Tensor) -> torch.dtype:
    return _active_autocast_dtype(input.device.type) or input.dtype


def _finalize_forward_output(flat_output: Tensor, leading_shape: torch.Size, input: Tensor) -> Tensor:
    target_dtype = _forward_output_dtype(input)
    if flat_output.dtype != target_dtype:
        flat_output = flat_output.to(dtype=target_dtype)
    return flat_output.reshape(*leading_shape, flat_output.shape[-1]).contiguous()


def _dense_linear_flops(in_features: int, out_features: int) -> int:
    return 2 * in_features * out_features


def _route_usage(ids: Tensor, count: int) -> tuple[Tensor, int]:
    if ids.numel() == 0:
        usage = torch.zeros(count, device=ids.device, dtype=torch.long)
    else:
        usage = torch.bincount(ids.reshape(-1), minlength=count)
    return usage, int((usage == 0).sum().item())


def _selected_softmax_ste_scores(
    logits: Tensor,
    ids: Tensor,
    values: Tensor,
    *,
    forward: Literal["ones", "topk_softmax"],
) -> Tensor:
    """Return selected router scores with a full-router surrogate gradient."""

    selected_probs = F.softmax(logits, dim=-1).gather(-1, ids)
    if forward == "ones":
        forward_scores = torch.ones_like(selected_probs)
    elif forward == "topk_softmax":
        forward_scores = F.softmax(values, dim=-1)
    else:  # pragma: no cover - Literal keeps this unreachable in typed callers.
        raise ValueError("forward must be ones or topk_softmax")
    return forward_scores + selected_probs - selected_probs.detach()


class SparseRowLinear(nn.Module):
    """Token-choice row-bank baseline with dense output mixing."""

    in_features: int
    out_features: int
    row_banks: int
    rows_per_token: int
    activation: ActivationName
    row_weight: nn.Parameter
    row_bias: nn.Parameter
    row_output: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        row_banks: int,
        rows_per_token: int,
        *,
        activation: ActivationName = "silu",
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require_positive_int("in_features", in_features)
        _require_positive_int("out_features", out_features)
        _require_positive_int("row_banks", row_banks)
        _validate_topk("rows_per_token", rows_per_token, row_banks)
        _activation(activation, torch.empty(0))
        _require_bool("bias", bias)
        self.in_features = in_features
        self.out_features = out_features
        self.row_banks = row_banks
        self.rows_per_token = rows_per_token
        self.activation = activation

        factory_kwargs = {"device": device, "dtype": dtype}
        self.row_weight = nn.Parameter(torch.empty(row_banks, in_features, **factory_kwargs))
        self.row_bias = nn.Parameter(torch.empty(row_banks, **factory_kwargs))
        self.row_output = nn.Parameter(torch.empty(row_banks, out_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.row_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.row_bias, -bound, bound)
        nn.init.kaiming_uniform_(self.row_output, a=math.sqrt(5))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def route(self, input: Tensor) -> SparseRowColumnRouteInfo:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        row_logits = F.linear(flat_input, self.row_weight, self.row_bias)
        row_values, row_ids = row_logits.topk(self.rows_per_token, dim=-1)
        row_scores = _activation(self.activation, row_values)
        diagnostics = self._diagnostics_from_route(row_ids, input.shape[:-1])
        return SparseRowColumnRouteInfo(
            row_ids=row_ids.reshape(*leading_shape, self.rows_per_token),
            row_scores=row_scores.reshape(*leading_shape, self.rows_per_token),
            column_block_ids=torch.empty(*leading_shape, 0, dtype=torch.long, device=input.device),
            column_scores=torch.empty(*leading_shape, 0, dtype=input.dtype, device=input.device),
            column_ids=torch.empty(*leading_shape, 0, dtype=torch.long, device=input.device),
            diagnostics=diagnostics,
        )

    def forward(self, input: Tensor) -> Tensor:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        row_logits = F.linear(flat_input, self.row_weight, self.row_bias)
        row_values, row_ids = row_logits.topk(self.rows_per_token, dim=-1)
        row_scores = _activation(self.activation, row_values)
        selected_output = self.row_output[row_ids]
        flat_output = torch.einsum("nr,nro->no", row_scores, selected_output)
        if self.bias is not None:
            flat_output = flat_output + self.bias
        return _finalize_forward_output(flat_output, leading_shape, input)

    def diagnostics(self, input: Tensor | None = None) -> dict[str, object]:
        if input is None:
            return self._base_diagnostics()
        route_info = self.route(input)
        return route_info.diagnostics

    def _diagnostics_from_route(
        self,
        row_ids: Tensor,
        leading_shape: torch.Size,
    ) -> dict[str, object]:
        diagnostics = self._base_diagnostics()
        usage, dead_rows = _route_usage(row_ids, self.row_banks)
        diagnostics.update(
            {
                "tokens": math.prod(leading_shape),
                "row_usage": usage,
                "dead_rows": dead_rows,
            }
        )
        return diagnostics

    def _base_diagnostics(self) -> dict[str, object]:
        active_flops = 2 * self.rows_per_token * (self.in_features + self.out_features)
        return {
            "mode": "row_only",
            "stored_rows": self.row_banks,
            "stored_columns": self.out_features,
            "active_rows_per_token": self.rows_per_token,
            "active_column_blocks_per_token": 0,
            "active_columns_per_token": self.out_features,
            "active_intersections_per_token": self.rows_per_token,
            "estimated_active_flops_per_token": active_flops,
            "estimated_routing_flops_per_token": 2 * self.row_banks * self.in_features,
            "estimated_dense_flops_per_token": _dense_linear_flops(
                self.in_features,
                self.out_features,
            ),
            "implementation_path": "dense_router_topk_gather",
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class SparseColumnLinear(nn.Module):
    """Output-column-block sparse Linear baseline."""

    in_features: int
    out_features: int
    column_blocks: int
    column_blocks_per_token: int
    block_size: int
    column_router_weight: nn.Parameter
    column_router_bias: nn.Parameter
    weight: nn.Parameter
    bias: nn.Parameter | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        column_blocks: int,
        column_blocks_per_token: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require_positive_int("in_features", in_features)
        _validate_column_blocks(out_features, column_blocks, column_blocks_per_token)
        _require_bool("bias", bias)
        self.in_features = in_features
        self.out_features = out_features
        self.column_blocks = column_blocks
        self.column_blocks_per_token = column_blocks_per_token
        self.block_size = out_features // column_blocks

        factory_kwargs = {"device": device, "dtype": dtype}
        self.column_router_weight = nn.Parameter(
            torch.empty(column_blocks, in_features, **factory_kwargs)
        )
        self.column_router_bias = nn.Parameter(torch.empty(column_blocks, **factory_kwargs))
        self.weight = nn.Parameter(torch.empty(out_features, in_features, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.column_router_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.column_router_bias, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def route(self, input: Tensor) -> SparseRowColumnRouteInfo:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        block_logits = F.linear(flat_input, self.column_router_weight, self.column_router_bias)
        block_values, block_ids = block_logits.topk(self.column_blocks_per_token, dim=-1)
        block_scores = _selected_softmax_ste_scores(
            block_logits,
            block_ids,
            block_values,
            forward="ones",
        )
        active_columns = self.column_blocks_per_token * self.block_size
        column_ids = _column_ids_from_block_ids(block_ids, self.block_size).reshape(
            flat_input.shape[0],
            active_columns,
        )
        diagnostics = self._diagnostics_from_route(block_ids, input.shape[:-1])
        return SparseRowColumnRouteInfo(
            row_ids=torch.empty(*leading_shape, 0, dtype=torch.long, device=input.device),
            row_scores=torch.empty(*leading_shape, 0, dtype=input.dtype, device=input.device),
            column_block_ids=block_ids.reshape(*leading_shape, self.column_blocks_per_token),
            column_scores=block_scores.reshape(*leading_shape, self.column_blocks_per_token),
            column_ids=column_ids.reshape(
                *leading_shape,
                self.column_blocks_per_token * self.block_size,
            ),
            diagnostics=diagnostics,
        )

    def forward(self, input: Tensor) -> Tensor:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        route_info = self.route(input)
        block_ids = route_info.column_block_ids.reshape(
            flat_input.shape[0],
            self.column_blocks_per_token,
        )
        block_scores = route_info.column_scores.reshape(
            flat_input.shape[0],
            self.column_blocks_per_token,
        )
        output_dtype = _forward_output_dtype(input)
        flat_output = torch.zeros(
            flat_input.shape[0],
            self.out_features,
            device=input.device,
            dtype=output_dtype,
        )
        for block_position in range(self.column_blocks_per_token):
            selected_blocks = block_ids[:, block_position]
            for block_id in selected_blocks.unique(sorted=True):
                block_index = int(block_id.item())
                token_mask = selected_blocks == block_index
                start = block_index * self.block_size
                end = start + self.block_size
                block_bias = self.bias[start:end] if self.bias is not None else None
                block_values = F.linear(
                    flat_input[token_mask],
                    self.weight[start:end],
                    block_bias,
                ).to(dtype=output_dtype)
                block_values = block_values * block_scores[token_mask, block_position].unsqueeze(-1)
                block_values = block_values.to(dtype=output_dtype)
                flat_output[token_mask, start:end] = flat_output[token_mask, start:end] + block_values
        return _finalize_forward_output(flat_output, leading_shape, input)

    def diagnostics(self, input: Tensor | None = None) -> dict[str, object]:
        if input is None:
            return self._base_diagnostics()
        return self.route(input).diagnostics

    def _diagnostics_from_route(
        self,
        block_ids: Tensor,
        leading_shape: torch.Size,
    ) -> dict[str, object]:
        diagnostics = self._base_diagnostics()
        usage, dead_blocks = _route_usage(block_ids, self.column_blocks)
        diagnostics.update(
            {
                "tokens": math.prod(leading_shape),
                "column_block_usage": usage,
                "dead_column_blocks": dead_blocks,
            }
        )
        return diagnostics

    def _base_diagnostics(self) -> dict[str, object]:
        active_columns = self.column_blocks_per_token * self.block_size
        return {
            "mode": "column_only",
            "stored_rows": self.out_features,
            "stored_columns": self.out_features,
            "column_blocks": self.column_blocks,
            "column_block_size": self.block_size,
            "active_rows_per_token": 0,
            "active_column_blocks_per_token": self.column_blocks_per_token,
            "active_columns_per_token": active_columns,
            "active_intersections_per_token": active_columns,
            "estimated_active_flops_per_token": 2 * active_columns * self.in_features,
            "estimated_routing_flops_per_token": 2 * self.column_blocks * self.in_features,
            "estimated_dense_flops_per_token": _dense_linear_flops(
                self.in_features,
                self.out_features,
            ),
            "implementation_path": "column_block_topk_scatter",
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CoupledRowColumnLinear(nn.Module):
    """Sparse row-bank activations coupled to sparse output-column blocks."""

    in_features: int
    out_features: int
    row_banks: int
    rows_per_token: int
    column_blocks: int
    column_blocks_per_token: int
    block_size: int
    activation: ActivationName

    def __init__(
        self,
        in_features: int,
        out_features: int,
        row_banks: int,
        rows_per_token: int,
        column_blocks: int,
        column_blocks_per_token: int,
        *,
        activation: ActivationName = "silu",
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require_positive_int("in_features", in_features)
        _require_positive_int("row_banks", row_banks)
        _validate_topk("rows_per_token", rows_per_token, row_banks)
        _validate_column_blocks(out_features, column_blocks, column_blocks_per_token)
        _activation(activation, torch.empty(0))
        _require_bool("bias", bias)
        self.in_features = in_features
        self.out_features = out_features
        self.row_banks = row_banks
        self.rows_per_token = rows_per_token
        self.column_blocks = column_blocks
        self.column_blocks_per_token = column_blocks_per_token
        self.block_size = out_features // column_blocks
        self.activation = activation

        factory_kwargs = {"device": device, "dtype": dtype}
        self.row_weight = nn.Parameter(torch.empty(row_banks, in_features, **factory_kwargs))
        self.row_bias = nn.Parameter(torch.empty(row_banks, **factory_kwargs))
        self.row_output = nn.Parameter(torch.empty(row_banks, out_features, **factory_kwargs))
        self.column_router_weight = nn.Parameter(
            torch.empty(column_blocks, in_features, **factory_kwargs)
        )
        self.column_router_bias = nn.Parameter(torch.empty(column_blocks, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.row_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.row_output, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.column_router_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.row_bias, -bound, bound)
        nn.init.uniform_(self.column_router_bias, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def route(self, input: Tensor) -> SparseRowColumnRouteInfo:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        row_logits = F.linear(flat_input, self.row_weight, self.row_bias)
        row_values, row_ids = row_logits.topk(self.rows_per_token, dim=-1)
        row_scores = _activation(self.activation, row_values)
        block_logits = F.linear(flat_input, self.column_router_weight, self.column_router_bias)
        block_values, block_ids = block_logits.topk(self.column_blocks_per_token, dim=-1)
        block_scores = _selected_softmax_ste_scores(
            block_logits,
            block_ids,
            block_values,
            forward="ones",
        )
        active_columns = self.column_blocks_per_token * self.block_size
        column_ids = _column_ids_from_block_ids(block_ids, self.block_size).reshape(
            flat_input.shape[0],
            active_columns,
        )
        diagnostics = self._diagnostics_from_route(row_ids, block_ids, input.shape[:-1])
        return SparseRowColumnRouteInfo(
            row_ids=row_ids.reshape(*leading_shape, self.rows_per_token),
            row_scores=row_scores.reshape(*leading_shape, self.rows_per_token),
            column_block_ids=block_ids.reshape(*leading_shape, self.column_blocks_per_token),
            column_scores=block_scores.reshape(*leading_shape, self.column_blocks_per_token),
            column_ids=column_ids.reshape(
                *leading_shape,
                self.column_blocks_per_token * self.block_size,
            ),
            diagnostics=diagnostics,
        )

    def forward(self, input: Tensor) -> Tensor:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        route_info = self.route(input)
        row_ids = route_info.row_ids.reshape(flat_input.shape[0], self.rows_per_token)
        row_scores = route_info.row_scores.reshape(flat_input.shape[0], self.rows_per_token)
        active_columns = self.column_blocks_per_token * self.block_size
        column_ids = route_info.column_ids.reshape(flat_input.shape[0], active_columns)
        block_scores = route_info.column_scores.reshape(
            flat_input.shape[0],
            self.column_blocks_per_token,
        )
        selected_output = self.row_output[row_ids]
        selected_columns = torch.gather(
            selected_output,
            dim=2,
            index=column_ids.unsqueeze(1).expand(-1, self.rows_per_token, -1),
        )
        values = torch.einsum("nr,nrc->nc", row_scores, selected_columns)
        if self.bias is not None:
            values = values + self.bias[column_ids]
        values = values * block_scores.repeat_interleave(self.block_size, dim=-1)
        values = values.to(dtype=_forward_output_dtype(input))
        flat_output = torch.zeros(
            flat_input.shape[0],
            self.out_features,
            device=input.device,
            dtype=values.dtype,
        )
        flat_output = _scatter_columns(flat_output, column_ids, values)
        return _finalize_forward_output(flat_output, leading_shape, input)

    def diagnostics(self, input: Tensor | None = None) -> dict[str, object]:
        if input is None:
            return self._base_diagnostics()
        return self.route(input).diagnostics

    def _diagnostics_from_route(
        self,
        row_ids: Tensor,
        block_ids: Tensor,
        leading_shape: torch.Size,
    ) -> dict[str, object]:
        diagnostics = self._base_diagnostics()
        row_usage, dead_rows = _route_usage(row_ids, self.row_banks)
        block_usage, dead_blocks = _route_usage(block_ids, self.column_blocks)
        diagnostics.update(
            {
                "tokens": math.prod(leading_shape),
                "row_usage": row_usage,
                "dead_rows": dead_rows,
                "column_block_usage": block_usage,
                "dead_column_blocks": dead_blocks,
            }
        )
        return diagnostics

    def _base_diagnostics(self) -> dict[str, object]:
        active_columns = self.column_blocks_per_token * self.block_size
        return {
            "mode": "coupled_row_column",
            "stored_rows": self.row_banks,
            "stored_columns": self.out_features,
            "column_blocks": self.column_blocks,
            "column_block_size": self.block_size,
            "active_rows_per_token": self.rows_per_token,
            "active_column_blocks_per_token": self.column_blocks_per_token,
            "active_columns_per_token": active_columns,
            "active_intersections_per_token": self.rows_per_token * active_columns,
            "estimated_active_flops_per_token": 2
            * self.rows_per_token
            * (self.in_features + active_columns),
            "estimated_routing_flops_per_token": 2
            * (self.row_banks + self.column_blocks)
            * self.in_features,
            "estimated_dense_flops_per_token": _dense_linear_flops(
                self.in_features,
                self.out_features,
            ),
            "implementation_path": "row_and_column_topk_intersection_scatter",
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class CheckerboardSparseMoELinear(nn.Module):
    """Checkerboard MoE over row experts and output-column blocks."""

    in_features: int
    out_features: int
    row_experts: int
    rows_per_token: int
    column_blocks: int
    column_blocks_per_token: int
    block_size: int
    expert_rank: int | None
    router_type: CheckerboardRouterType
    router_hidden_features: int | None
    always_on_rows: int

    def __init__(
        self,
        in_features: int,
        out_features: int,
        row_experts: int,
        rows_per_token: int,
        column_blocks: int,
        column_blocks_per_token: int,
        *,
        expert_rank: int | None = None,
        router_type: CheckerboardRouterType = "linear",
        router_hidden_features: int | None = None,
        always_on_rows: int = 0,
        activation: ActivationName = "silu",
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require_positive_int("in_features", in_features)
        _require_positive_int("row_experts", row_experts)
        _validate_topk("rows_per_token", rows_per_token, row_experts)
        _validate_column_blocks(out_features, column_blocks, column_blocks_per_token)
        if expert_rank is not None:
            _require_positive_int("expert_rank", expert_rank)
            if expert_rank > min(in_features, out_features // column_blocks):
                raise ValueError("expert_rank must be <= min(in_features, column block size)")
        if router_type not in {"linear", "mlp"}:
            raise ValueError("router_type must be one of: linear, mlp")
        if router_type == "mlp":
            if router_hidden_features is None:
                raise ValueError("router_hidden_features is required when router_type='mlp'")
            _require_positive_int("router_hidden_features", router_hidden_features)
        elif router_hidden_features is not None:
            raise ValueError("router_hidden_features requires router_type='mlp'")
        if isinstance(always_on_rows, bool) or not isinstance(always_on_rows, int) or always_on_rows < 0:
            raise ValueError("always_on_rows must be a non-negative integer")
        _activation(activation, torch.empty(0))
        _require_bool("bias", bias)
        self.in_features = in_features
        self.out_features = out_features
        self.row_experts = row_experts
        self.rows_per_token = rows_per_token
        self.column_blocks = column_blocks
        self.column_blocks_per_token = column_blocks_per_token
        self.block_size = out_features // column_blocks
        self.expert_rank = expert_rank
        self.router_type = router_type
        self.router_hidden_features = router_hidden_features
        self.always_on_rows = always_on_rows
        self.activation = activation

        factory_kwargs = {"device": device, "dtype": dtype}
        if router_type == "linear":
            self.row_router_weight = nn.Parameter(
                torch.empty(row_experts, in_features, **factory_kwargs)
            )
            self.row_router_bias = nn.Parameter(torch.empty(row_experts, **factory_kwargs))
            self.column_router_weight = nn.Parameter(
                torch.empty(column_blocks, in_features, **factory_kwargs)
            )
            self.column_router_bias = nn.Parameter(torch.empty(column_blocks, **factory_kwargs))
            self.register_parameter("router_hidden_weight", None)
            self.register_parameter("router_hidden_bias", None)
            self.register_parameter("row_router_out_weight", None)
            self.register_parameter("row_router_out_bias", None)
            self.register_parameter("column_router_out_weight", None)
            self.register_parameter("column_router_out_bias", None)
        else:
            if router_hidden_features is None:
                raise RuntimeError("router_hidden_features validation failed")
            self.register_parameter("row_router_weight", None)
            self.register_parameter("row_router_bias", None)
            self.register_parameter("column_router_weight", None)
            self.register_parameter("column_router_bias", None)
            self.router_hidden_weight = nn.Parameter(
                torch.empty(router_hidden_features, in_features, **factory_kwargs)
            )
            self.router_hidden_bias = nn.Parameter(torch.empty(router_hidden_features, **factory_kwargs))
            self.row_router_out_weight = nn.Parameter(
                torch.empty(row_experts, router_hidden_features, **factory_kwargs)
            )
            self.row_router_out_bias = nn.Parameter(torch.empty(row_experts, **factory_kwargs))
            self.column_router_out_weight = nn.Parameter(
                torch.empty(column_blocks, router_hidden_features, **factory_kwargs)
            )
            self.column_router_out_bias = nn.Parameter(torch.empty(column_blocks, **factory_kwargs))
        if always_on_rows:
            self.always_on_weight = nn.Parameter(
                torch.empty(always_on_rows, in_features, **factory_kwargs)
            )
            self.always_on_bias = nn.Parameter(torch.empty(always_on_rows, **factory_kwargs))
            self.always_on_output = nn.Parameter(
                torch.empty(always_on_rows, out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("always_on_weight", None)
            self.register_parameter("always_on_bias", None)
            self.register_parameter("always_on_output", None)
        if expert_rank is None:
            self.expert_weight = nn.Parameter(
                torch.empty(
                    row_experts,
                    column_blocks,
                    self.block_size,
                    in_features,
                    **factory_kwargs,
                )
            )
            self.register_parameter("expert_down", None)
            self.register_parameter("expert_up", None)
        else:
            self.register_parameter("expert_weight", None)
            self.expert_down = nn.Parameter(
                torch.empty(
                    row_experts,
                    column_blocks,
                    expert_rank,
                    in_features,
                    **factory_kwargs,
                )
            )
            self.expert_up = nn.Parameter(
                torch.empty(
                    row_experts,
                    column_blocks,
                    self.block_size,
                    expert_rank,
                    **factory_kwargs,
                )
            )
        if bias:
            self.expert_bias = nn.Parameter(
                torch.empty(row_experts, column_blocks, self.block_size, **factory_kwargs)
            )
        else:
            self.register_parameter("expert_bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.router_type == "linear":
            if (
                self.row_router_weight is None
                or self.column_router_weight is None
                or self.row_router_bias is None
                or self.column_router_bias is None
            ):
                raise RuntimeError("linear checkerboard router parameters are not initialized")
            nn.init.kaiming_uniform_(self.row_router_weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.column_router_weight, a=math.sqrt(5))
        else:
            if (
                self.router_hidden_weight is None
                or self.router_hidden_bias is None
                or self.row_router_out_weight is None
                or self.row_router_out_bias is None
                or self.column_router_out_weight is None
                or self.column_router_out_bias is None
            ):
                raise RuntimeError("MLP checkerboard router parameters are not initialized")
            nn.init.kaiming_uniform_(self.router_hidden_weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.row_router_out_weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.column_router_out_weight, a=math.sqrt(5))
        if self.expert_weight is not None:
            for row in range(self.row_experts):
                for block in range(self.column_blocks):
                    nn.init.kaiming_uniform_(self.expert_weight[row, block], a=math.sqrt(5))
        else:
            if self.expert_down is None or self.expert_up is None:
                raise RuntimeError("factorized checkerboard experts are not initialized")
            for row in range(self.row_experts):
                for block in range(self.column_blocks):
                    nn.init.kaiming_uniform_(self.expert_down[row, block], a=math.sqrt(5))
                    nn.init.kaiming_uniform_(self.expert_up[row, block], a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        if self.router_type == "linear":
            if self.row_router_bias is None or self.column_router_bias is None:
                raise RuntimeError("linear checkerboard router biases are not initialized")
            nn.init.uniform_(self.row_router_bias, -bound, bound)
            nn.init.uniform_(self.column_router_bias, -bound, bound)
        else:
            if (
                self.router_hidden_bias is None
                or self.row_router_out_bias is None
                or self.column_router_out_bias is None
            ):
                raise RuntimeError("MLP checkerboard router biases are not initialized")
            nn.init.uniform_(self.router_hidden_bias, -bound, bound)
            nn.init.uniform_(self.row_router_out_bias, -bound, bound)
            nn.init.uniform_(self.column_router_out_bias, -bound, bound)
        if self.always_on_weight is not None:
            nn.init.kaiming_uniform_(self.always_on_weight, a=math.sqrt(5))
        if self.always_on_output is not None:
            nn.init.kaiming_uniform_(self.always_on_output, a=math.sqrt(5))
        if self.always_on_bias is not None:
            nn.init.uniform_(self.always_on_bias, -bound, bound)
        if self.expert_bias is not None:
            nn.init.uniform_(self.expert_bias, -bound, bound)

    def _router_logits(self, flat_input: Tensor) -> tuple[Tensor, Tensor]:
        if self.router_type == "linear":
            if (
                self.row_router_weight is None
                or self.row_router_bias is None
                or self.column_router_weight is None
                or self.column_router_bias is None
            ):
                raise RuntimeError("linear checkerboard router parameters are not initialized")
            return (
                F.linear(flat_input, self.row_router_weight, self.row_router_bias),
                F.linear(flat_input, self.column_router_weight, self.column_router_bias),
            )
        if (
            self.router_hidden_weight is None
            or self.router_hidden_bias is None
            or self.row_router_out_weight is None
            or self.row_router_out_bias is None
            or self.column_router_out_weight is None
            or self.column_router_out_bias is None
        ):
            raise RuntimeError("MLP checkerboard router parameters are not initialized")
        hidden = _activation(
            self.activation,
            F.linear(flat_input, self.router_hidden_weight, self.router_hidden_bias),
        )
        return (
            F.linear(hidden, self.row_router_out_weight, self.row_router_out_bias),
            F.linear(hidden, self.column_router_out_weight, self.column_router_out_bias),
        )

    def _always_on_output(self, flat_input: Tensor) -> Tensor | None:
        if self.always_on_rows == 0:
            return None
        if (
            self.always_on_weight is None
            or self.always_on_bias is None
            or self.always_on_output is None
        ):
            raise RuntimeError("always-on checkerboard rows are not initialized")
        activations = _activation(
            self.activation,
            F.linear(flat_input, self.always_on_weight, self.always_on_bias),
        )
        return torch.matmul(activations, self.always_on_output).to(
            dtype=_forward_output_dtype(flat_input)
        )

    def route(self, input: Tensor) -> SparseRowColumnRouteInfo:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        row_logits, block_logits = self._router_logits(flat_input)
        row_values, row_ids = row_logits.topk(self.rows_per_token, dim=-1)
        row_scores = _selected_softmax_ste_scores(
            row_logits,
            row_ids,
            row_values,
            forward="topk_softmax",
        )
        block_values, block_ids = block_logits.topk(self.column_blocks_per_token, dim=-1)
        block_scores = _selected_softmax_ste_scores(
            block_logits,
            block_ids,
            block_values,
            forward="topk_softmax",
        )
        active_columns = self.column_blocks_per_token * self.block_size
        column_ids = _column_ids_from_block_ids(block_ids, self.block_size).reshape(
            flat_input.shape[0],
            active_columns,
        )
        diagnostics = self._diagnostics_from_route(row_ids, block_ids, input.shape[:-1])
        return SparseRowColumnRouteInfo(
            row_ids=row_ids.reshape(*leading_shape, self.rows_per_token),
            row_scores=row_scores.reshape(*leading_shape, self.rows_per_token),
            column_block_ids=block_ids.reshape(*leading_shape, self.column_blocks_per_token),
            column_scores=block_scores.reshape(*leading_shape, self.column_blocks_per_token),
            column_ids=column_ids.reshape(
                *leading_shape,
                self.column_blocks_per_token * self.block_size,
            ),
            diagnostics=diagnostics,
        )

    def forward(self, input: Tensor) -> Tensor:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        route_info = self.route(input)
        row_ids = route_info.row_ids.reshape(flat_input.shape[0], self.rows_per_token)
        row_scores = route_info.row_scores.reshape(flat_input.shape[0], self.rows_per_token)
        block_ids = route_info.column_block_ids.reshape(
            flat_input.shape[0],
            self.column_blocks_per_token,
        )
        block_scores = route_info.column_scores.reshape(
            flat_input.shape[0],
            self.column_blocks_per_token,
        )
        column_ids = _column_ids_from_block_ids(block_ids, self.block_size)

        if self.expert_weight is not None:
            selected_weight = self.expert_weight[row_ids.unsqueeze(-1), block_ids.unsqueeze(1)]
            values = torch.einsum("ni,nrcbi->nrcb", flat_input, selected_weight)
        else:
            if self.expert_down is None or self.expert_up is None:
                raise RuntimeError("factorized checkerboard experts are not initialized")
            selected_down = self.expert_down[row_ids.unsqueeze(-1), block_ids.unsqueeze(1)]
            selected_up = self.expert_up[row_ids.unsqueeze(-1), block_ids.unsqueeze(1)]
            hidden = torch.einsum("ni,nrcki->nrck", flat_input, selected_down)
            values = torch.einsum("nrck,nrcbk->nrcb", hidden, selected_up)
        if self.expert_bias is not None:
            selected_bias = self.expert_bias[row_ids.unsqueeze(-1), block_ids.unsqueeze(1)]
            values = values + selected_bias
        values = values * row_scores[:, :, None, None] * block_scores[:, None, :, None]
        block_values = values.sum(dim=1)
        block_values = block_values.to(dtype=_forward_output_dtype(input))
        always_output = self._always_on_output(flat_input)
        if always_output is None:
            flat_output = torch.zeros(
                flat_input.shape[0],
                self.out_features,
                device=input.device,
                dtype=block_values.dtype,
            )
        else:
            flat_output = always_output.to(dtype=block_values.dtype)
        flat_output = _scatter_columns(flat_output, column_ids, block_values)
        return _finalize_forward_output(flat_output, leading_shape, input)

    def diagnostics(self, input: Tensor | None = None) -> dict[str, object]:
        if input is None:
            return self._base_diagnostics()
        return self.route(input).diagnostics

    def _diagnostics_from_route(
        self,
        row_ids: Tensor,
        block_ids: Tensor,
        leading_shape: torch.Size,
    ) -> dict[str, object]:
        diagnostics = self._base_diagnostics()
        row_usage, dead_rows = _route_usage(row_ids, self.row_experts)
        block_usage, dead_blocks = _route_usage(block_ids, self.column_blocks)
        if row_ids.numel() == 0:
            expert_usage = torch.zeros(
                self.row_experts,
                self.column_blocks,
                dtype=torch.long,
                device=row_ids.device,
            )
        else:
            pair_ids = (
                row_ids.unsqueeze(-1) * self.column_blocks + block_ids.unsqueeze(1)
            ).reshape(-1)
            expert_usage = torch.bincount(
                pair_ids,
                minlength=self.row_experts * self.column_blocks,
            ).reshape(self.row_experts, self.column_blocks)
        diagnostics.update(
            {
                "tokens": math.prod(leading_shape),
                "row_usage": row_usage,
                "dead_rows": dead_rows,
                "column_block_usage": block_usage,
                "dead_column_blocks": dead_blocks,
                "expert_usage": expert_usage,
                "dead_experts": int((expert_usage == 0).sum().item()),
            }
        )
        return diagnostics

    def _base_diagnostics(self) -> dict[str, object]:
        active_columns = self.column_blocks_per_token * self.block_size
        active_experts = self.rows_per_token * self.column_blocks_per_token
        if self.expert_rank is None:
            expert_flops = 2 * self.block_size * self.in_features
        else:
            expert_flops = 2 * self.expert_rank * (self.in_features + self.block_size)
        if self.router_type == "linear":
            router_flops = 2 * (self.row_experts + self.column_blocks) * self.in_features
        else:
            router_hidden = self.router_hidden_features or 0
            router_flops = 2 * router_hidden * self.in_features + 2 * (
                self.row_experts + self.column_blocks
            ) * router_hidden
        always_on_flops = 2 * self.always_on_rows * (self.in_features + self.out_features)
        return {
            "mode": "checkerboard_moe",
            "stored_rows": self.row_experts + self.always_on_rows,
            "stored_columns": self.out_features,
            "column_blocks": self.column_blocks,
            "column_block_size": self.block_size,
            "factorized_experts": self.expert_rank is not None,
            "expert_rank": self.expert_rank or min(self.in_features, self.block_size),
            "router_type": self.router_type,
            "router_hidden_features": self.router_hidden_features or 0,
            "always_on_rows": self.always_on_rows,
            "active_rows_per_token": self.rows_per_token + self.always_on_rows,
            "active_sparse_rows_per_token": self.rows_per_token,
            "active_always_on_rows_per_token": self.always_on_rows,
            "active_column_blocks_per_token": self.column_blocks_per_token,
            "active_columns_per_token": active_columns,
            "active_intersections_per_token": active_experts,
            "active_sparse_tiles_per_token": active_experts,
            "estimated_active_flops_per_token": active_experts * expert_flops + always_on_flops,
            "estimated_routing_flops_per_token": router_flops,
            "estimated_total_flops_per_token": active_experts * expert_flops
            + always_on_flops
            + router_flops,
            "estimated_dense_flops_per_token": _dense_linear_flops(
                self.in_features,
                self.out_features,
            ),
            "implementation_path": "checkerboard_expert_grid_topk_scatter",
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
