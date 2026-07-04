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


def _dense_linear_flops(in_features: int, out_features: int) -> int:
    return 2 * in_features * out_features


def _route_usage(ids: Tensor, count: int) -> tuple[Tensor, int]:
    if ids.numel() == 0:
        usage = torch.zeros(count, device=ids.device, dtype=torch.long)
    else:
        usage = torch.bincount(ids.reshape(-1), minlength=count)
    return usage, int((usage == 0).sum().item())


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
        return flat_output.reshape(*leading_shape, self.out_features)

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
        block_scores = F.softmax(block_values, dim=-1)
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
        active_columns = self.column_blocks_per_token * self.block_size
        column_ids = route_info.column_ids.reshape(flat_input.shape[0], active_columns)
        selected_weight = self.weight[column_ids]
        values = torch.einsum("ni,nci->nc", flat_input, selected_weight)
        if self.bias is not None:
            values = values + self.bias[column_ids]
        flat_output = torch.zeros(
            flat_input.shape[0],
            self.out_features,
            device=input.device,
            dtype=input.dtype,
        )
        flat_output = _scatter_columns(flat_output, column_ids, values)
        return flat_output.reshape(*leading_shape, self.out_features)

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
        block_scores = F.softmax(block_values, dim=-1)
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
        selected_output = self.row_output[row_ids]
        selected_columns = torch.gather(
            selected_output,
            dim=2,
            index=column_ids.unsqueeze(1).expand(-1, self.rows_per_token, -1),
        )
        values = torch.einsum("nr,nrc->nc", row_scores, selected_columns)
        if self.bias is not None:
            values = values + self.bias[column_ids]
        flat_output = torch.zeros(
            flat_input.shape[0],
            self.out_features,
            device=input.device,
            dtype=input.dtype,
        )
        flat_output = _scatter_columns(flat_output, column_ids, values)
        return flat_output.reshape(*leading_shape, self.out_features)

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

    def __init__(
        self,
        in_features: int,
        out_features: int,
        row_experts: int,
        rows_per_token: int,
        column_blocks: int,
        column_blocks_per_token: int,
        *,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _require_positive_int("in_features", in_features)
        _require_positive_int("row_experts", row_experts)
        _validate_topk("rows_per_token", rows_per_token, row_experts)
        _validate_column_blocks(out_features, column_blocks, column_blocks_per_token)
        _require_bool("bias", bias)
        self.in_features = in_features
        self.out_features = out_features
        self.row_experts = row_experts
        self.rows_per_token = rows_per_token
        self.column_blocks = column_blocks
        self.column_blocks_per_token = column_blocks_per_token
        self.block_size = out_features // column_blocks

        factory_kwargs = {"device": device, "dtype": dtype}
        self.row_router_weight = nn.Parameter(
            torch.empty(row_experts, in_features, **factory_kwargs)
        )
        self.row_router_bias = nn.Parameter(torch.empty(row_experts, **factory_kwargs))
        self.column_router_weight = nn.Parameter(
            torch.empty(column_blocks, in_features, **factory_kwargs)
        )
        self.column_router_bias = nn.Parameter(torch.empty(column_blocks, **factory_kwargs))
        self.expert_weight = nn.Parameter(
            torch.empty(
                row_experts,
                column_blocks,
                self.block_size,
                in_features,
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
        nn.init.kaiming_uniform_(self.row_router_weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.column_router_weight, a=math.sqrt(5))
        for row in range(self.row_experts):
            for block in range(self.column_blocks):
                nn.init.kaiming_uniform_(self.expert_weight[row, block], a=math.sqrt(5))
        bound = 1 / math.sqrt(self.in_features)
        nn.init.uniform_(self.row_router_bias, -bound, bound)
        nn.init.uniform_(self.column_router_bias, -bound, bound)
        if self.expert_bias is not None:
            nn.init.uniform_(self.expert_bias, -bound, bound)

    def route(self, input: Tensor) -> SparseRowColumnRouteInfo:
        flat_input, leading_shape = _validate_input(input, self.in_features)
        row_logits = F.linear(flat_input, self.row_router_weight, self.row_router_bias)
        row_values, row_ids = row_logits.topk(self.rows_per_token, dim=-1)
        row_scores = F.softmax(row_values, dim=-1)
        block_logits = F.linear(flat_input, self.column_router_weight, self.column_router_bias)
        block_values, block_ids = block_logits.topk(self.column_blocks_per_token, dim=-1)
        block_scores = F.softmax(block_values, dim=-1)
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

        selected_weight = self.expert_weight[row_ids.unsqueeze(-1), block_ids.unsqueeze(1)]
        values = torch.einsum("ni,nrcbi->nrcb", flat_input, selected_weight)
        if self.expert_bias is not None:
            selected_bias = self.expert_bias[row_ids.unsqueeze(-1), block_ids.unsqueeze(1)]
            values = values + selected_bias
        values = values * row_scores[:, :, None, None] * block_scores[:, None, :, None]
        block_values = values.sum(dim=1)
        flat_output = torch.zeros(
            flat_input.shape[0],
            self.out_features,
            device=input.device,
            dtype=input.dtype,
        )
        flat_output = _scatter_columns(flat_output, column_ids, block_values)
        return flat_output.reshape(*leading_shape, self.out_features)

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
        return {
            "mode": "checkerboard_moe",
            "stored_rows": self.row_experts,
            "stored_columns": self.out_features,
            "column_blocks": self.column_blocks,
            "column_block_size": self.block_size,
            "active_rows_per_token": self.rows_per_token,
            "active_column_blocks_per_token": self.column_blocks_per_token,
            "active_columns_per_token": active_columns,
            "active_intersections_per_token": active_experts,
            "estimated_active_flops_per_token": 2
            * active_experts
            * self.block_size
            * self.in_features,
            "estimated_routing_flops_per_token": 2
            * (self.row_experts + self.column_blocks)
            * self.in_features,
            "estimated_dense_flops_per_token": _dense_linear_flops(
                self.in_features,
                self.out_features,
            ),
            "implementation_path": "checkerboard_expert_grid_topk_scatter",
        }

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
