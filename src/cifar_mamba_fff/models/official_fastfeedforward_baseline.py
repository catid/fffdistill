from __future__ import annotations

import importlib
import math
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class OfficialFFFCapability:
    """Shape and budget metadata for an official ``fastfeedforward.FFF`` candidate."""

    input_width: int
    output_width: int
    depth: int
    leaf_width: int
    parameter_budget: int
    trainable_parameters: int
    registered_parameters: int
    stored_rows: int
    active_rows_per_token_eval: int
    compatible: bool
    reason: str

    def log_record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class OfficialFFFRegressionResult:
    """Deterministic bounded single-layer evidence for an official FFF baseline."""

    capability: OfficialFFFCapability
    train_steps: int
    token_count: int
    mse: float
    normalized_mse: float
    cosine_similarity: float
    tokens_per_second: float
    output_shape: tuple[int, ...]

    def log_record(self) -> dict[str, object]:
        record = asdict(self)
        record["capability"] = self.capability.log_record()
        return record


def _require_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _require_non_negative_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def official_fff_trainable_parameter_count(
    input_width: int,
    leaf_width: int,
    output_width: int,
    depth: int,
) -> int:
    """Return trainable parameter count for upstream ``fastfeedforward.FFF``.

    The upstream module always has routing node weights/biases and per-leaf two-layer
    MLP weights/biases. Its ``depth`` scalar is registered as a frozen parameter and
    is intentionally excluded here.
    """

    _require_positive_int("input_width", input_width)
    _require_positive_int("leaf_width", leaf_width)
    _require_positive_int("output_width", output_width)
    _require_non_negative_int("depth", depth)
    leaves = 1 << depth
    nodes = leaves - 1
    node_parameters = nodes * (input_width + 1)
    leaf_parameters = leaves * (
        input_width * leaf_width + leaf_width + leaf_width * output_width + output_width
    )
    return node_parameters + leaf_parameters


def official_fff_registered_parameter_count(
    input_width: int,
    leaf_width: int,
    output_width: int,
    depth: int,
) -> int:
    """Return all registered upstream parameters, including frozen ``depth``."""

    return official_fff_trainable_parameter_count(input_width, leaf_width, output_width, depth) + 1


def official_fff_stored_rows(leaf_width: int, depth: int) -> int:
    """Return the upstream FFF stored route-node plus leaf-hidden row count."""

    _require_positive_int("leaf_width", leaf_width)
    _require_non_negative_int("depth", depth)
    leaves = 1 << depth
    return (leaves - 1) + leaves * leaf_width


def official_fff_active_rows_per_token_eval(leaf_width: int, depth: int) -> int:
    """Return eval-time active route-node plus selected leaf-hidden rows per token."""

    _require_positive_int("leaf_width", leaf_width)
    _require_non_negative_int("depth", depth)
    return depth + leaf_width


def official_fff_leaf_width_for_parameter_budget(
    input_width: int,
    output_width: int,
    parameter_budget: int,
    *,
    depth: int,
) -> int:
    """Choose the largest upstream ``leaf_width`` fitting a trainable parameter budget."""

    _require_positive_int("input_width", input_width)
    _require_positive_int("output_width", output_width)
    _require_positive_int("parameter_budget", parameter_budget)
    _require_non_negative_int("depth", depth)
    leaves = 1 << depth
    nodes = leaves - 1
    fixed_parameters = nodes * (input_width + 1) + leaves * output_width
    parameters_per_leaf_width = leaves * (input_width + 1 + output_width)
    leaf_width = (parameter_budget - fixed_parameters) // parameters_per_leaf_width
    if leaf_width < 1:
        minimum = official_fff_trainable_parameter_count(input_width, 1, output_width, depth)
        raise ValueError(
            f"parameter_budget={parameter_budget} cannot fit official fastfeedforward.FFF "
            f"with depth={depth}; minimum trainable parameters are {minimum}"
        )
    return leaf_width


def detect_official_fff_capability(
    input_width: int,
    output_width: int,
    parameter_budget: int,
    *,
    depth: int = 1,
    require_eval: bool = True,
) -> OfficialFFFCapability:
    """Return shape/budget compatibility for an upstream FFF Linear replacement.

    ``fastfeedforward==0.2.1`` preserves ``[..., input_width] -> [..., output_width]``
    for positive widths. Depth 0 is deliberately marked incompatible when eval is
    required because upstream ``eval_forward`` raises before selecting a leaf.
    """

    _require_positive_int("input_width", input_width)
    _require_positive_int("output_width", output_width)
    _require_positive_int("parameter_budget", parameter_budget)
    _require_non_negative_int("depth", depth)
    if require_eval and depth == 0:
        return OfficialFFFCapability(
            input_width=input_width,
            output_width=output_width,
            depth=depth,
            leaf_width=0,
            parameter_budget=parameter_budget,
            trainable_parameters=0,
            registered_parameters=0,
            stored_rows=0,
            active_rows_per_token_eval=0,
            compatible=False,
            reason=(
                "fastfeedforward.FFF depth=0 is train-shape-compatible but eval_forward "
                "fails in the installed API; use depth >= 1 for baseline evaluation"
            ),
        )
    try:
        leaf_width = official_fff_leaf_width_for_parameter_budget(
            input_width,
            output_width,
            parameter_budget,
            depth=depth,
        )
    except ValueError as exc:
        return OfficialFFFCapability(
            input_width=input_width,
            output_width=output_width,
            depth=depth,
            leaf_width=0,
            parameter_budget=parameter_budget,
            trainable_parameters=0,
            registered_parameters=0,
            stored_rows=0,
            active_rows_per_token_eval=0,
            compatible=False,
            reason=str(exc),
        )
    trainable = official_fff_trainable_parameter_count(
        input_width,
        leaf_width,
        output_width,
        depth,
    )
    return OfficialFFFCapability(
        input_width=input_width,
        output_width=output_width,
        depth=depth,
        leaf_width=leaf_width,
        parameter_budget=parameter_budget,
        trainable_parameters=trainable,
        registered_parameters=official_fff_registered_parameter_count(
            input_width,
            leaf_width,
            output_width,
            depth,
        ),
        stored_rows=official_fff_stored_rows(leaf_width, depth),
        active_rows_per_token_eval=official_fff_active_rows_per_token_eval(leaf_width, depth),
        compatible=True,
        reason="shape-compatible official fastfeedforward.FFF candidate",
    )


def make_official_fff(
    input_width: int,
    leaf_width: int,
    output_width: int,
    depth: int,
    **kwargs: object,
) -> nn.Module:
    module = importlib.import_module("fastfeedforward")
    cls = getattr(module, "FFF", None)
    if cls is None:
        raise RuntimeError("fastfeedforward imported, but FFF symbol was not found")
    try:
        return cls(
            input_width=input_width,
            leaf_width=leaf_width,
            output_width=output_width,
            depth=depth,
            **kwargs,
        )
    except TypeError as exc:
        raise RuntimeError(
            "Installed fastfeedforward.FFF is not shape-compatible with "
            f"(input_width={input_width}, leaf_width={leaf_width}, "
            f"output_width={output_width}, depth={depth}) and kwargs {kwargs}"
        ) from exc


def make_matched_official_fff(
    linear: nn.Linear,
    *,
    parameter_budget: int,
    depth: int = 1,
    require_eval: bool = True,
    **kwargs: object,
) -> tuple[nn.Module, OfficialFFFCapability]:
    """Create a shape-compatible upstream FFF replacement for ``linear`` if budget permits."""

    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
    capability = detect_official_fff_capability(
        linear.in_features,
        linear.out_features,
        parameter_budget,
        depth=depth,
        require_eval=require_eval,
    )
    if not capability.compatible:
        raise ValueError(capability.reason)
    module = make_official_fff(
        input_width=capability.input_width,
        leaf_width=capability.leaf_width,
        output_width=capability.output_width,
        depth=capability.depth,
        **kwargs,
    )
    module.to(device=linear.weight.device, dtype=linear.weight.dtype)
    module.train(linear.training)
    return module, capability


def forward_smoke(module: nn.Module, input_width: int, device: str = "cuda") -> tuple[int, ...]:
    x = torch.randn(2, input_width, device=device)
    y = module.to(device)(x)
    return tuple(y.shape)


def _cosine_similarity(prediction: Tensor, target: Tensor) -> float:
    if prediction.numel() == 0:
        return math.nan
    cosine = F.cosine_similarity(prediction.reshape(1, -1), target.reshape(1, -1), dim=1)
    return float(cosine.item())


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_official_fff_layer_regression_baseline(
    linear: nn.Linear,
    inputs: Tensor,
    *,
    parameter_budget: int | None = None,
    depth: int = 1,
    train_steps: int = 0,
    lr: float = 1e-3,
    seed: int = 0,
    timing_iterations: int = 3,
    timing_warmup: int = 1,
) -> OfficialFFFRegressionResult:
    """Run a bounded deterministic single-layer official FFF regression harness.

    This is intentionally a small shape/budget harness, not a CIFAR-10 validation
    run. It trains only the official FFF module against one provided activation
    tensor and reports layerwise MSE/cosine plus local forward throughput.
    """

    if not isinstance(linear, nn.Linear):
        raise TypeError(f"expected nn.Linear, got {type(linear).__name__}")
    if inputs.shape[-1] != linear.in_features:
        raise ValueError(
            f"inputs last dimension must be {linear.in_features}, got {inputs.shape[-1]}"
        )
    _require_non_negative_int("train_steps", train_steps)
    _require_positive_int("timing_iterations", timing_iterations)
    _require_non_negative_int("timing_warmup", timing_warmup)
    if lr <= 0:
        raise ValueError("lr must be positive")

    budget = parameter_budget
    if budget is None:
        budget = sum(parameter.numel() for parameter in linear.parameters())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        official, capability = make_matched_official_fff(
            linear,
            parameter_budget=budget,
            depth=depth,
            require_eval=True,
        )
    inputs = inputs.detach().to(device=linear.weight.device, dtype=linear.weight.dtype).contiguous()
    official = official.to(device=inputs.device, dtype=inputs.dtype)
    with torch.no_grad():
        targets = linear(inputs).detach()

    if train_steps > 0:
        official.train()
        optimizer = torch.optim.AdamW(official.parameters(), lr=lr)
        for _ in range(train_steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = official(inputs)
            loss = F.mse_loss(prediction, targets)
            loss.backward()
            optimizer.step()

    official.eval()
    with torch.no_grad():
        for _ in range(timing_warmup):
            official(inputs)
        _synchronize_if_cuda(inputs.device)
        start = time.perf_counter()
        for _ in range(timing_iterations):
            prediction = official(inputs)
        _synchronize_if_cuda(inputs.device)
        elapsed = time.perf_counter() - start

    mse = float(F.mse_loss(prediction, targets).item())
    target_energy = float(targets.float().pow(2).mean().item())
    normalized_mse = mse / target_energy if target_energy > 0.0 else math.inf
    token_count = inputs.reshape(-1, inputs.shape[-1]).shape[0]
    return OfficialFFFRegressionResult(
        capability=capability,
        train_steps=train_steps,
        token_count=token_count,
        mse=mse,
        normalized_mse=normalized_mse,
        cosine_similarity=_cosine_similarity(prediction.float(), targets.float()),
        tokens_per_second=(token_count * timing_iterations) / elapsed if elapsed > 0.0 else math.inf,
        output_shape=tuple(prediction.shape),
    )
