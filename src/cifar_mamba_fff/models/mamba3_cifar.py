from __future__ import annotations

import importlib
from dataclasses import dataclass
from functools import partial

import torch
from torch import nn

from cifar_mamba_fff.metrics import count_parameters


def find_official_mamba3() -> type:
    candidates = (
        ("mamba_ssm", "Mamba3"),
        ("mamba_ssm.modules.mamba3", "Mamba3"),
        ("mamba_ssm.modules.mamba3_simple", "Mamba3"),
        ("mamba_ssm.modules.mamba3_mimo", "Mamba3"),
    )
    errors: list[str] = []
    for module_name, attr in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - optional dependency
            errors.append(f"{module_name}: {exc}")
            continue
        cls = getattr(module, attr, None)
        if cls is not None:
            return cls
        errors.append(f"{module_name}: missing {attr}")
    raise RuntimeError(
        "Official Mamba3 was not found. Do not fall back to Mamba2 or a fake block. "
        + "; ".join(errors)
    )


def find_official_block() -> type:
    try:
        module = importlib.import_module("mamba_ssm.modules.block")
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Official mamba_ssm.modules.block.Block was not found. "
            "Do not use a fake residual wrapper for Mamba3."
        ) from exc
    block_cls = getattr(module, "Block", None)
    if block_cls is None:
        raise RuntimeError("Official mamba_ssm.modules.block is missing Block")
    return block_cls


@dataclass(frozen=True)
class Mamba3CifarConfig:
    d_model: int = 256
    depth: int = 20
    patch_size: int = 4
    d_state: int = 64
    expand: int = 2
    headdim: int = 64
    is_mimo: bool = True
    mimo_rank: int = 2
    chunk_size: int = 16
    bidirectional: bool = False
    drop_path: float = 0.1
    norm_epsilon: float = 1e-5
    residual_in_fp32: bool = True
    num_classes: int = 10
    target_min_params: int = 9_000_000
    target_max_params: int = 11_000_000

    def validate(self) -> None:
        if 32 % self.patch_size != 0:
            raise ValueError("patch_size must divide CIFAR-10 image size 32")
        if self.depth <= 0 or self.d_model <= 0:
            raise ValueError("depth and d_model must be positive")
        if self.headdim <= 0 or self.d_state <= 0:
            raise ValueError("headdim and d_state must be positive")
        if self.expand <= 0:
            raise ValueError("expand must be positive")
        d_inner = self.d_model * self.expand
        if d_inner % self.headdim != 0:
            raise ValueError(
                "official Mamba3 requires d_model * expand to be divisible by headdim; "
                f"got d_model={self.d_model}, expand={self.expand}, headdim={self.headdim}"
            )
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= self.drop_path < 1.0:
            raise ValueError("drop_path must be in [0, 1)")
        if self.norm_epsilon <= 0.0:
            raise ValueError("norm_epsilon must be positive")
        seq_len = (32 // self.patch_size) ** 2
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                "official Mamba3 TileLang path expects the CIFAR patch sequence length to be "
                f"divisible by chunk_size; got seq_len={seq_len}, chunk_size={self.chunk_size}"
            )


class DropPath(nn.Module):
    def __init__(self, probability: float) -> None:
        super().__init__()
        if not 0.0 <= probability < 1.0:
            raise ValueError("drop path probability must be in [0, 1)")
        self.probability = float(probability)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.probability == 0.0 or not self.training:
            return x
        keep_probability = 1.0 - self.probability
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.empty(mask_shape, dtype=x.dtype, device=x.device).bernoulli_(keep_probability)
        return x * mask.div(keep_probability)


class Mamba3DropPathMixer(nn.Module):
    def __init__(self, mixer: nn.Module, drop_path: float) -> None:
        super().__init__()
        self.mixer = mixer
        self.drop_path = DropPath(drop_path)

    def forward(self, hidden_states: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        hidden_states = self.mixer(hidden_states, *args, **kwargs)
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
        return self.drop_path(hidden_states)

    def allocate_inference_cache(
        self,
        batch_size: int,
        max_seqlen: int,
        dtype: torch.dtype | None = None,
        **kwargs: object,
    ) -> object:
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


class BidirectionalMamba3Block(nn.Module):
    def __init__(self, forward_block: nn.Module, reverse_block: nn.Module) -> None:
        super().__init__()
        self.forward_block = forward_block
        self.reverse_block = reverse_block

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        y_fwd, residual_fwd = self.forward_block(hidden_states, residual, **kwargs)
        reversed_hidden = torch.flip(hidden_states, dims=[1])
        reversed_residual = torch.flip(residual, dims=[1]) if residual is not None else None
        y_rev, residual_rev = self.reverse_block(reversed_hidden, reversed_residual, **kwargs)
        y_rev = torch.flip(y_rev, dims=[1])
        residual_rev = torch.flip(residual_rev, dims=[1])
        return 0.5 * (y_fwd + y_rev), 0.5 * (residual_fwd + residual_rev)


class Mamba3CifarTeacher(nn.Module):
    def __init__(self, config: Mamba3CifarConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        mamba3_cls = find_official_mamba3()
        block_cls = find_official_block()
        self.patch = nn.Conv2d(3, config.d_model, kernel_size=config.patch_size, stride=config.patch_size)
        seq_len = (32 // config.patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, seq_len, config.d_model))
        self.blocks = nn.ModuleList(
            [
                self._make_block(block_cls, mamba3_cls, config, layer_idx=layer_idx)
                for layer_idx in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.d_model, eps=config.norm_epsilon)
        self.head = nn.Linear(config.d_model, config.num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    @staticmethod
    def _make_one(cls: type, config: Mamba3CifarConfig, *, layer_idx: int) -> nn.Module:
        base_kwargs = {
            "d_model": config.d_model,
            "d_state": config.d_state,
            "expand": config.expand,
            "headdim": config.headdim,
            "layer_idx": layer_idx,
            "n_layer": config.depth,
        }
        full_kwargs = {
            **base_kwargs,
            "is_mimo": config.is_mimo,
            "mimo_rank": config.mimo_rank,
            "chunk_size": config.chunk_size,
        }
        constructor_attempts = [full_kwargs]
        if not config.is_mimo:
            constructor_attempts.append(
                {**base_kwargs, "is_mimo": config.is_mimo, "mimo_rank": config.mimo_rank}
            )
            constructor_attempts.append(base_kwargs)

        failures: list[str] = []
        for kwargs in constructor_attempts:
            try:
                module = cls(**kwargs)
            except TypeError as exc:
                failures.append(f"{kwargs}: {exc}")
                continue
            if config.is_mimo and getattr(module, "is_mimo", config.is_mimo) is not True:
                raise RuntimeError("Official Mamba3 constructor did not preserve requested MIMO mode")
            if config.is_mimo and getattr(module, "mimo_rank", None) != config.mimo_rank:
                raise RuntimeError("Official Mamba3 constructor did not preserve requested mimo_rank")
            if getattr(module, "chunk_size", None) != config.chunk_size:
                raise RuntimeError("Official Mamba3 constructor did not preserve requested chunk_size")
            return module
        raise RuntimeError(f"Could not instantiate official Mamba3 with config {config}: {failures}")

    @staticmethod
    def _drop_path_for_layer(config: Mamba3CifarConfig, layer_idx: int) -> float:
        if config.depth <= 1:
            return config.drop_path
        return config.drop_path * layer_idx / float(config.depth - 1)

    @classmethod
    def _make_residual_block(
        cls,
        block_cls: type,
        mamba3_cls: type,
        config: Mamba3CifarConfig,
        *,
        layer_idx: int,
    ) -> nn.Module:
        drop_path = cls._drop_path_for_layer(config, layer_idx)

        def mixer_factory(_dim: int) -> nn.Module:
            return Mamba3DropPathMixer(
                cls._make_one(mamba3_cls, config, layer_idx=layer_idx),
                drop_path,
            )

        norm_factory = partial(nn.LayerNorm, eps=config.norm_epsilon)
        return block_cls(
            config.d_model,
            mixer_factory,
            nn.Identity,
            norm_cls=norm_factory,
            fused_add_norm=False,
            residual_in_fp32=config.residual_in_fp32,
        )

    @classmethod
    def _make_block(
        cls,
        block_cls: type,
        mamba3_cls: type,
        config: Mamba3CifarConfig,
        *,
        layer_idx: int,
    ) -> nn.Module:
        if not config.bidirectional:
            return cls._make_residual_block(block_cls, mamba3_cls, config, layer_idx=layer_idx)
        return BidirectionalMamba3Block(
            cls._make_residual_block(block_cls, mamba3_cls, config, layer_idx=layer_idx),
            cls._make_residual_block(block_cls, mamba3_cls, config, layer_idx=layer_idx),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = x + self.pos
        residual = None
        for block in self.blocks:
            x, residual = block(x, residual)
        x = (x + residual) if residual is not None else x
        x = x.to(dtype=self.norm.weight.dtype)
        return self.norm(x).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))

    def assert_target_parameter_count(self) -> int:
        total = count_parameters(self)
        if not (self.config.target_min_params <= total <= self.config.target_max_params):
            raise ValueError(
                f"teacher parameter count {total:,} is outside "
                f"[{self.config.target_min_params:,}, {self.config.target_max_params:,}]"
            )
        return total
