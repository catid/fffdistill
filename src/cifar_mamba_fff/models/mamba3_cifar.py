from __future__ import annotations

import importlib
from dataclasses import dataclass

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


@dataclass(frozen=True)
class Mamba3CifarConfig:
    d_model: int = 224
    depth: int = 18
    patch_size: int = 2
    d_state: int = 128
    expand: int = 2
    headdim: int = 64
    is_mimo: bool = True
    mimo_rank: int = 4
    chunk_size: int = 16
    bidirectional: bool = False
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
        seq_len = (32 // self.patch_size) ** 2
        if seq_len % self.chunk_size != 0:
            raise ValueError(
                "official Mamba3 TileLang path expects the CIFAR patch sequence length to be "
                f"divisible by chunk_size; got seq_len={seq_len}, chunk_size={self.chunk_size}"
            )


class BidirectionalMamba3Block(nn.Module):
    def __init__(self, forward_block: nn.Module, reverse_block: nn.Module) -> None:
        super().__init__()
        self.forward_block = forward_block
        self.reverse_block = reverse_block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fwd = self.forward_block(x)
        y_rev = torch.flip(self.reverse_block(torch.flip(x, dims=[1])), dims=[1])
        if isinstance(y_fwd, tuple):
            y_fwd = y_fwd[0]
        if isinstance(y_rev, tuple):
            y_rev = y_rev[0]
        return 0.5 * (y_fwd + y_rev)


class Mamba3CifarTeacher(nn.Module):
    def __init__(self, config: Mamba3CifarConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        mamba3_cls = find_official_mamba3()
        self.patch = nn.Conv2d(3, config.d_model, kernel_size=config.patch_size, stride=config.patch_size)
        seq_len = (32 // config.patch_size) ** 2
        self.pos = nn.Parameter(torch.zeros(1, seq_len, config.d_model))
        self.blocks = nn.ModuleList([self._make_block(mamba3_cls, config) for _ in range(config.depth)])
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    @staticmethod
    def _make_one(cls: type, config: Mamba3CifarConfig) -> nn.Module:
        base_kwargs = {
            "d_model": config.d_model,
            "d_state": config.d_state,
            "expand": config.expand,
            "headdim": config.headdim,
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

    @classmethod
    def _make_block(cls, mamba3_cls: type, config: Mamba3CifarConfig) -> nn.Module:
        if not config.bidirectional:
            return cls._make_one(mamba3_cls, config)
        return BidirectionalMamba3Block(cls._make_one(mamba3_cls, config), cls._make_one(mamba3_cls, config))

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch(x).flatten(2).transpose(1, 2)
        x = x + self.pos
        for block in self.blocks:
            x = block(x)
            if isinstance(x, tuple):
                x = x[0]
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
