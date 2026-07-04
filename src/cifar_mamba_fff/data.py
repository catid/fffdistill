from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

CIFAR10_TRAIN_SIZE = 50_000
CIFAR10_TEST_SIZE = 10_000
CIFAR10_VAL_SIZE = 5_000
CIFAR10_DISTILL_TRAIN_SIZE = CIFAR10_TRAIN_SIZE - CIFAR10_VAL_SIZE
CIFAR10_SMOKE_TRAIN_SIZE = 1_024
CIFAR10_SMOKE_VAL_SIZE = 256
CIFAR10_SMOKE_TEST_SIZE = 256


@dataclass(frozen=True)
class Cifar10DataConfig:
    data_dir: Path = Path("data/cifar10")
    batch_size: int = 512
    num_workers: int = 8
    seed: int = 1337
    train_size: int = CIFAR10_DISTILL_TRAIN_SIZE
    val_size: int = CIFAR10_VAL_SIZE
    download: bool = True
    quick_smoke: bool = False
    smoke_train_size: int = CIFAR10_SMOKE_TRAIN_SIZE
    smoke_val_size: int = CIFAR10_SMOKE_VAL_SIZE
    smoke_test_size: int = CIFAR10_SMOKE_TEST_SIZE
    randaugment: bool = False
    label_smoothing: float = 0.0
    mixup: float = 0.0
    cutmix: float = 0.0
    use_test: bool = False

    def validate(self, *, allow_test: bool = False) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        _validate_positive_int("train_size", self.train_size)
        _validate_positive_int("val_size", self.val_size)
        _validate_positive_int("smoke_train_size", self.smoke_train_size)
        _validate_positive_int("smoke_val_size", self.smoke_val_size)
        _validate_positive_int("smoke_test_size", self.smoke_test_size)
        _validate_train_val_size(self.train_size, self.val_size)
        _validate_train_val_size(self.smoke_train_size, self.smoke_val_size)
        if self.smoke_test_size > CIFAR10_TEST_SIZE:
            raise ValueError("smoke_test_size cannot exceed CIFAR-10 test size")
        _validate_probability("label_smoothing", self.label_smoothing, upper_inclusive=False)
        _validate_nonnegative_finite("mixup", self.mixup)
        _validate_nonnegative_finite("cutmix", self.cutmix)
        if self.use_test and not allow_test:
            raise ValueError("CIFAR-10 test access is disabled until final validation-selected runs")


def _import_torchvision():
    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - depends on optional install
        raise RuntimeError(
            "torchvision is required for CIFAR-10 data. Run scripts/setup.sh first."
        ) from exc
    return datasets, transforms


def _validate_positive_int(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _validate_train_val_size(train_size: int, val_size: int) -> None:
    if train_size + val_size > CIFAR10_TRAIN_SIZE:
        raise ValueError("train_size + val_size cannot exceed CIFAR-10 train size")


def _validate_probability(name: str, value: float, *, upper_inclusive: bool) -> None:
    if not isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    upper_ok = value <= 1.0 if upper_inclusive else value < 1.0
    if not upper_ok:
        limit = "<= 1.0" if upper_inclusive else "< 1.0"
        raise ValueError(f"{name} must be {limit}")


def _validate_nonnegative_finite(name: str, value: float) -> None:
    if not isfinite(float(value)) or value < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")


def _train_val_sizes(config: Cifar10DataConfig) -> tuple[int, int]:
    if config.quick_smoke:
        return config.smoke_train_size, config.smoke_val_size
    return config.train_size, config.val_size


def _download_enabled(config: Cifar10DataConfig) -> bool:
    return bool(config.download and not config.quick_smoke)


def _eval_transform(transforms):
    return transforms.ToTensor()


def _train_transform(config: Cifar10DataConfig, transforms):
    train_transforms = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if config.randaugment:
        train_transforms.append(transforms.RandAugment())
    train_transforms.append(transforms.ToTensor())
    return transforms.Compose(train_transforms)


def _split_train_val_indices(
    dataset_size: int,
    *,
    train_size: int,
    val_size: int,
    seed: int,
) -> tuple[list[int], list[int]]:
    if train_size + val_size > dataset_size:
        raise ValueError("requested train/val split exceeds available CIFAR-10 training examples")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()
    train_end = train_size
    val_end = train_end + val_size
    return indices[:train_end], indices[train_end:val_end]


def _validate_cifar_size(name: str, dataset: Dataset, expected_size: int) -> None:
    actual_size = len(dataset)
    if actual_size != expected_size:
        raise ValueError(f"{name} must contain {expected_size} examples, found {actual_size}")


def build_cifar10_datasets(config: Cifar10DataConfig) -> tuple[Dataset, Dataset]:
    config.validate()
    datasets, transforms = _import_torchvision()
    full_train = datasets.CIFAR10(
        root=str(config.data_dir),
        train=True,
        download=_download_enabled(config),
        transform=_train_transform(config, transforms),
    )
    full_eval = datasets.CIFAR10(
        root=str(config.data_dir),
        train=True,
        download=False,
        transform=_eval_transform(transforms),
    )
    _validate_cifar_size("CIFAR-10 train split source", full_train, CIFAR10_TRAIN_SIZE)
    _validate_cifar_size("CIFAR-10 eval split source", full_eval, CIFAR10_TRAIN_SIZE)

    train_size, val_size = _train_val_sizes(config)
    train_indices, val_indices = _split_train_val_indices(
        len(full_train),
        train_size=train_size,
        val_size=val_size,
        seed=config.seed,
    )
    return Subset(full_train, train_indices), Subset(full_eval, val_indices)


def build_cifar10_loaders(config: Cifar10DataConfig) -> tuple[DataLoader, DataLoader]:
    train_set, val_set = build_cifar10_datasets(config)
    loader_kwargs = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": True,
        "persistent_workers": config.num_workers > 0,
    }
    train_generator = torch.Generator().manual_seed(config.seed)
    train_loader = DataLoader(
        train_set,
        shuffle=True,
        drop_last=not config.quick_smoke,
        generator=train_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)
    return train_loader, val_loader


def build_cifar10_test_dataset(config: Cifar10DataConfig) -> Dataset:
    config.validate(allow_test=True)
    if not config.use_test:
        raise ValueError("set use_test=True only for validation-selected final CIFAR-10 test runs")

    datasets, transforms = _import_torchvision()
    test_set = datasets.CIFAR10(
        root=str(config.data_dir),
        train=False,
        download=_download_enabled(config),
        transform=_eval_transform(transforms),
    )
    _validate_cifar_size("CIFAR-10 test set", test_set, CIFAR10_TEST_SIZE)
    if config.quick_smoke:
        return Subset(test_set, list(range(config.smoke_test_size)))
    return test_set


def build_cifar10_test_loader(config: Cifar10DataConfig) -> DataLoader:
    test_set = build_cifar10_test_dataset(config)
    return DataLoader(
        test_set,
        batch_size=config.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=config.num_workers > 0,
    )
