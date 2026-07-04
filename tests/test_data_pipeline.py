from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
from torch.utils.data import Dataset

from cifar_mamba_fff import data


class _Transform:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, sample: object) -> object:
        return sample


class _RandomCrop(_Transform):
    pass


class _RandomHorizontalFlip(_Transform):
    pass


class _RandAugment(_Transform):
    pass


class _ToTensor(_Transform):
    pass


class _Compose:
    def __init__(self, transforms: list[object]) -> None:
        self.transforms = transforms

    def __call__(self, sample: object) -> object:
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class _FakeCIFAR10(Dataset):
    calls: ClassVar[list[_FakeCIFAR10]] = []

    def __init__(
        self,
        *,
        root: str,
        train: bool = True,
        download: bool = False,
        transform: object | None = None,
        target_transform: object | None = None,
    ) -> None:
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.target_transform = target_transform
        self.size = data.CIFAR10_TRAIN_SIZE if train else data.CIFAR10_TEST_SIZE
        self.calls.append(self)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, index % 10

    @classmethod
    def train_files_ready(cls, root: str) -> bool:
        del root
        return True


@pytest.fixture()
def fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> type[_FakeCIFAR10]:
    _FakeCIFAR10.calls = []
    datasets = SimpleNamespace(CIFAR10=_FakeCIFAR10)
    transforms = SimpleNamespace(
        Compose=_Compose,
        RandomCrop=_RandomCrop,
        RandomHorizontalFlip=_RandomHorizontalFlip,
        RandAugment=_RandAugment,
        ToTensor=_ToTensor,
    )
    monkeypatch.setattr(data, "_import_torchvision", lambda: (datasets, transforms))
    monkeypatch.setattr(data, "Cifar10TrainOnly", _FakeCIFAR10)
    return _FakeCIFAR10


def _transform_names(compose: _Compose) -> list[str]:
    return [type(transform).__name__.removeprefix("_") for transform in compose.transforms]


def test_default_split_is_45k_5k_and_deterministic(fake_torchvision, tmp_path) -> None:
    config = data.Cifar10DataConfig(data_dir=tmp_path, download=False, seed=2027)

    train_a, val_a = data.build_cifar10_datasets(config)
    train_b, val_b = data.build_cifar10_datasets(config)

    assert len(train_a) == data.CIFAR10_DISTILL_TRAIN_SIZE
    assert len(val_a) == data.CIFAR10_VAL_SIZE
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices

    train_indices = set(train_a.indices)
    val_indices = set(val_a.indices)
    assert train_indices.isdisjoint(val_indices)
    assert len(train_indices | val_indices) == data.CIFAR10_TRAIN_SIZE
    assert all(call.download is False for call in fake_torchvision.calls)


def test_split_uses_split_seed_not_loader_seed(fake_torchvision, tmp_path) -> None:
    train_a, val_a = data.build_cifar10_datasets(
        data.Cifar10DataConfig(data_dir=tmp_path, download=False, seed=1)
    )
    train_b, val_b = data.build_cifar10_datasets(
        data.Cifar10DataConfig(data_dir=tmp_path, download=False, seed=2)
    )
    train_c, val_c = data.build_cifar10_datasets(
        data.Cifar10DataConfig(data_dir=tmp_path, download=False, seed=1, split_seed=2)
    )

    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    assert train_a.indices != train_c.indices
    assert val_a.indices != val_c.indices


def test_train_transform_uses_crop_flip_and_optional_randaugment(
    fake_torchvision,
    tmp_path,
) -> None:
    config = data.Cifar10DataConfig(data_dir=tmp_path, download=False, randaugment=True)

    train_set, val_set = data.build_cifar10_datasets(config)

    assert _transform_names(train_set.dataset.transform) == [
        "RandomCrop",
        "RandomHorizontalFlip",
        "RandAugment",
        "ToTensor",
    ]
    assert isinstance(val_set.dataset.transform, _ToTensor)


def test_quick_smoke_uses_small_splits_and_disables_download(
    fake_torchvision,
    tmp_path,
) -> None:
    config = data.Cifar10DataConfig(
        data_dir=tmp_path,
        download=True,
        quick_smoke=True,
        smoke_train_size=32,
        smoke_val_size=8,
        batch_size=64,
        num_workers=0,
    )

    train_set, val_set = data.build_cifar10_datasets(config)
    train_loader, _ = data.build_cifar10_loaders(config)

    assert len(train_set) == 32
    assert len(val_set) == 8
    assert train_loader.drop_last is False
    assert all(call.download is False for call in fake_torchvision.calls)


def test_missing_train_files_raise_prepare_error(
    fake_torchvision,
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _MissingTrainFiles(_FakeCIFAR10):
        @classmethod
        def train_files_ready(cls, root: str) -> bool:
            del root
            return False

    monkeypatch.setattr(data, "Cifar10TrainOnly", _MissingTrainFiles)

    with pytest.raises(RuntimeError, match=r"prepare_cifar10\.py"):
        data.build_cifar10_datasets(data.Cifar10DataConfig(data_dir=tmp_path, download=False))


def test_train_only_reader_uses_train_files_without_test_batch(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path
    batch_dir = root / data.Cifar10TrainOnly.base_folder
    batch_dir.mkdir(parents=True)
    batch_file = batch_dir / "data_batch_1"
    meta_file = batch_dir / "batches.meta"
    with batch_file.open("wb") as handle:
        pickle.dump(
            {
                "data": np.arange(2 * 3 * 32 * 32, dtype=np.uint8).reshape(2, -1),
                "labels": [3, 7],
            },
            handle,
        )
    with meta_file.open("wb") as handle:
        pickle.dump({"label_names": [str(index) for index in range(10)]}, handle)

    checked_paths: list[str] = []

    def fake_check_integrity(path: str, md5: str | None = None) -> bool:
        del md5
        assert "test_batch" not in path
        checked_paths.append(path)
        return Path(path).exists()

    monkeypatch.setattr(data.Cifar10TrainOnly, "train_list", [("data_batch_1", None)])
    monkeypatch.setattr(
        data.Cifar10TrainOnly,
        "meta",
        {"filename": "batches.meta", "key": "label_names", "md5": None},
    )
    monkeypatch.setattr(data, "check_integrity", fake_check_integrity)

    assert data.Cifar10TrainOnly.train_files_ready(root) is True
    dataset = data.Cifar10TrainOnly(root=root)

    image, target = dataset[1]

    assert len(dataset) == 2
    assert target == 7
    assert image.size == (32, 32)
    assert checked_paths == [str(batch_file), str(meta_file)]


def test_test_dataset_requires_explicit_use_test_and_stays_unaugmented(
    fake_torchvision,
    tmp_path,
) -> None:
    with pytest.raises(ValueError, match="use_test=True"):
        data.build_cifar10_test_dataset(
            data.Cifar10DataConfig(data_dir=tmp_path, download=False, use_test=False)
        )

    config = data.Cifar10DataConfig(
        data_dir=tmp_path,
        download=False,
        use_test=True,
        batch_size=128,
        num_workers=0,
    )

    test_set = data.build_cifar10_test_dataset(config)
    test_loader = data.build_cifar10_test_loader(config)

    assert len(test_set) == data.CIFAR10_TEST_SIZE
    assert fake_torchvision.calls[-2].train is False
    assert isinstance(test_set.transform, _ToTensor)
    assert test_loader.batch_size == 128
    assert test_loader.drop_last is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"label_smoothing": 1.0},
        {"mixup": -0.1},
        {"cutmix": float("inf")},
        {"smoke_test_size": data.CIFAR10_TEST_SIZE + 1},
        {"seed": -1},
        {"split_seed": -1},
    ],
)
def test_config_validates_batch_augmentation_hooks(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        data.Cifar10DataConfig(**kwargs).validate()

    data.Cifar10DataConfig(label_smoothing=0.1, mixup=0.2, cutmix=1.0).validate()
