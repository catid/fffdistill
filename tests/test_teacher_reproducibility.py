from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
from torch.utils.data import Dataset

import cifar_mamba_fff.train_teacher as train_teacher
from cifar_mamba_fff import data
from cifar_mamba_fff.data import Cifar10DataConfig
from cifar_mamba_fff.models.mamba3_cifar import Mamba3CifarConfig
from cifar_mamba_fff.train_teacher import TeacherRunConfig, TeacherTrainConfig


class _Transform:
    def __init__(self, *args: object, **kwargs: object) -> None:
        self.args = args
        self.kwargs = kwargs

    def __call__(self, sample: object) -> object:
        return sample


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
        train: bool,
        download: bool,
        transform: object,
    ) -> None:
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform
        self.size = data.CIFAR10_TRAIN_SIZE if train else data.CIFAR10_TEST_SIZE
        self.calls.append(self)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> tuple[int, int]:
        return index, index % 10


@pytest.fixture()
def fake_torchvision(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeCIFAR10.calls = []
    datasets = SimpleNamespace(CIFAR10=_FakeCIFAR10)
    transforms = SimpleNamespace(
        Compose=_Compose,
        RandomCrop=_Transform,
        RandomHorizontalFlip=_Transform,
        RandAugment=_Transform,
        ToTensor=_Transform,
    )
    monkeypatch.setattr(data, "_import_torchvision", lambda: (datasets, transforms))


def _split_indices(config: Cifar10DataConfig) -> tuple[list[int], list[int]]:
    train_set, val_set = data.build_cifar10_datasets(config)
    return list(train_set.indices), list(val_set.indices)


def test_cifar_split_indices_ignore_augmentation_seed_when_split_seed_is_fixed(
    fake_torchvision: None,
    tmp_path: Path,
) -> None:
    del fake_torchvision
    base = Cifar10DataConfig(
        data_dir=tmp_path,
        download=False,
        seed=11,
        split_seed=222,
        train_size=128,
        val_size=32,
    )
    changed_run_seed = Cifar10DataConfig(
        data_dir=tmp_path,
        download=False,
        seed=999,
        split_seed=222,
        train_size=128,
        val_size=32,
    )

    assert _split_indices(base) == _split_indices(changed_run_seed)


def test_cifar_split_indices_change_when_split_seed_changes(
    fake_torchvision: None,
    tmp_path: Path,
) -> None:
    del fake_torchvision
    base = Cifar10DataConfig(
        data_dir=tmp_path,
        download=False,
        seed=11,
        split_seed=222,
        train_size=128,
        val_size=32,
    )
    changed_split_seed = Cifar10DataConfig(
        data_dir=tmp_path,
        download=False,
        seed=11,
        split_seed=333,
        train_size=128,
        val_size=32,
    )

    train_a, val_a = _split_indices(base)
    train_b, val_b = _split_indices(changed_split_seed)

    assert train_a != train_b or val_a != val_b


def _minimal_run_config(
    tmp_path: Path,
    *,
    seed: int = 2026,
    epochs: int = 1,
) -> TeacherRunConfig:
    return TeacherRunConfig(
        seed=seed,
        dataset_name="cifar10",
        data=Cifar10DataConfig(
            data_dir=tmp_path / "data",
            batch_size=2,
            num_workers=0,
            quick_smoke=True,
            download=False,
            use_test=False,
        ),
        model=Mamba3CifarConfig(),
        train=TeacherTrainConfig(
            epochs=epochs,
            batch_size_per_gpu=2,
            num_workers=0,
            mixup=0.0,
            cutmix=0.0,
        ),
    )


def test_run_teacher_training_seeds_from_run_config_before_training_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seed_calls: list[int] = []
    run_config = _minimal_run_config(tmp_path, seed=4242)

    monkeypatch.setattr(train_teacher, "seed_everything", seed_calls.append)
    monkeypatch.setattr(train_teacher.torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA is required"):
        train_teacher.run_teacher_training(
            run_config,
            output_dir=tmp_path / "teacher",
            quick_smoke=True,
            save_checkpoint=False,
        )

    assert seed_calls == [run_config.seed]


class _FakeModel:
    def state_dict(self) -> dict[str, object]:
        return {"weights": "fake"}


def test_run_teacher_training_saves_checkpoint_only_for_strictly_best_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_config = _minimal_run_config(tmp_path, seed=2027, epochs=3)
    checkpoint_calls: list[tuple[Path, float, int]] = []
    val_accuracies = iter([0.20, 0.20, 0.30])

    monkeypatch.setattr(train_teacher.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(train_teacher, "build_cifar10_loaders", lambda config: ([object()], [object()]))
    monkeypatch.setattr(
        train_teacher,
        "build_teacher_model",
        lambda model_config, *, device=None, enforce_target_params=True: (_FakeModel(), 10_000_000),
    )
    monkeypatch.setattr(
        train_teacher,
        "build_muon_adamw_optimizer",
        lambda model, train_config, *, assignment_log_path=None: (
            object(),
            {
                "muon_tensors": 1,
                "adamw_tensors": 1,
                "muon_parameters": 9_000_000,
                "adamw_parameters": 1_000_000,
            },
        ),
    )
    monkeypatch.setattr(
        train_teacher,
        "build_lr_scheduler",
        lambda optimizer, train_config, *, steps_per_epoch: object(),
    )
    monkeypatch.setattr(
        train_teacher,
        "_train_one_step",
        lambda model, batch, optimizer, scheduler, run_config, device: {
            "train_loss": 2.3,
            "train_accuracy_hard_labels": 0.125,
            "lr_muon": 0.02,
            "lr_adamw": 0.001,
        },
    )

    def fake_evaluate_steps(
        model: object,
        loader: Iterable[object],
        run_config: TeacherRunConfig,
        device: object,
        *,
        max_steps: int | None,
    ) -> dict[str, float]:
        del model, loader, run_config, device, max_steps
        return {"val_loss": 2.2, "val_accuracy": next(val_accuracies), "val_steps": 1.0}

    def fake_save_checkpoint(checkpoint_path: Path, payload: dict[str, object]) -> None:
        metrics = payload["metrics"]
        assert isinstance(metrics, dict)
        checkpoint_calls.append((checkpoint_path, float(metrics["val_accuracy"]), int(metrics["epoch"])))

    monkeypatch.setattr(train_teacher, "_evaluate_steps", fake_evaluate_steps)
    monkeypatch.setattr(train_teacher, "save_teacher_checkpoint_atomic", fake_save_checkpoint)

    summary = train_teacher.run_teacher_training(
        run_config,
        output_dir=tmp_path / "teacher",
        quick_smoke=False,
        save_checkpoint=True,
    )

    assert summary["best_val_accuracy"] == pytest.approx(0.30)
    assert checkpoint_calls == [
        (tmp_path / "teacher" / "teacher_best.pt", 0.20, 0),
        (tmp_path / "teacher" / "teacher_best.pt", 0.30, 2),
    ]


def test_save_teacher_checkpoint_atomic_writes_temp_then_replaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    save_calls: list[tuple[object, Path]] = []
    replace_calls: list[tuple[Path, Path]] = []
    checkpoint_path = tmp_path / "nested" / "teacher_best.pt"
    payload = {"metrics": {"val_accuracy": 0.5}}

    def fake_torch_save(saved_payload: object, path: Path) -> None:
        save_calls.append((saved_payload, path))
        path.write_bytes(b"checkpoint")

    def fake_replace(src: Path, dst: Path) -> None:
        replace_calls.append((src, dst))
        src.rename(dst)

    monkeypatch.setattr(train_teacher.torch, "save", fake_torch_save)
    monkeypatch.setattr(train_teacher.os, "replace", fake_replace)

    train_teacher.save_teacher_checkpoint_atomic(checkpoint_path, payload)

    tmp_path_used = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    assert save_calls == [(payload, tmp_path_used)]
    assert replace_calls == [(tmp_path_used, checkpoint_path)]
    assert checkpoint_path.read_bytes() == b"checkpoint"
    assert not tmp_path_used.exists()
