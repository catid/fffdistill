from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import socket
import sys
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "prepare_cifar10.py"
CIFAR_DIRNAME = "cifar-10-batches-py"
TRAIN_BATCH_BYTES = {
    f"data_batch_{index}": f"offline fake CIFAR train batch {index}\n".encode()
    for index in range(1, 6)
}
META_FILENAME = "batches.meta"
META_BYTES = b"offline fake CIFAR metadata\n"
ARCHIVE_BYTES = b"offline fake CIFAR archive\n"


def _load_prepare_module() -> ModuleType:
    if not SCRIPT_PATH.exists():
        pytest.fail(
            "Expected scripts/prepare_cifar10.py to exist with helper API: "
            "check_train_ready(cifar_dir), prepare_cifar10(data_dir, *, summary_path=None), "
            "write_summary(summary_path, summary), and a monkeypatchable download_and_extract helper."
        )

    spec = importlib.util.spec_from_file_location("prepare_cifar10_under_test", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        pytest.fail(f"Could not import {SCRIPT_PATH} as a Python module.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def prepare_module() -> ModuleType:
    return _load_prepare_module()


def _require_helper(module: ModuleType, name: str) -> Any:
    helper = getattr(module, name, None)
    if not callable(helper):
        pytest.fail(f"Expected {SCRIPT_PATH} to define callable helper {name}().")
    return helper


def _summary_mapping(result: object) -> Mapping[str, Any]:
    if isinstance(result, Mapping):
        return result
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    if hasattr(result, "__dict__"):
        return vars(result)
    pytest.fail(
        "Expected CIFAR preparation helpers to return a mapping, dataclass, "
        "or simple object exposing train_ready/test_accessed fields."
    )


def _md5_bytes(contents: bytes) -> str:
    return hashlib.md5(contents, usedforsecurity=False).hexdigest()


class _FakeCIFAR10:
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = _md5_bytes(ARCHIVE_BYTES)
    url = "https://example.invalid/cifar-10-python.tar.gz"
    train_list: ClassVar[list[tuple[str, str]]] = [
        (name, _md5_bytes(contents)) for name, contents in TRAIN_BATCH_BYTES.items()
    ]
    meta: ClassVar[dict[str, str]] = {"filename": META_FILENAME, "md5": _md5_bytes(META_BYTES)}


def _patch_cifar_metadata(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    if hasattr(module, "CIFAR10"):
        monkeypatch.setattr(module, "CIFAR10", _FakeCIFAR10)

    train_member_names = {f"{CIFAR_DIRNAME}/{name}" for name in TRAIN_BATCH_BYTES}
    train_member_names.add(f"{CIFAR_DIRNAME}/{META_FILENAME}")
    if hasattr(module, "TRAIN_MEMBER_NAMES"):
        monkeypatch.setattr(module, "TRAIN_MEMBER_NAMES", train_member_names)

    checksums = _fake_train_md5()
    for attr in (
        "CIFAR10_TRAIN_BATCH_MD5",
        "TRAIN_BATCH_MD5",
        "TRAIN_BATCH_CHECKSUMS",
        "EXPECTED_TRAIN_MD5",
    ):
        if hasattr(module, attr):
            monkeypatch.setattr(module, attr, checksums)


def _write_fake_train_tree(data_dir: Path, *, include_test_batch: bool = True) -> Path:
    cifar_dir = data_dir / CIFAR_DIRNAME
    cifar_dir.mkdir(parents=True)
    for name, contents in TRAIN_BATCH_BYTES.items():
        (cifar_dir / name).write_bytes(contents)
    (cifar_dir / META_FILENAME).write_bytes(META_BYTES)
    if include_test_batch:
        (cifar_dir / "test_batch").write_bytes(b"this file must not be inspected\n")
    return cifar_dir


def _fake_train_md5() -> dict[str, str]:
    return {
        name: hashlib.md5(contents, usedforsecurity=False).hexdigest()
        for name, contents in TRAIN_BATCH_BYTES.items()
    }


def _block_test_batch_access(monkeypatch: pytest.MonkeyPatch) -> None:
    original_open = Path.open
    original_stat = Path.stat

    def guarded_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "test_batch":
            pytest.fail("check_train_ready() must not open or read CIFAR-10 test_batch.")
        return original_open(self, *args, **kwargs)

    def guarded_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "test_batch":
            pytest.fail("check_train_ready() must not stat or inspect CIFAR-10 test_batch.")
        return original_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    monkeypatch.setattr(Path, "stat", guarded_stat)


def _block_network(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_network(*_args: object, **_kwargs: object) -> None:
        pytest.fail(
            "prepare_cifar10 offline tests must not use network APIs directly; "
            "call the monkeypatchable download/extract helper instead."
        )

    monkeypatch.setattr(socket, "create_connection", fail_network)
    monkeypatch.setattr(urllib.request, "urlopen", fail_network)
    monkeypatch.setattr(urllib.request, "urlretrieve", fail_network)


def _patch_download_extract(
    module: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    data_dir: Path,
) -> dict[str, list[Path]]:
    calls: dict[str, list[Path]] = {"download": [], "extract": [], "download_and_extract": []}

    def fake_download_archive(target_dir: str | Path, *_args: object, **_kwargs: object) -> bool:
        calls["download"].append(Path(target_dir))
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        (Path(target_dir) / _FakeCIFAR10.filename).write_bytes(ARCHIVE_BYTES)
        return True

    def fake_extract_train_files(target_dir: str | Path, *_args: object, **_kwargs: object) -> bool:
        calls["extract"].append(Path(target_dir))
        _write_fake_train_tree(data_dir, include_test_batch=False)
        return True

    def fake_download_and_extract(target_dir: str | Path, *_args: object, **_kwargs: object) -> Path:
        calls["download_and_extract"].append(Path(target_dir))
        return _write_fake_train_tree(data_dir)

    if hasattr(module, "download_archive") and hasattr(module, "extract_train_files"):
        monkeypatch.setattr(module, "download_archive", fake_download_archive)
        monkeypatch.setattr(module, "extract_train_files", fake_extract_train_files)
        return calls

    for attr in ("download_and_extract_cifar10", "download_and_extract"):
        if hasattr(module, attr):
            monkeypatch.setattr(module, attr, fake_download_and_extract)
            return calls

    pytest.fail(
        "Expected prepare_cifar10.py to expose a monkeypatchable download/extract helper "
        "named download_and_extract_cifar10() or download_and_extract()."
    )


def _patch_download_extract_to_fail(module: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_download_extract(*_args: object, **_kwargs: object) -> None:
        pytest.fail("prepare_cifar10 should reuse ready train files without downloading/extracting.")

    patched = False
    for attr in (
        "download_archive",
        "extract_train_files",
        "download_and_extract_cifar10",
        "download_and_extract",
    ):
        if hasattr(module, attr):
            monkeypatch.setattr(module, attr, fail_download_extract)
            patched = True

    if not patched:
        pytest.fail(
            "Expected a monkeypatchable download/extract helper so existing train-file readiness "
            "can be tested without network or archive extraction."
        )


def _call_train_ready_check(module: ModuleType, data_dir: Path) -> Mapping[str, Any]:
    if hasattr(module, "check_train_ready"):
        result = _require_helper(module, "check_train_ready")(data_dir)
        return _summary_mapping(result)

    if hasattr(module, "check_train_files"):
        ready, missing, md5_failures, checked = _require_helper(module, "check_train_files")(data_dir)
        return {
            "train_ready": ready,
            "missing_train_files": missing,
            "md5_failures": md5_failures,
            "checked_train_files": checked,
            "test_accessed": False,
        }

    pytest.fail(
        "Expected prepare_cifar10.py to define check_train_ready(data_dir) or "
        "check_train_files(data_dir) for offline train-batch readiness checks."
    )


def _call_prepare(module: ModuleType, data_dir: Path, summary_path: Path) -> Mapping[str, Any]:
    if hasattr(module, "prepare_cifar10"):
        try:
            result = _require_helper(module, "prepare_cifar10")(data_dir, summary_path=summary_path)
        except TypeError as exc:
            pytest.fail(
                "Expected prepare_cifar10(data_dir: Path, *, summary_path: Path | None = None) "
                f"to be callable by offline tests; got TypeError: {exc}"
            )
        return _summary_mapping(result)

    if hasattr(module, "prepare_cifar10_train"):
        try:
            result = _require_helper(module, "prepare_cifar10_train")(
                data_dir=data_dir,
                output_json=summary_path,
                download=True,
                extract=True,
                timeout_s=1,
                prefer_curl=False,
            )
        except TypeError as exc:
            pytest.fail(
                "Expected prepare_cifar10_train(*, data_dir, output_json, download, extract, "
                f"timeout_s, prefer_curl) to be callable by offline tests; got TypeError: {exc}"
            )
        return _summary_mapping(result)

    pytest.fail(
        "Expected prepare_cifar10.py to define prepare_cifar10() or prepare_cifar10_train() "
        "for offline preparation tests."
    )


def _call_prepare_with_ready_train_files(
    module: ModuleType,
    data_dir: Path,
    summary_path: Path,
) -> Mapping[str, Any]:
    if hasattr(module, "prepare_cifar10"):
        return _call_prepare(module, data_dir, summary_path)

    if hasattr(module, "prepare_cifar10_train"):
        result = _require_helper(module, "prepare_cifar10_train")(
            data_dir=data_dir,
            output_json=summary_path,
            download=True,
            extract=True,
            timeout_s=1,
            prefer_curl=False,
        )
        return _summary_mapping(result)

    pytest.fail(
        "Expected prepare_cifar10.py to define prepare_cifar10() or prepare_cifar10_train() "
        "for existing train-file tests."
    )


def _read_summary_file(summary_path: Path) -> Mapping[str, Any]:
    assert summary_path.exists(), "prepare_cifar10() should write the requested summary_path."
    loaded = json.loads(summary_path.read_text())
    assert isinstance(loaded, Mapping), "summary_path should contain one JSON object."
    return loaded


def test_check_train_ready_ignores_test_batch(
    prepare_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "cifar10"
    _write_fake_train_tree(data_dir)
    _patch_cifar_metadata(prepare_module, monkeypatch)
    _block_test_batch_access(monkeypatch)

    summary = _call_train_ready_check(prepare_module, data_dir)

    assert summary.get("train_ready") is True
    assert summary.get("test_accessed", False) is False


def test_prepare_cifar10_download_extract_is_monkeypatchable_and_writes_summary(
    prepare_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "cifar10"
    summary_path = tmp_path / "prepare_summary.json"
    _patch_cifar_metadata(prepare_module, monkeypatch)
    download_calls = _patch_download_extract(prepare_module, monkeypatch, data_dir)
    _block_test_batch_access(monkeypatch)
    _block_network(monkeypatch)

    result = _call_prepare(prepare_module, data_dir, summary_path)
    written = _read_summary_file(summary_path)

    assert download_calls["download"] == [data_dir] or download_calls["download_and_extract"] == [
        data_dir
    ]
    if download_calls["download"]:
        assert download_calls["extract"] == [data_dir]
    assert result.get("train_ready") is True
    assert result.get("test_accessed", False) is False
    assert written.get("train_ready") is True
    assert written.get("test_accessed") is False


def test_prepare_cifar10_reuses_existing_train_files(
    prepare_module: ModuleType,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_dir = tmp_path / "cifar10"
    summary_path = tmp_path / "prepare_summary.json"
    _write_fake_train_tree(data_dir)
    _patch_cifar_metadata(prepare_module, monkeypatch)
    _patch_download_extract_to_fail(prepare_module, monkeypatch)
    _block_test_batch_access(monkeypatch)
    _block_network(monkeypatch)

    result = _call_prepare_with_ready_train_files(prepare_module, data_dir, summary_path)
    written = _read_summary_file(summary_path)

    assert result.get("train_ready") is True
    assert result.get("test_accessed", False) is False
    assert written.get("train_ready") is True
    assert written.get("test_accessed") is False
