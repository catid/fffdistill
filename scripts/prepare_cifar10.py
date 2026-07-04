#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from torchvision.datasets import CIFAR10

TRAIN_MEMBER_NAMES = {f"cifar-10-batches-py/{name}" for name, _ in CIFAR10.train_list}
TRAIN_MEMBER_NAMES.add(f"cifar-10-batches-py/{CIFAR10.meta['filename']}")


@dataclass(frozen=True)
class PrepareSummary:
    data_dir: str
    archive_path: str
    extracted_dir: str
    official_url: str
    train_ready: bool
    archive_ready: bool
    downloaded: bool
    extracted: bool
    test_accessed: bool
    checked_train_files: int
    missing_train_files: list[str]
    md5_failures: list[str]
    elapsed_seconds: float


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _archive_path(data_dir: Path) -> Path:
    return data_dir / CIFAR10.filename


def _extracted_dir(data_dir: Path) -> Path:
    return data_dir / "cifar-10-batches-py"


def _expected_train_files(data_dir: Path) -> dict[Path, str]:
    extracted = _extracted_dir(data_dir)
    expected = {extracted / name: md5 for name, md5 in CIFAR10.train_list}
    expected[extracted / str(CIFAR10.meta["filename"])] = str(CIFAR10.meta["md5"])
    return expected


def check_archive(data_dir: Path) -> bool:
    archive = _archive_path(data_dir)
    return archive.exists() and _md5(archive) == CIFAR10.tgz_md5


def check_train_files(data_dir: Path) -> tuple[bool, list[str], list[str], int]:
    missing: list[str] = []
    md5_failures: list[str] = []
    checked = 0
    for path, expected_md5 in _expected_train_files(data_dir).items():
        relative = str(path.relative_to(data_dir))
        if not path.exists():
            missing.append(relative)
            continue
        checked += 1
        if _md5(path) != expected_md5:
            md5_failures.append(relative)
    return not missing and not md5_failures, missing, md5_failures, checked


def _download_with_curl(url: str, output_path: Path, timeout_s: int) -> None:
    part_path = output_path.with_name(output_path.name + ".part")
    command = [
        "curl",
        "-L",
        "--fail",
        "--retry",
        "5",
        "--connect-timeout",
        "30",
        "--speed-time",
        "60",
        "--speed-limit",
        "1024",
        "-C",
        "-",
        "-o",
        str(part_path),
        url,
    ]
    subprocess.run(command, check=True, timeout=timeout_s)
    if _md5(part_path) != CIFAR10.tgz_md5:
        raise RuntimeError(f"downloaded archive failed MD5 verification: {part_path}")
    part_path.replace(output_path)


def _download_with_urllib(url: str, output_path: Path, timeout_s: int) -> None:
    part_path = output_path.with_name(output_path.name + ".part")
    start = time.monotonic()
    with (
        urllib.request.urlopen(url, timeout=min(timeout_s, 60)) as response,
        part_path.open("wb") as handle,
    ):
        while True:
            if time.monotonic() - start > timeout_s:
                raise TimeoutError(f"download exceeded {timeout_s}s: {url}")
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    if _md5(part_path) != CIFAR10.tgz_md5:
        raise RuntimeError(f"downloaded archive failed MD5 verification: {part_path}")
    part_path.replace(output_path)


def download_archive(data_dir: Path, *, timeout_s: int, prefer_curl: bool = True) -> bool:
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = _archive_path(data_dir)
    if check_archive(data_dir):
        return False
    if prefer_curl and shutil.which("curl"):
        _download_with_curl(CIFAR10.url, archive, timeout_s)
    else:
        _download_with_urllib(CIFAR10.url, archive, timeout_s)
    return True


def extract_train_files(data_dir: Path) -> bool:
    archive = _archive_path(data_dir)
    if not check_archive(data_dir):
        raise RuntimeError(f"CIFAR-10 archive missing or invalid: {archive}")
    train_ready, _, _, _ = check_train_files(data_dir)
    if train_ready:
        return False

    data_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:gz") as tar:
        members = []
        for member in tar.getmembers():
            if member.name not in TRAIN_MEMBER_NAMES:
                continue
            target = (data_dir / member.name).resolve()
            root = data_dir.resolve()
            if root not in target.parents:
                raise RuntimeError(f"unsafe tar member path: {member.name}")
            members.append(member)
        found = {member.name for member in members}
        missing_members = sorted(TRAIN_MEMBER_NAMES - found)
        if missing_members:
            raise RuntimeError("archive missing train members: " + ", ".join(missing_members))
        tar.extractall(data_dir, members=members, filter="data")
    train_ready, missing, md5_failures, _ = check_train_files(data_dir)
    if not train_ready:
        raise RuntimeError(
            "extracted train files are incomplete: "
            f"missing={missing}, md5_failures={md5_failures}"
        )
    return True


def prepare_cifar10_train(
    *,
    data_dir: Path,
    output_json: Path,
    download: bool,
    extract: bool,
    timeout_s: int,
    prefer_curl: bool,
) -> PrepareSummary:
    start = time.perf_counter()
    data_dir.mkdir(parents=True, exist_ok=True)

    initial_train_ready, initial_missing, initial_md5_failures, initial_checked = check_train_files(data_dir)
    downloaded = False
    if download and not initial_train_ready and not check_archive(data_dir):
        downloaded = download_archive(data_dir, timeout_s=timeout_s, prefer_curl=prefer_curl)
    extracted = False
    if extract and not initial_train_ready:
        extracted = extract_train_files(data_dir)

    if initial_train_ready:
        train_ready = initial_train_ready
        missing = initial_missing
        md5_failures = initial_md5_failures
        checked = initial_checked
    else:
        train_ready, missing, md5_failures, checked = check_train_files(data_dir)
    summary = PrepareSummary(
        data_dir=str(data_dir),
        archive_path=str(_archive_path(data_dir)),
        extracted_dir=str(_extracted_dir(data_dir)),
        official_url=CIFAR10.url,
        train_ready=train_ready,
        archive_ready=check_archive(data_dir),
        downloaded=downloaded,
        extracted=extracted,
        test_accessed=False,
        checked_train_files=checked,
        missing_train_files=missing,
        md5_failures=md5_failures,
        elapsed_seconds=time.perf_counter() - start,
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare CIFAR-10 train files without test access")
    parser.add_argument("--data-dir", default="data/cifar10")
    parser.add_argument("--output-json", default="outputs/cifar10_prepare.json")
    parser.add_argument("--download", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--extract", type=lambda value: value.lower() == "true", default=True)
    parser.add_argument("--timeout-s", type=int, default=1800)
    parser.add_argument("--prefer-curl", type=lambda value: value.lower() == "true", default=True)
    args = parser.parse_args()
    if args.timeout_s <= 0:
        raise ValueError("--timeout-s must be positive")
    summary = prepare_cifar10_train(
        data_dir=Path(args.data_dir),
        output_json=Path(args.output_json),
        download=bool(args.download),
        extract=bool(args.extract),
        timeout_s=int(args.timeout_s),
        prefer_curl=bool(args.prefer_curl),
    )
    print(json.dumps(asdict(summary), indent=2, sort_keys=True))
    if not summary.train_ready:
        raise SystemExit("CIFAR-10 train files are not ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
