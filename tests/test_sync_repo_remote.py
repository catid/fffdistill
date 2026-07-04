from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sync_repo_remote.sh"


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init"], cwd=path, check=True, capture_output=True, text=True)
    (path / "tracked.txt").write_text("base\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-m",
            "base",
        ],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_sync_repo_remote_work_target_allows_clean_repo(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    completed = subprocess.run(
        ["bash", str(SCRIPT), "work"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Local worktree is already at" in completed.stdout


def test_sync_repo_remote_blocks_staged_changes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "tracked.txt").write_text("changed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True, capture_output=True, text=True)

    completed = subprocess.run(
        ["bash", str(SCRIPT), "work"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Refusing to sync uncommitted local worktree changes" in completed.stderr


def test_sync_repo_remote_blocks_untracked_changes(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    (tmp_path / "untracked.txt").write_text("new\n", encoding="utf-8")

    completed = subprocess.run(
        ["bash", str(SCRIPT), "work"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "Refusing to sync uncommitted local worktree changes" in completed.stderr
