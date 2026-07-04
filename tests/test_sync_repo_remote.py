from __future__ import annotations

import os
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


def test_sync_repo_remote_rsync_target_invokes_fake_remote_verification(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    log_path = tmp_path.parent / f"{tmp_path.name}_tool.log"
    bin_dir = tmp_path.parent / f"{tmp_path.name}_bin"
    bin_dir.mkdir()
    fake_tool = "\n".join(
        [
            "#!/usr/bin/env bash",
            "{",
            "  printf 'cmd=%s\\n' \"$(basename \"$0\")\"",
            "  for arg in \"$@\"; do",
            "    echo \"arg=$arg\"",
            "  done",
            "} >> \"$SYNC_LOG\"",
            "exit 0",
            "",
        ]
    )
    for name in ("ssh", "rsync"):
        tool_path = bin_dir / name
        tool_path.write_text(fake_tool, encoding="utf-8")
        tool_path.chmod(0o755)

    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SYNC_LOG": str(log_path),
    }
    completed = subprocess.run(
        ["bash", str(SCRIPT), "ripper", "/tmp/remote-fff"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == 0
    assert "syncing ripper:/tmp/remote-fff" in completed.stdout
    log_text = log_path.read_text(encoding="utf-8")
    assert "cmd=rsync" in log_text
    assert "arg=--delete" in log_text
    assert "arg=--exclude\narg=outputs/" in log_text
    assert f"arg={tmp_path}/" in log_text
    assert "arg=ripper:/tmp/remote-fff/" in log_text
    assert "cmd=ssh" in log_text
    assert "arg=ripper" in log_text
    assert "cd '/tmp/remote-fff'" in log_text
    assert "git rev-parse HEAD" in log_text
