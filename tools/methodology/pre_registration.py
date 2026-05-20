"""Mechanism pre-registration check.

Walk-forward refuses to run unless the setup's `mechanism_tags` field
was committed to git in a commit that is older than (not equal to) HEAD.
This prevents post-hoc rationalization — researcher can't write the
mechanism docs AFTER seeing which windows fail.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Optional


class PreRegistrationError(RuntimeError):
    """Raised when mechanism docs are not pre-registered to git."""


def _git_commit_for_file_string(
    repo_root: Path, file_path: Path, search_key: str,
) -> Optional[str]:
    """Return the OLDEST commit SHA that introduced `search_key` in `file_path`.
    Uses git log -S (pickaxe), which finds adds/removes of the string.
    """
    rel = file_path.relative_to(repo_root)
    result = subprocess.run(
        ["git", "log", "-S", search_key, "--format=%H", "--", str(rel)],
        cwd=str(repo_root), capture_output=True, text=True, check=True,
    )
    lines = [ln.strip() for ln in result.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    # `git log` orders newest-first; last line is the oldest commit where the
    # string first appeared.
    return lines[-1]


def _git_head_sha(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(repo_root), capture_output=True, text=True, check=True,
    )
    return result.stdout.strip()


def _git_commit_count_between(repo_root: Path, older_sha: str, newer_sha: str) -> int:
    """Number of commits in (older_sha, newer_sha] range (exclusive of older)."""
    result = subprocess.run(
        ["git", "rev-list", "--count", f"{older_sha}..{newer_sha}"],
        cwd=str(repo_root), capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def check_mechanism_pre_registered(
    repo_root: Path, config_path: Path, setup_name: str,
) -> None:
    """Verify the setup's `mechanism_tags` field was committed at least one
    commit before HEAD.

    Raises:
        PreRegistrationError: if mechanism_tags is missing/empty OR the
            commit that introduced it is HEAD (post-hoc).
    """
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    block = cfg.get("setups", {}).get(setup_name, {})
    tags = block.get("mechanism_tags")
    if not tags:
        raise PreRegistrationError(
            f"Setup '{setup_name}' has no mechanism_tags. "
            f"Pre-register by committing mechanism_tags + mechanism_notes to "
            f"{config_path} BEFORE running walk-forward."
        )

    try:
        # Search for the mechanism_tags key itself — invariant across tag rotations
        search_key = '"mechanism_tags"'
        intro_sha = _git_commit_for_file_string(repo_root, config_path, search_key)
        if intro_sha is None:
            raise PreRegistrationError(
                f"Could not find any commit introducing mechanism_tags for "
                f"'{setup_name}' in git log for {config_path}. Pre-register first."
            )

        head = _git_head_sha(repo_root)
        commits_between = _git_commit_count_between(repo_root, intro_sha, head)
    except subprocess.CalledProcessError as e:
        raise PreRegistrationError(
            f"git command failed while checking pre-registration: {e.cmd} "
            f"(exit code {e.returncode}). Ensure {repo_root} is a git repo "
            f"with commits, and git is on PATH."
        ) from e

    if commits_between < 1:
        raise PreRegistrationError(
            f"Setup '{setup_name}' mechanism_tags was added in HEAD commit "
            f"({intro_sha[:8]}). This is post-hoc rationalization. Make another "
            f"unrelated commit (or wait for one) before running walk-forward."
        )
