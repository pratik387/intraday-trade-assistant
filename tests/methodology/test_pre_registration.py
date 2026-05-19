"""Tests for tools.methodology.pre_registration."""
import subprocess
from pathlib import Path

import pytest

from tools.methodology.pre_registration import (
    check_mechanism_pre_registered,
    PreRegistrationError,
)


def _init_git_repo(tmp_path: Path) -> None:
    subprocess.run(["git", "init"], cwd=str(tmp_path), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=str(tmp_path), check=True)


def test_pre_registration_passes_when_mechanism_tags_committed_before_head(tmp_path):
    """A setup block with mechanism_tags committed at least one commit
    before HEAD passes the pre-registration check."""
    _init_git_repo(tmp_path)

    cfg = tmp_path / "config.json"
    cfg.write_text(
        '{"setups": {"my_setup": {"mechanism_tags": ["FII_net_flow_positive_30d"], "mechanism_notes": "..."}}}',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "config.json"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-m", "pre-register mechanism"],
        cwd=str(tmp_path), check=True, capture_output=True,
    )

    # Advance HEAD with an unrelated commit
    (tmp_path / "other.txt").write_text("noise", encoding="utf-8")
    subprocess.run(["git", "add", "other.txt"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-m", "noise"],
        cwd=str(tmp_path), check=True, capture_output=True,
    )

    # Should NOT raise
    check_mechanism_pre_registered(repo_root=tmp_path, config_path=cfg, setup_name="my_setup")


def test_pre_registration_fails_when_mechanism_tags_missing(tmp_path):
    """Setup with no mechanism_tags raises with clear error."""
    _init_git_repo(tmp_path)

    cfg = tmp_path / "config.json"
    cfg.write_text('{"setups": {"my_setup": {"enabled": true}}}', encoding="utf-8")
    subprocess.run(["git", "add", "config.json"], cwd=str(tmp_path), check=True)
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path), check=True, capture_output=True,
    )

    with pytest.raises(PreRegistrationError, match="no mechanism_tags"):
        check_mechanism_pre_registered(repo_root=tmp_path, config_path=cfg, setup_name="my_setup")


def test_pre_registration_fails_when_tags_added_in_head_commit(tmp_path):
    """If mechanism_tags first appears in HEAD, that's post-hoc — fail."""
    _init_git_repo(tmp_path)

    cfg = tmp_path / "config.json"
    cfg.write_text('{"setups": {"my_setup": {"enabled": true}}}', encoding="utf-8")
    subprocess.run(["git", "add", "config.json"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "init"],
                   cwd=str(tmp_path), check=True, capture_output=True)

    # Add mechanism_tags in a commit that becomes HEAD
    cfg.write_text(
        '{"setups": {"my_setup": {"enabled": true, "mechanism_tags": ["FII_net_flow_positive_30d"]}}}',
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "config.json"], cwd=str(tmp_path), check=True)
    subprocess.run(["git", "commit", "-m", "add tags"],
                   cwd=str(tmp_path), check=True, capture_output=True)

    with pytest.raises(PreRegistrationError, match="post-hoc"):
        check_mechanism_pre_registered(repo_root=tmp_path, config_path=cfg, setup_name="my_setup")
