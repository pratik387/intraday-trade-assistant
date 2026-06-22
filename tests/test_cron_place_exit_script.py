from pathlib import Path

def test_place_exit_cron_exists_and_executable():
    p = Path("scripts/cron-place-exit.sh")
    assert p.exists(), "scripts/cron-place-exit.sh missing"
    text = p.read_text()
    assert "--mode overnight --action place-exit" in text
    assert text.startswith("#!/bin/bash")
