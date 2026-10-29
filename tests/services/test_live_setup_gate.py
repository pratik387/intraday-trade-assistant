"""Live-mode intraday setup gate (2026-10-29).

`enabled` is read by paper, backtest AND live, so before this gate every
paper-validated intraday setup also traded real money on the live daemon
(earnings_downshock_continuation_short shorted HUHTAMAKI live on 2026-10-28).
`live_enabled` is now required, and read only when the run is live.
"""
import io
import json
from pathlib import Path

import pytest

import services.plan_orchestrator as po

ROOT = Path(__file__).resolve().parents[2]
CFG = json.load(io.open(ROOT / "config" / "configuration.json", encoding="utf-8"))
ENABLED_INTRADAY = {
    n: r for n, r in (CFG.get("setups") or {}).items()
    if isinstance(r, dict) and str(r.get("horizon")) != "multi_day"
    and r.get("mode", "intraday") == "intraday" and r.get("enabled")
}


@pytest.fixture
def cfg(monkeypatch):
    root = {"setups": {
        "live_one": {"enabled": True, "mode": "intraday", "live_enabled": True},
        "paper_one": {"enabled": True, "mode": "intraday", "live_enabled": False},
        "unflagged": {"enabled": True, "mode": "intraday"},
        "off": {"enabled": False, "mode": "intraday"},
        "night": {"enabled": True, "mode": "overnight"},
    }}
    monkeypatch.setattr(po, "_ROOT_CFG_CACHE", root)
    return root


def test_mode_must_be_set_before_the_gate_is_read(cfg):
    with pytest.raises(po.OrchestratorConfigError):
        po._live_gate_blocks("live_one", cfg["setups"]["live_one"])


def test_unknown_mode_is_rejected(cfg):
    with pytest.raises(ValueError):
        po.set_runtime_trading_mode("real")


@pytest.mark.parametrize("mode", ["paper", "backtest"])
def test_paper_and_backtest_ignore_live_enabled(cfg, mode):
    po.set_runtime_trading_mode(mode)
    for name in ("live_one", "paper_one", "unflagged"):
        assert po._live_gate_blocks(name, cfg["setups"][name]) is False


def test_live_blocks_setups_not_live_enabled(cfg):
    del cfg["setups"]["unflagged"]
    po.set_runtime_trading_mode("live")
    assert po._live_gate_blocks("live_one", cfg["setups"]["live_one"]) is False
    assert po._live_gate_blocks("paper_one", cfg["setups"]["paper_one"]) is True


def test_live_startup_fails_on_an_enabled_intraday_setup_without_the_flag(cfg):
    with pytest.raises(po.OrchestratorConfigError, match="unflagged"):
        po.set_runtime_trading_mode("live")


def test_live_startup_rejects_a_string_flag(cfg):
    cfg["setups"]["unflagged"]["live_enabled"] = "false"
    with pytest.raises(po.OrchestratorConfigError, match="unflagged"):
        po.set_runtime_trading_mode("live")


def test_live_gate_ignores_overnight_setups(cfg):
    del cfg["setups"]["unflagged"]
    po.set_runtime_trading_mode("live")
    assert po._live_gate_blocks("night", cfg["setups"]["night"]) is False


def test_live_startup_returns_the_live_list(cfg):
    del cfg["setups"]["unflagged"]
    assert po.set_runtime_trading_mode("live") == ["live_one"]


def test_get_detector_skips_a_paper_only_setup_in_live(cfg, monkeypatch):
    del cfg["setups"]["unflagged"]
    po.set_runtime_trading_mode("live")

    class _Spec:
        enabled = True
        detector_class_path = "x.Y"

    class _Reg:
        def get(self, name):
            return _Spec()

    built = []
    monkeypatch.setattr(po, "_get_registry", lambda: _Reg())
    monkeypatch.setattr(po, "_import_path", lambda p: (lambda c: built.append(c) or object()))
    orch = po.PlanOrchestrator()
    assert orch._get_detector("paper_one") is None
    assert built == []
    assert orch._get_detector("live_one") is not None


@pytest.mark.parametrize("name", sorted(ENABLED_INTRADAY))
def test_real_config_declares_live_enabled_on_every_enabled_intraday_setup(name):
    assert isinstance(ENABLED_INTRADAY[name].get("live_enabled"), bool), name


def test_real_config_only_up_spike_trades_live():
    live = sorted(n for n, r in ENABLED_INTRADAY.items() if r.get("live_enabled"))
    assert live == ["up_spike_fade_short"]


def test_main_sets_the_trading_mode():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "set_runtime_trading_mode(" in src
