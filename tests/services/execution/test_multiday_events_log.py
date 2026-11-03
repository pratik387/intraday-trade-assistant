"""The multi_day cron bypasses setup_logging, so main.py configures the
events.jsonl writer itself (2026-11-03; before that every ENTRY/EXIT event
write failed with a None path). One family file across cron runs."""
import io
import json
from pathlib import Path

import pandas as pd

from diagnostics.diag_event_log import _EventWriter, diag_event_log

ROOT = Path(__file__).resolve().parents[3]
CFG = json.load(io.open(ROOT / "config" / "configuration.json", encoding="utf-8"))


def test_config_names_one_family_events_dir():
    assert CFG["multi_day_portfolio"]["events_log_dir"] == "logs/multiday_events"


def test_main_configures_the_writer_for_paper_and_live_multiday_runs_only():
    src = (ROOT / "main.py").read_text(encoding="utf-8")
    branch = src[src.index("else:  # multi_day"):src.index("[multi_day exit-pass]")]
    assert "if not args.dry_run:" in branch
    assert 'diag_event_log.set_output(' in branch
    # key read before the try (fail fast); writer errors never stop the run
    assert branch.index('cfg["multi_day_portfolio"]["events_log_dir"]') < branch.index("try:")
    assert "except OSError" in branch


def test_entry_and_exit_from_two_runs_land_in_one_file(tmp_path):
    saved = (_EventWriter._shared_dir, _EventWriter._shared_run_id)
    try:
        for run, call in (("multiday_verify-entry_2026-11-03", "entry"), ("multiday_exit_2026-11-05", "exit")):
            diag_event_log.set_output(out_dir=str(tmp_path), run_id=run)
            plan = {"trade_id": "low52_2026-11-02_ABC", "setup": "low52_capitulation_revert_long"}
            if call == "entry":
                diag_event_log.log_entry_fill(symbol="NSE:ABC", plan=plan, side="BUY", qty=10, price=100.0,
                                              entry_ts=pd.Timestamp("2026-11-03 09:33"), order_meta={})
            else:
                diag_event_log.log_exit(symbol="NSE:ABC", plan=plan, reason="kday_close_moc", exit_price=101.0,
                                        exit_qty=10, ts=pd.Timestamp("2026-11-05 15:28"), pnl=5.0)
            diag_event_log.reset()
        rows = [json.loads(l) for l in (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
        assert [r["type"] for r in rows] == ["ENTRY", "EXIT"]
        assert {r["trade_id"] for r in rows} == {"low52_2026-11-02_ABC"}
        assert [r["run_id"] for r in rows] == ["multiday_verify-entry_2026-11-03", "multiday_exit_2026-11-05"]
    finally:
        diag_event_log.reset()
        _EventWriter._shared_dir, _EventWriter._shared_run_id = saved
