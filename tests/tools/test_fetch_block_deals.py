"""Tests for tools.block_deal_calendar.fetch_block_deals.

Block-deal parsing is pinned (regression guard for the 2026-10-06 bulk
extension) and the new NSE BULK path is covered with a scripted curl_cffi
session (no network): CSV parse of Indian-grouped quantities ("4,50,000"),
BUY/SELL normalisation, IST-naive dates, legacy-cache import, merge/dedupe, the
``--deal-type bulk`` CLI and cookie re-bootstrap on 403.
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tools.block_deal_calendar import fetch_block_deals as mod


# ---------------------------------------------------------------------------
# Fakes.
# ---------------------------------------------------------------------------

class FakeResponse:
    def __init__(self, status_code=200, body=b"", headers=None):
        self.status_code = status_code
        self.content = body.encode("utf-8") if isinstance(body, str) else body
        self.headers = headers or {}

    def json(self):
        import json
        return json.loads(self.content.decode("utf-8"))


class FakeCurlSession:
    def __init__(self, script: dict):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls: list[tuple[str, dict | None]] = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params) if params else None))
        queue = self.script.get(url)
        if not queue:
            return FakeResponse(200, "<html>home</html>")
        return queue.pop(0) if len(queue) > 1 else queue[0]


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(mod.time, "sleep", lambda *_a, **_k: None)


def _install(monkeypatch, script: dict) -> FakeCurlSession:
    fake = FakeCurlSession(script)
    monkeypatch.setattr(mod, "_HAS_CURL_CFFI", True)
    monkeypatch.setattr(mod.crequests, "Session", lambda impersonate=None: fake)
    return fake


BULK_CSV = (
    "﻿Date ,Symbol ,Security Name ,Client Name ,Buy / Sell ,Quantity Traded ,"
    "Trade Price / Wght. Avg. Price ,Remarks \n"
    "05-OCT-2026,ACEVECTOR,AceVector Limited,GRT STRATEGIC VENTURES LLP,BUY,\"28,90,953\",25.65,-\n"
    "05-OCT-2026,AJOONI,Ajooni Biotech Limited,ANKITA VISHAL SHAH,sell,\"4,50,000\",7.28,-\n"
    "06-OCT-2026,SMALL,Small Co,SOMEONE,BUY,5,7.00,-\n"
    "06-OCT-2026,BADQTY,Bad Co,SOMEONE,BUY,abc,7.00,-\n"
    "06-OCT-2026,BADSIDE,Bad Co,SOMEONE,HOLD,\"1,000\",7.00,-\n"
    "bad-date,X,Bad Co,SOMEONE,BUY,\"1,000\",7.00,-\n"
)


# ---------------------------------------------------------------------------
# Block path: regression pins (behaviour must be unchanged).
# ---------------------------------------------------------------------------

def test_block_parse_nse_rows_unchanged():
    rows = mod.parse_nse_rows([
        {"BD_DT_DATE": "05-OCT-2026", "BD_SYMBOL": "ACEVECTOR", "BD_SCRIP_NAME": "AceVector",
         "BD_CLIENT_NAME": "GRT", "BD_BUY_SELL": "BUY", "BD_QTY_TRD": 2890953,
         "BD_TP_WATP": 25.65, "BD_REMARKS": "-"},
        {"BD_DT_DATE": "05-OCT-2026", "BD_SYMBOL": "PS", "BD_SCRIP_NAME": "x",
         "BD_CLIENT_NAME": "c", "BD_BUY_SELL": "S", "BD_QTY_TRD": "10", "BD_TP_WATP": "1.5"},
        {"BD_DT_DATE": "bad", "BD_SYMBOL": "X", "BD_BUY_SELL": "BUY", "BD_QTY_TRD": 1, "BD_TP_WATP": 1},
    ])
    assert len(rows) == 2
    assert rows[0]["symbol"] == "NSE:ACEVECTOR" and rows[0]["trade_date"] == date(2026, 10, 5)
    assert rows[0]["trade_value_cr"] == round(2890953 * 25.65 / 1e7, 4)
    assert rows[1]["buy_or_sell"] == "SELL"
    assert list(rows[0]) == mod._OUTPUT_COLUMNS


def test_block_write_events_dedupes(tmp_path):
    out = tmp_path / "block.parquet"
    row = {"trade_date": date(2026, 10, 5), "symbol": "NSE:A", "raw_symbol": "A",
           "client_name": "c", "buy_or_sell": "BUY", "qty": 10, "trade_price": 1.0,
           "trade_value_cr": 0.0, "exchange": "NSE", "company_name": "A"}
    mod.write_events([row], out)
    df = mod.write_events([row, dict(row, qty=20)], out)
    assert len(df) == 2
    assert list(df.columns) == mod._OUTPUT_COLUMNS


def test_block_cli_still_requires_start_end_and_rejects_import_legacy(tmp_path):
    with pytest.raises(SystemExit):
        mod.main(["--skip-nse", "--skip-bse"])
    with pytest.raises(SystemExit):
        mod.main(["--start", "2026-10-05", "--end", "2026-10-06", "--skip-nse", "--skip-bse",
                  "--import-legacy", str(tmp_path / "x.parquet")])


# ---------------------------------------------------------------------------
# Bulk: parsing.
# ---------------------------------------------------------------------------

def test_parse_indian_int_and_side():
    assert mod.parse_indian_int("4,50,000") == 450000
    assert mod.parse_indian_int("28,90,953") == 2890953
    assert mod.parse_indian_int(5) == 5
    assert mod.parse_indian_int("5") == 5
    assert mod.parse_indian_int("abc") is None
    assert mod.parse_indian_int(None) is None
    assert mod.normalize_side("BUY") == "BUY"
    assert mod.normalize_side("buy ") == "BUY"
    assert mod.normalize_side("P") == "BUY"
    assert mod.normalize_side("SELL") == "SELL"
    assert mod.normalize_side("s") == "SELL"
    assert mod.normalize_side("HOLD") is None
    assert mod.normalize_side(None) is None


def test_parse_nse_bulk_csv_normalises_schema():
    rows = mod.parse_nse_bulk_csv(BULK_CSV)
    assert [r["symbol"] for r in rows] == ["ACEVECTOR", "AJOONI", "SMALL"]
    r0, r1, r2 = rows
    assert list(r0) == mod.BULK_COLUMNS
    assert r0["date"] == pd.Timestamp("2026-10-05") and r0["date"].tzinfo is None
    assert r0["qty"] == 2890953 and r0["price"] == 25.65 and r0["side"] == "BUY"
    assert r0["security_name"] == "AceVector Limited"
    assert r0["client_name"] == "GRT STRATEGIC VENTURES LLP"
    assert r0["remarks"] == "-" and r0["source"] == "nse_bulk_deals"
    assert r1["qty"] == 450000 and r1["side"] == "SELL"
    assert r2["qty"] == 5


def test_parse_nse_bulk_csv_empty_and_bad_header():
    assert mod.parse_nse_bulk_csv("") == []
    assert mod.parse_nse_bulk_csv("foo,bar\n1,2\n") == []


# ---------------------------------------------------------------------------
# Bulk: legacy import.
# ---------------------------------------------------------------------------

def _legacy_df():
    return pd.DataFrame({
        "Date": ["03-JAN-2023", "03-JAN-2023", "30-APR-2026"],
        "Symbol": ["AJOONI", "AJOONI", "TICL"],
        "SecurityName": ["Ajooni Biotech Limited", "Ajooni Biotech Limited", "Twamev Cons"],
        "ClientName": ["ANKITA VISHAL SHAH", "ANKITA VISHAL SHAH", "RAJAT MISHRA"],
        "Buy/Sell": ["BUY", "SELL", "BUY"],
        "QuantityTraded": ["5", "4,50,000", "10,04,710"],
        "TradePrice/Wght.Avg.Price": ["7.00", "7.28", "23.45"],
        "Remarks": ["-", "-", "-"],
    })


def test_import_legacy_bulk(tmp_path):
    legacy = tmp_path / "legacy.parquet"
    _legacy_df().to_parquet(legacy, index=False)
    rows = mod.import_legacy_bulk(legacy)
    assert len(rows) == 3
    assert rows[1]["qty"] == 450000 and rows[1]["side"] == "SELL"
    assert rows[0]["date"] == pd.Timestamp("2023-01-03")
    assert rows[2]["symbol"] == "TICL" and rows[2]["qty"] == 1004710
    assert {r["source"] for r in rows} == {"legacy_cache"}


def test_import_legacy_bulk_missing_columns_raises(tmp_path):
    bad = tmp_path / "bad.parquet"
    pd.DataFrame({"Date": ["03-JAN-2023"], "Symbol": ["A"]}).to_parquet(bad, index=False)
    with pytest.raises(ValueError):
        mod.import_legacy_bulk(bad)


# ---------------------------------------------------------------------------
# Bulk: persistence.
# ---------------------------------------------------------------------------

def test_write_bulk_events_merges_dedupes_and_upgrades_source(tmp_path):
    out = tmp_path / "bulk.parquet"
    legacy = tmp_path / "legacy.parquet"
    _legacy_df().to_parquet(legacy, index=False)
    mod.write_bulk_events(mod.import_legacy_bulk(legacy), out)

    # Re-scrape only one of the legacy deals plus one new deal.
    scraped = mod.parse_nse_bulk_csv(
        "Date ,Symbol ,Security Name ,Client Name ,Buy / Sell ,Quantity Traded ,"
        "Trade Price / Wght. Avg. Price ,Remarks \n"
        "03-JAN-2023,AJOONI,Ajooni Biotech Limited,ANKITA VISHAL SHAH,SELL,\"4,50,000\",7.28,-\n"
        "05-OCT-2026,NEWCO,New Co,BUYER,BUY,\"1,00,000\",10.5,-\n"
    )
    df = mod.write_bulk_events(scraped, out)
    assert list(df.columns) == mod.BULK_COLUMNS
    assert len(df) == 4                                  # 3 legacy (1 replaced) + 1 new
    sell = df[(df["symbol"] == "AJOONI") & (df["side"] == "SELL")]
    assert len(sell) == 1 and sell["source"].iloc[0] == "nse_bulk_deals"
    assert df.loc[df["symbol"] == "TICL", "source"].iloc[0] == "legacy_cache"
    assert str(df["date"].dtype) == "datetime64[ns]" and df["date"].dt.tz is None
    assert str(df["qty"].dtype) == "Int64"
    back = pd.read_parquet(out)
    assert len(back) == 4 and back["date"].min() == pd.Timestamp("2023-01-03")


def test_write_bulk_events_empty_keeps_existing(tmp_path):
    out = tmp_path / "bulk.parquet"
    mod.write_bulk_events(mod.parse_nse_bulk_csv(BULK_CSV), out)
    df = mod.write_bulk_events([], out)
    assert len(df) == 3


# ---------------------------------------------------------------------------
# Bulk: HTTP + CLI.
# ---------------------------------------------------------------------------

def test_bulk_fetch_uses_csv_form_and_rebootstraps_on_403(monkeypatch, no_sleep):
    fake = _install(monkeypatch, {
        mod._NSE_API: [FakeResponse(403, ""), FakeResponse(200, BULK_CSV.encode("utf-8"))],
    })
    rows, stats = mod.fetch_nse_bulk_range(date(2026, 10, 5), date(2026, 10, 6), sleep_secs=0)
    api_calls = [p for u, p in fake.calls if u == mod._NSE_API]
    assert len(api_calls) == 2
    assert api_calls[0] == {"optionType": "bulk_deals", "from": "05-10-2026",
                            "to": "06-10-2026", "csv": "true"}
    # bootstrap visits home + detail page: initial (2) + after the 403 (2)
    boot_calls = [u for u, _ in fake.calls if u in (mod._NSE_HOME, mod._NSE_DETAIL)]
    assert len(boot_calls) == 4
    assert stats == {"chunks_attempted": 1, "chunks_ok": 1, "chunks_failed": 0, "raw_records": 6}
    assert len(rows) == 3


def test_bulk_fetch_html_body_rebootstraps(monkeypatch, no_sleep):
    _install(monkeypatch, {
        mod._NSE_API: [FakeResponse(200, "<html>akamai</html>"),
                       FakeResponse(200, BULK_CSV.encode("utf-8"))],
    })
    rows, stats = mod.fetch_nse_bulk_range(date(2026, 10, 5), date(2026, 10, 6), sleep_secs=0)
    assert stats["chunks_ok"] == 1 and len(rows) == 3


def test_bulk_fetch_weekly_chunks_and_total_failure_rc(monkeypatch, no_sleep, tmp_path):
    fake = _install(monkeypatch, {mod._NSE_API: [FakeResponse(404, "")]})
    rc = mod.main(["--deal-type", "bulk", "--start", "2026-09-20", "--end", "2026-10-06",
                   "--sleep-secs", "0", "--out-path", str(tmp_path / "b.parquet")])
    api_calls = [p for u, p in fake.calls if u == mod._NSE_API]
    assert [(p["from"], p["to"]) for p in api_calls] == [
        ("20-09-2026", "26-09-2026"), ("27-09-2026", "03-10-2026"), ("04-10-2026", "06-10-2026"),
    ]
    assert rc == 4


def test_bulk_cli_import_legacy_only_no_network(monkeypatch, tmp_path):
    legacy = tmp_path / "legacy.parquet"
    _legacy_df().to_parquet(legacy, index=False)
    out = tmp_path / "bulk.parquet"

    def boom(*_a, **_k):
        raise AssertionError("network must not be touched")
    monkeypatch.setattr(mod, "NSEBlockDealClient", boom)

    rc = mod.main(["--deal-type", "bulk", "--import-legacy", str(legacy), "--out-path", str(out)])
    assert rc == 0
    df = pd.read_parquet(out)
    assert len(df) == 3 and set(df["source"]) == {"legacy_cache"}


def test_bulk_cli_requires_start_end_without_import(tmp_path):
    with pytest.raises(SystemExit):
        mod.main(["--deal-type", "bulk", "--out-path", str(tmp_path / "b.parquet")])
