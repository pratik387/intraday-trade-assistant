"""Option-chain ingestion package — sub-project #8 (expiry_pin_strike_reversal).

NSE F&O bhavcopy ingestion + parquet store. Daily settlement-OI snapshots
keyed by (session_date, symbol, expiry_date, strike, option_type).

Public CLI: tools/option_chain/fetch_oi_snapshot.py
Public loader API: services/option_chain_loader.py

Per specs/2026-04-29-expiry_pin_strike_reversal-plan.md Part A and
docs/option_chain_data_source.md.
"""
