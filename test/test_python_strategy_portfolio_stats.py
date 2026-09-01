"""Tests for hosted strategy= tagging and Python portfolio stats."""

from datetime import datetime, timedelta

from services.python_strategy_stats_service import (
    compute_correlations,
    fifo_round_trips,
    strategy_tag_aliases,
)
from utils.hosted_strategy_tag import apply_default_strategy_tag, hosted_strategy_name


def test_apply_default_strategy_tag_fills_missing_and_blank():
    assert apply_default_strategy_tag({}, "ORB Spread") == {"strategy": "ORB Spread"}
    assert apply_default_strategy_tag({"strategy": ""}, "ORB Spread") == {"strategy": "ORB Spread"}
    assert apply_default_strategy_tag({"strategy": "   "}, "ORB Spread") == {
        "strategy": "ORB Spread"
    }


def test_apply_default_strategy_tag_does_not_override_explicit():
    kwargs = {"strategy": "CustomTag", "symbol": "NIFTY"}
    assert apply_default_strategy_tag(kwargs, "ORB Spread")["strategy"] == "CustomTag"


def test_hosted_strategy_name_from_env(monkeypatch):
    monkeypatch.setenv("STRATEGY_NAME", "Gap VWAP")
    assert hosted_strategy_name() == "Gap VWAP"
    assert hosted_strategy_name("  Explicit  ") == "Explicit"


def test_fifo_round_trips_long_then_short():
    t0 = datetime(2026, 9, 1, 9, 30)
    trades = [
        {
            "symbol": "NIFTY",
            "exchange": "NFO",
            "product": "MIS",
            "action": "BUY",
            "quantity": 75,
            "price": 100,
            "trade_timestamp": t0,
        },
        {
            "symbol": "NIFTY",
            "exchange": "NFO",
            "product": "MIS",
            "action": "SELL",
            "quantity": 75,
            "price": 110,
            "trade_timestamp": t0 + timedelta(hours=1),
        },
    ]
    closed, leftover = fifo_round_trips(trades)
    assert leftover == 0
    assert len(closed) == 1
    assert closed[0]["pnl"] == 750


def test_fifo_round_trips_partial_and_open():
    trades = [
        {
            "symbol": "SBIN",
            "exchange": "NSE",
            "product": "MIS",
            "action": "BUY",
            "quantity": 100,
            "price": 800,
        },
        {
            "symbol": "SBIN",
            "exchange": "NSE",
            "product": "MIS",
            "action": "SELL",
            "quantity": 40,
            "price": 810,
        },
    ]
    closed, leftover = fifo_round_trips(trades)
    assert leftover == 60
    assert len(closed) == 1
    assert closed[0]["pnl"] == 400


def test_strategy_tag_aliases_include_name_id_and_stem():
    aliases = strategy_tag_aliases(
        "orb_spread_signal_20260716003641",
        {
            "name": "ORB Spread",
            "file_name": "orb_spread_signal.py",
            "file_path": "C:/strats/orb_spread_signal.py",
        },
    )
    assert "orb spread" in aliases
    assert "orb_spread_signal_20260716003641" in aliases
    assert "orb_spread_signal" in aliases
    assert "orb_spread_signal.py" in aliases
    assert "orb_spread" in aliases


def test_strategy_tag_aliases_include_config_and_timestamp_stem():
    aliases = strategy_tag_aliases(
        "gap_vwap_asym_signal_20260825095400",
        {
            "name": "Gap_VWAP_Asym",
            "file_name": "gap_vwap_asym_signal_20260825095400.py",
            "file_path": "scripts/gap_vwap_asym_signal_20260825095400.py",
            "strategy_tags": ["ORB_Asym", "gap_vwap_asym"],
        },
    )
    assert "orb_asym" in aliases
    assert "gap_vwap_asym" in aliases
    assert "gap_and_go" not in aliases


def test_strategy_tag_aliases_scrape_script_literals(tmp_path):
    script = tmp_path / "gap_and_go_signal.py"
    script.write_text(
        'STRATEGY = "ignored_if_json_present"\n'
        'client.post(json={"apikey": "x", "strategy": "gap_and_go", "symbol": "SBIN"})\n',
        encoding="utf-8",
    )
    aliases = strategy_tag_aliases(
        "gap_and_go_signal_20260804000253",
        {
            "name": "Gap_and_Go_intraday",
            "file_path": str(script),
        },
    )
    assert "gap_and_go" in aliases
    assert "gap_and_go_intraday" in aliases
    items = [
        {
            "id": "a",
            "name": "A",
            "daily_pnl": {"2026-09-01": 10.0, "2026-09-02": -10.0},
        },
        {
            "id": "b",
            "name": "B",
            "daily_pnl": {"2026-09-01": -10.0, "2026-09-02": 10.0},
        },
    ]
    pairs = compute_correlations(items)
    assert len(pairs) == 1
    assert pairs[0]["correlation"] == -1.0
