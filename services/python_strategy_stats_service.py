"""Per-hosted-strategy stats for Strategy Portfolio.

Attributes Analyzer/sandbox fills by the ``strategy`` tag on each trade,
matched to a hosted Python strategy's display name, id, or file stem.

Round-trip P&L is FIFO per (symbol, exchange, product). Live broker books
are not strategy-tagged, so this grid is accurate for Analyzer/sandbox
fills (and any other path that writes ``sandbox_trades.strategy``).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from datetime import date, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import func

from utils.logging import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252
MIN_DAYS_FOR_SHARPE = 2
CORRELATION_MIN_DAYS = 2

# Hosted files are often ``name_signal_YYYYMMDDHHMMSS.py``.
_TIMESTAMP_SUFFIX = re.compile(r"_\d{14}$")
# Hardcoded REST/SDK tags inside the strategy script.
_SCRIPT_STRATEGY_TAG = re.compile(
    r"""(?:STRATEGY\s*=\s*|["']strategy["']\s*:\s*)["']([^"']+)["']""",
    re.IGNORECASE,
)


def _alias_forms(value: Any) -> set[str]:
    text = str(value or "").strip()
    if not text:
        return set()
    lower = text.lower()
    return {lower, lower.replace(" ", "_"), lower.replace("_", " ")}


def _stem_variants(stem: str) -> set[str]:
    variants = {stem}
    stripped = _TIMESTAMP_SUFFIX.sub("", stem)
    variants.add(stripped)
    for item in list(variants):
        if item.endswith("_signal"):
            variants.add(item[: -len("_signal")])
    return {v for v in variants if v}


def _resolve_strategy_file(file_path: str) -> Path | None:
    path = Path(file_path)
    if path.is_file():
        return path
    try:
        from blueprints.python_strategy import STRATEGIES_ROOT

        candidate = STRATEGIES_ROOT / file_path
        if candidate.is_file():
            return candidate
    except Exception:
        return None
    return None


def _tags_from_script(file_path: str) -> set[str]:
    path = _resolve_strategy_file(file_path)
    if path is None:
        return set()
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return set()
    return {
        match.group(1).strip()
        for match in _SCRIPT_STRATEGY_TAG.finditer(text)
        if match.group(1).strip()
    }


def strategy_tag_aliases(strategy_id: str, config: dict[str, Any]) -> set[str]:
    """Names that should match ``sandbox_trades.strategy`` for this host entry.

    Includes the hosted display name, id, file stem (with ``_signal`` / timestamp
    stripped), optional ``strategy_tags`` on the config, and tags scraped from
    the script (``STRATEGY = "..."`` / ``"strategy": "..."``). Scripts that POST
    JSON with a hardcoded tag still attribute to this row.
    """
    aliases: set[str] = set()
    aliases |= _alias_forms(strategy_id)
    aliases |= _alias_forms(config.get("name"))
    aliases |= _alias_forms(config.get("file_name"))
    file_path = config.get("file_path") or ""
    if file_path:
        stem = Path(file_path).stem
        aliases |= _alias_forms(stem)
        aliases |= _alias_forms(Path(file_path).name)
        for variant in _stem_variants(stem):
            aliases |= _alias_forms(variant)
        for form in _tags_from_script(file_path):
            aliases |= _alias_forms(form)

    extra = config.get("strategy_tags") or config.get("aliases") or []
    if isinstance(extra, str):
        extra = [extra]
    for tag in extra:
        aliases |= _alias_forms(tag)

    return {a.strip() for a in aliases if a and str(a).strip()}


def fifo_round_trips(trades: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Close opposite fills FIFO. Returns (closed round-trips, leftover lots)."""
    books: dict[tuple[str, str, str], deque] = defaultdict(deque)
    closed: list[dict[str, Any]] = []

    for trade in trades:
        action = str(trade.get("action") or "").upper()
        if action not in {"BUY", "SELL"}:
            continue
        try:
            qty = int(trade.get("quantity") or 0)
            price = float(trade.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if qty <= 0:
            continue
        key = (
            str(trade.get("symbol") or ""),
            str(trade.get("exchange") or ""),
            str(trade.get("product") or ""),
        )
        book = books[key]
        remaining = qty
        ts = trade.get("trade_timestamp")
        while remaining > 0 and book and book[0]["action"] != action:
            lot = book[0]
            close_qty = min(lot["qty"], remaining)
            sign = 1 if lot["action"] == "BUY" else -1
            pnl = sign * (price - lot["price"]) * close_qty
            closed.append(
                {
                    "symbol": key[0],
                    "exchange": key[1],
                    "product": key[2],
                    "quantity": close_qty,
                    "pnl": pnl,
                    "exit_time": ts,
                }
            )
            lot["qty"] -= close_qty
            remaining -= close_qty
            if lot["qty"] == 0:
                book.popleft()
        if remaining > 0:
            book.append({"qty": remaining, "price": price, "action": action, "time": ts})

    leftover = sum(lot["qty"] for book in books.values() for lot in book)
    return closed, leftover


def _exit_date(ts: Any) -> date | None:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts.date()
    if isinstance(ts, date):
        return ts
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).date()
    except ValueError:
        return None


def _annualised_sharpe(daily_pnl: list[float]) -> float | None:
    if len(daily_pnl) < MIN_DAYS_FOR_SHARPE:
        return None
    mean = sum(daily_pnl) / len(daily_pnl)
    var = sum((x - mean) ** 2 for x in daily_pnl) / (len(daily_pnl) - 1)
    if var <= 0:
        return None
    # Rupee P&L series (no allocated capital on the host config) — scale like
    # a daily return Sharpe so the number is comparable across days, not strategies.
    return (mean / math.sqrt(var)) * math.sqrt(TRADING_DAYS_PER_YEAR)


def _metrics_from_round_trips(closed: list[dict[str, Any]], leftover: int) -> dict[str, Any]:
    n = len(closed)
    if n == 0:
        return {
            "n_trades": 0,
            "win_rate": None,
            "profit_factor": None,
            "avg_pnl": None,
            "total_pnl": 0.0,
            "sharpe": None,
            "open_quantity": leftover,
            "daily_pnl": {},
        }

    pnls = [float(t["pnl"]) for t in closed]
    wins = sum(1 for p in pnls if p > 0)
    gross_win = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p <= 0))
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float("inf")
    daily: dict[str, float] = defaultdict(float)
    for trip in closed:
        d = _exit_date(trip.get("exit_time"))
        if d is not None:
            daily[d.isoformat()] += float(trip["pnl"])

    return {
        "n_trades": n,
        "win_rate": (wins / n) * 100.0,
        "profit_factor": profit_factor,
        "avg_pnl": sum(pnls) / n,
        "total_pnl": sum(pnls),
        "sharpe": _annualised_sharpe(list(daily.values())) if daily else None,
        "open_quantity": leftover,
        "daily_pnl": dict(daily),
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    n = len(xs)
    if n < CORRELATION_MIN_DAYS:
        return None
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    return cov / denom


def compute_correlations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for i, a in enumerate(items):
        for b in items[i + 1 :]:
            dates = sorted(set(a["daily_pnl"]) | set(b["daily_pnl"]))
            if len(dates) < CORRELATION_MIN_DAYS:
                pairs.append(
                    {
                        "a": a["id"],
                        "b": b["id"],
                        "a_name": a["name"],
                        "b_name": b["name"],
                        "correlation": None,
                        "n_days": len(dates),
                    }
                )
                continue
            xs = [float(a["daily_pnl"].get(d, 0.0)) for d in dates]
            ys = [float(b["daily_pnl"].get(d, 0.0)) for d in dates]
            pairs.append(
                {
                    "a": a["id"],
                    "b": b["id"],
                    "a_name": a["name"],
                    "b_name": b["name"],
                    "correlation": _pearson(xs, ys),
                    "n_days": len(dates),
                }
            )
    return pairs


def _load_sandbox_trades() -> list[dict[str, Any]]:
    try:
        from database.sandbox_db import SandboxTrades, db_session
    except Exception:
        logger.exception("Could not import sandbox trades for strategy stats")
        return []

    try:
        rows = (
            db_session.query(SandboxTrades)
            .order_by(SandboxTrades.trade_timestamp.asc(), SandboxTrades.id.asc())
            .all()
        )
        trades = []
        for row in rows:
            trades.append(
                {
                    "strategy": (row.strategy or "").strip(),
                    "symbol": row.symbol,
                    "exchange": row.exchange,
                    "product": row.product,
                    "action": row.action,
                    "quantity": row.quantity,
                    "price": row.price,
                    "trade_timestamp": row.trade_timestamp,
                }
            )
        return trades
    except Exception:
        logger.exception("Failed to load sandbox trades for strategy stats")
        return []
    finally:
        try:
            db_session.remove()
        except Exception:
            pass


def _load_sandbox_order_counts() -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    try:
        from database.sandbox_db import SandboxOrders, db_session
    except Exception:
        logger.exception("Could not import sandbox orders for strategy stats")
        return counts

    try:
        rows = (
            db_session.query(func.lower(SandboxOrders.strategy), func.count(SandboxOrders.id))
            .group_by(func.lower(SandboxOrders.strategy))
            .all()
        )
        for tag, count in rows:
            if tag:
                counts[str(tag)] = int(count)
        return counts
    except Exception:
        logger.exception("Failed to load sandbox order counts for strategy stats")
        return counts
    finally:
        try:
            db_session.remove()
        except Exception:
            pass


def _json_safe_metric(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float):
        if math.isinf(value):
            return None, "inf"
        if math.isnan(value):
            return None
    return value


def build_python_portfolio(user_id: str | None = None) -> dict[str, Any]:
    """Build the Python tab payload for Strategy Portfolio."""
    from blueprints.python_strategy import (
        STRATEGY_CONFIGS,
        cleanup_dead_processes,
        get_schedule_status,
        normalize_exchange,
    )

    cleanup_dead_processes()
    trades = _load_sandbox_trades()
    order_counts = _load_sandbox_order_counts()

    by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        tag = (trade.get("strategy") or "").strip().lower()
        if tag:
            by_tag[tag].append(trade)

    items: list[dict[str, Any]] = []
    for strategy_id, config in STRATEGY_CONFIGS.items():
        if user_id and config.get("user_id") and config.get("user_id") != user_id:
            continue
        aliases = strategy_tag_aliases(strategy_id, config)
        matched: list[dict[str, Any]] = []
        seen_ids: set[int] = set()
        for alias in aliases:
            for trade in by_tag.get(alias, []):
                marker = id(trade)
                if marker in seen_ids:
                    continue
                seen_ids.add(marker)
                matched.append(trade)
        matched.sort(key=lambda t: (t.get("trade_timestamp") is None, t.get("trade_timestamp")))
        closed, leftover = fifo_round_trips(matched)
        metrics = _metrics_from_round_trips(closed, leftover)

        if config.get("is_running"):
            status, status_message = "running", "Running"
        elif config.get("error_message"):
            status, status_message = "error", config.get("error_message")
        else:
            status, status_message = get_schedule_status(config)

        n_orders = sum(order_counts.get(alias, 0) for alias in aliases)
        pf = metrics["profit_factor"]
        pf_out: float | None
        pf_inf = False
        if pf == float("inf"):
            pf_out, pf_inf = None, True
        else:
            pf_out, pf_inf = pf, False

        items.append(
            {
                "id": strategy_id,
                "name": config.get("name") or strategy_id,
                "file_name": config.get("file_name") or "",
                "exchange": normalize_exchange(config.get("exchange")),
                "status": status,
                "status_message": status_message,
                "n_orders": n_orders,
                "n_fills": len(matched),
                "n_trades": metrics["n_trades"],
                "win_rate": metrics["win_rate"],
                "profit_factor": pf_out,
                "profit_factor_infinite": pf_inf,
                "avg_pnl": metrics["avg_pnl"],
                "total_pnl": metrics["total_pnl"],
                "sharpe": metrics["sharpe"],
                "open_quantity": metrics["open_quantity"],
                "daily_pnl": metrics["daily_pnl"],
            }
        )

    correlations = compute_correlations(items)
    public_items = []
    for item in items:
        public = dict(item)
        public.pop("daily_pnl", None)
        public_items.append(public)

    return {
        "items": public_items,
        "correlations": [
            {
                **pair,
                "correlation": _json_safe_metric(pair["correlation"]),
            }
            for pair in correlations
        ],
        "source": "sandbox_trades",
        "note": (
            "Stats use Analyzer/sandbox fills tagged with this strategy's hosted name "
            "(STRATEGY_NAME). Live broker positions are not strategy-tagged."
        ),
    }
