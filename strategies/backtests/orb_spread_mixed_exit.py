"""
ORB_Spread mixed / asymmetric exits (research only — Host untouched).

Post-pivot entries: LH/HL, OR 30–80, range-day skip at 10:15.
Primary P&L: adopted debit-spread clip  clip(spot_pts, 0, 50) − 7.5
Overlay:     BS Δnet on ATM vs OTM1 50pt vertical, σ=15% (theta-aware).
"""
from __future__ import annotations

from datetime import date as ddate, datetime, time as dtime, timedelta
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
CACHE = BASE / "orb_asym" / "nifty_5m_cache.pkl"
OUT = BASE / "orb_spread"
OUT.mkdir(exist_ok=True)

OR_MIN, OR_MAX = 30, 80
ENTRY_END = dtime(12, 0)
HARD = dtime(15, 15)
TARGET, STOP = 40.0, 25.0
WIDTH, COST = 50.0, 7.5
EXPIRY_SWITCH = ddate(2025, 9, 1)
SPLIT = ddate(2024, 1, 1)
LOT = 65
SIGMA = 0.15
PREV_MOVE = 0.42  # live quiet-day %; CE blocked when prior |net| > this


def n_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs(S: float, K: float, T: float, sigma: float, call: bool, r: float = 0.0) -> float:
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 1.0 / (365 * 24):
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    vol = sigma * sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol
    df = exp(-r * T)
    if call:
        return S * n_cdf(d1) - K * df * n_cdf(d2)
    return K * df * n_cdf(-d2) - S * n_cdf(-d1)


def atm50(S: float) -> float:
    return float(round(S / 50.0) * 50.0)


def years_left(ts: pd.Timestamp, expiry: ddate) -> float:
    end = pd.Timestamp(datetime.combine(expiry, dtime(15, 30)), tz=ts.tz)
    sec = max((end - ts).total_seconds(), 60.0)
    return sec / (365.0 * 24 * 3600)


def vert_net(S: float, T: float, long_k: float, short_k: float, call: bool) -> float:
    return bs(S, long_k, T, SIGMA, call) - bs(S, short_k, T, SIGMA, call)


def next_expiry(day: ddate, trading: set[ddate]) -> ddate | None:
    wd = 1 if day >= EXPIRY_SWITCH else 3
    d = day + timedelta(days=(wd - day.weekday()) % 7)
    for _ in range(10):
        if d in trading:
            return d
        d += timedelta(days=1)
    return None


def clip_pnl(spot_pts: float) -> float:
    return float(np.clip(spot_pts, 0.0, WIDTH) - COST)


def metrics(days, pnls, label=""):
    s = pd.Series(pnls, dtype=float)
    n = len(s)
    if n == 0:
        return {"label": label, "n": 0}
    gw = float(s[s > 0].sum())
    gl = abs(float(s[s <= 0].sum()))
    daily = pd.DataFrame({"d": days, "p": pnls}).groupby("d")["p"].sum()
    sh = float("nan")
    if len(daily) >= 2 and daily.std(ddof=1) != 0:
        sh = float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
    d = pd.Series(days)
    out = {
        "label": label,
        "n": n,
        "wr": round(float((s > 0).mean() * 100), 1),
        "avg": round(float(s.mean()), 2),
        "total": round(float(s.sum()), 1),
        "pf": round(float(gw / gl) if gl > 0 else float("inf"), 2),
        "sharpe": round(sh, 2),
        "inr": round(float(s.mean()) * LOT, 0),
    }
    for tag, mask in (("h1", d < SPLIT), ("h2", d >= SPLIT)):
        sub = s[mask]
        dd = d[mask]
        out[f"n_{tag}"] = int(len(sub))
        if len(sub) == 0:
            out[f"wr_{tag}"] = out[f"sh_{tag}"] = out[f"avg_{tag}"] = float("nan")
            continue
        out[f"wr_{tag}"] = round(float((sub > 0).mean() * 100), 1)
        out[f"avg_{tag}"] = round(float(sub.mean()), 2)
        dly = pd.DataFrame({"d": dd.tolist(), "p": sub.tolist()}).groupby("d")["p"].sum()
        out[f"sh_{tag}"] = (
            round(float(dly.mean() / dly.std(ddof=1) * np.sqrt(252)), 2)
            if len(dly) >= 2 and dly.std(ddof=1) != 0
            else float("nan")
        )
    return out


def fmt(m: dict) -> str:
    if m.get("n", 0) == 0:
        return f"{m.get('label', '')}  n=0"
    return (
        f"{m['label']:<28} n={m['n']:4d}  WR={m['wr']:5.1f}%  "
        f"avg={m['avg']:+6.2f}  Sh={m['sharpe']:6.2f}  PF={m['pf']:5.2f}  "
        f"H1 n={m['n_h1']} WR={m['wr_h1']} Sh={m['sh_h1']}  "
        f"H2 n={m['n_h2']} WR={m['wr_h2']} Sh={m['sh_h2']}"
    )


def load_bars() -> pd.DataFrame:
    df = pd.read_pickle(CACHE)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df = df.sort_index()
    df["_day"] = [ts.date() for ts in df.index]
    return df


def walk_exit(path: pd.DataFrame, entry_ts, entry_px: float, sign: int,
              expiry: ddate, hold_expiry: bool):
    after = path[path.index > entry_ts]
    if after.empty:
        return entry_ts, entry_px, 0.0, "NO_BARS"
    entry_day = entry_ts.date()
    for ts, bar in after.iterrows():
        d, t = ts.date(), ts.time()
        px = float(bar["close"])
        pts = sign * (px - entry_px)
        if pts >= TARGET:
            return ts, px, pts, "TARGET"
        if pts <= -STOP:
            return ts, px, pts, "STOP"
        if not hold_expiry and d == entry_day and t >= HARD:
            return ts, px, pts, "HARD_1515"
        if hold_expiry and d == expiry and t >= HARD:
            return ts, px, pts, "EXPIRY"
    last = after.iloc[-1]
    ts = after.index[-1]
    px = float(last["close"])
    return ts, px, sign * (px - entry_px), "FALLBACK"


def main() -> None:
    df = load_bars()
    print(f"Loaded {len(df):,} bars {df.index[0]} → {df.index[-1]}", flush=True)
    trading = set(df["_day"].unique())
    day_map = {d: g for d, g in df.groupby("_day", sort=False)}

    daily = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    daily["net_pct"] = (daily["close"] - daily["open"]).abs() / daily["open"] * 100
    daily["prev_net"] = daily["net_pct"].shift(1)
    prev_map = {
        idx.date(): float(v) if pd.notna(v) else np.nan
        for idx, v in daily["prev_net"].items()
    }

    entries = []
    for day, grp in day_map.items():
        orw = grp.between_time("09:15", "09:44")
        if len(orw) < 3:
            continue
        oh, ol = float(orw["high"].max()), float(orw["low"].min())
        rng = oh - ol
        if rng < OR_MIN or rng > OR_MAX:
            continue
        at = grp.between_time("10:15", "10:15")
        if not at.empty and ol < float(at["close"].iloc[0]) < oh:
            continue
        sig = grp.between_time("09:45", "12:00")
        if len(sig) < 3:
            continue
        c = sig["close"].values
        h = sig["high"].values
        lo = sig["low"].values
        idx = sig.index
        bear_done = bull_done = False
        for i in range(2, len(sig)):
            if idx[i].time() > ENTRY_END:
                break
            if not bear_done and h[i] < h[i - 1] < h[i - 2] and c[i] < oh:
                entries.append({
                    "day": day, "entry_time": idx[i], "direction": "SHORT",
                    "signal": "LowerHigh", "entry_price": float(c[i]),
                    "or_width": rng, "prev_net": prev_map.get(day, np.nan),
                })
                bear_done = True
            if not bull_done and lo[i] > lo[i - 1] > lo[i - 2] and c[i] > ol:
                entries.append({
                    "day": day, "entry_time": idx[i], "direction": "LONG",
                    "signal": "HigherLow", "entry_price": float(c[i]),
                    "or_width": rng, "prev_net": prev_map.get(day, np.nan),
                })
                bull_done = True

    entries_df = pd.DataFrame(entries)
    print(f"Entries (LH/HL, OR {OR_MIN}–{OR_MAX}): {len(entries_df)}", flush=True)

    rows = []
    for rec in entries_df.to_dict("records"):
        day = rec["day"]
        exp = next_expiry(day, trading)
        if exp is None:
            continue
        rec["expiry"] = exp
        start = rec["entry_time"]
        path = df[(df.index >= start) & (df["_day"] <= exp)]
        sign = 1 if rec["direction"] == "LONG" else -1
        call = rec["direction"] == "LONG"
        k_long = atm50(rec["entry_price"])
        k_short = k_long + 50.0 if call else k_long - 50.0
        t0 = years_left(start, exp)
        debit0 = vert_net(rec["entry_price"], t0, k_long, k_short, call)

        for name, hold in (("same_day", False), ("expiry", True)):
            ts, px, pts, reason = walk_exit(
                path, start, rec["entry_price"], sign, exp, hold
            )
            t1 = years_left(ts, exp)
            debit1 = vert_net(px, t1, k_long, k_short, call)
            rows.append({
                **rec,
                "policy": name,
                "exit_time": ts,
                "exit_price": px,
                "spot_pts": pts,
                "reason": reason,
                "clip_pnl": clip_pnl(pts),
                "bs_pnl": debit1 - debit0,
                "bs_debit0": debit0,
                "days_held": (ts.date() - day).days,
            })

    sim = pd.DataFrame(rows)
    same = sim[sim["policy"] == "same_day"].copy()
    expd = sim[sim["policy"] == "expiry"].copy()

    def mix(ce_hold: bool, pe_hold: bool) -> pd.DataFrame:
        ce = expd if ce_hold else same
        pe = expd if pe_hold else same
        a = ce[ce["direction"] == "LONG"]
        b = pe[pe["direction"] == "SHORT"]
        return pd.concat([a, b], ignore_index=True)

    mix_ce = mix(True, False)
    books = {
        "baseline_same_day": same,
        "mix_CE_exp_PE_1515": mix_ce,
        "both_hold_expiry": expd,
        "control_PE_exp_CE_1515": mix(False, True),
        "baseline_CE": same[same["direction"] == "LONG"],
        "baseline_PE": same[same["direction"] == "SHORT"],
        "mix_CE_only": mix_ce[mix_ce["direction"] == "LONG"],
        "mix_PE_only": mix_ce[mix_ce["direction"] == "SHORT"],
        "expiry_CE": expd[expd["direction"] == "LONG"],
        "expiry_PE": expd[expd["direction"] == "SHORT"],
    }

    def live_slice(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        keep = []
        for _, r in frame.iterrows():
            prev = r["prev_net"]
            if pd.isna(prev) or prev <= PREV_MOVE:
                keep.append(True)
            else:
                keep.append(r["direction"] == "SHORT")
        return frame.loc[keep].copy()

    print("\nMODEL: debit-spread CLIP  clip(spot,0,50)-7.5   (adopted; no theta)")
    clip_rows = []
    for name, frame in books.items():
        m = metrics(frame["day"].tolist(), frame["clip_pnl"].tolist(), name)
        print(fmt(m), flush=True)
        clip_rows.append(m)

    print("\nOVERLAY: BS Δnet ATM–OTM1 50pt vertical  σ=15%  (theta-aware)")
    bs_rows = []
    for name, frame in books.items():
        m = metrics(frame["day"].tolist(), frame["bs_pnl"].tolist(), name)
        print(fmt(m), flush=True)
        bs_rows.append(m)

    print("\nLIVE-ISH SLICE (prev-day |net|≤0.42% both; else PE only) — CLIP")
    live_keys = (
        "baseline_same_day", "mix_CE_exp_PE_1515", "both_hold_expiry",
        "control_PE_exp_CE_1515", "baseline_CE", "baseline_PE",
    )
    live_books = {k: live_slice(books[k]) for k in live_keys}
    live_clip = []
    for name, frame in live_books.items():
        m = metrics(frame["day"].tolist(), frame["clip_pnl"].tolist(), "live_" + name)
        print(fmt(m), flush=True)
        live_clip.append(m)

    print("\nLIVE-ISH SLICE — BS Δnet")
    live_bs = []
    for name, frame in live_books.items():
        m = metrics(frame["day"].tolist(), frame["bs_pnl"].tolist(), "live_" + name)
        print(fmt(m), flush=True)
        live_bs.append(m)

    same.to_csv(OUT / "orb_spread_mixed_exit_baseline_trades.csv", index=False)
    mix_ce.to_csv(OUT / "orb_spread_mixed_exit_mix_trades.csv", index=False)
    expd.to_csv(OUT / "orb_spread_mixed_exit_expiry_trades.csv", index=False)
    pd.DataFrame(clip_rows).to_csv(OUT / "orb_spread_mixed_exit_clip_summary.csv", index=False)
    pd.DataFrame(bs_rows).to_csv(OUT / "orb_spread_mixed_exit_bs_summary.csv", index=False)
    pd.DataFrame(live_clip).to_csv(OUT / "orb_spread_mixed_exit_live_clip_summary.csv", index=False)
    pd.DataFrame(live_bs).to_csv(OUT / "orb_spread_mixed_exit_live_bs_summary.csv", index=False)

    print("\nExit reasons baseline:", same["reason"].value_counts().to_dict())
    print("Exit reasons both-expiry:", expd["reason"].value_counts().to_dict())
    print("Exit reasons mix:", mix_ce["reason"].value_counts().to_dict())
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
