"""ORB_Asym v3 — LH/HL direction, 1-2-1 BWB overlay, BS Δnet MTM.

Karan screenshot (spot 24272):
  CE: BUY 24250 / SELL 24400×2 / BUY 24500
  PE: BUY 24150 / SELL 24000×2 / BUY 23900
  long_ce = floor50(S); long_pe = floor50(S)-100; body = long±150; far = long±250.

Strikes locked at entry. Score option MTM, not intrinsic TARGET.
v2 50/150 fly and Defensive OR-Hold are out. Debit spread is a control only.

Gates: Sharpe > 2, WR > 50%, n >= 80.
"""
from datetime import date as ddate, time as dtime, timedelta
from itertools import product
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "orb_asym"
CACHE = OUT / "nifty_5m_cache.pkl"
OR_MIN, OR_MAX = 30, 80
ENTRY_END = dtime(12, 0)
HOLD = 3
QUIET = 0.0042
SIGMA = 0.15
BODY, FAR, PE_OFFSET = 150, 250, 100
EXPIRY_SWITCH = ddate(2025, 9, 1)
SPLIT = ddate(2024, 1, 1)
SHARPE_GATE, WR_GATE, N_MIN = 2.0, 50.0, 80
INDEX = "NIFTY"  # NIFTY: Thu→Tue on EXPIRY_SWITCH. SENSEX: Tue→Thu (inverse).
LOT = 65
SENSEX_CACHE = OUT / "sensex_5m_cache.pkl"
SENSEX_LOT = 10


def n_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs(S, K, T, sigma, call, r=0.0):
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


def floor50(px):
    return float((int(px) // 50) * 50)


def fly_strikes(S, put):
    if INDEX == "SENSEX":
        step, pe, body, far = 100, 200, 300, 500
        long_k = float((int(S) // step) * step) - (pe if put else 0)
        sign = -1.0 if put else 1.0
        return long_k, long_k + sign * body, long_k + sign * far
    long_k = floor50(S) - PE_OFFSET if put else floor50(S)
    sign = -1.0 if put else 1.0
    return long_k, long_k + sign * BODY, long_k + sign * FAR


def fly_net(S, T, long_k, body, far, call):
    return bs(S, long_k, T, SIGMA, call) + bs(S, far, T, SIGMA, call) - 2.0 * bs(S, body, T, SIGMA, call)


def vert_strikes(S, put):
    long_k = floor50(S)
    otm = long_k - 50 if put else long_k + 50
    return long_k, otm


def vert_net(S, T, long_k, otm, call):
    return bs(S, long_k, T, SIGMA, call) - bs(S, otm, T, SIGMA, call)


def years(dte, hhmm):
    h, m = hhmm
    left = max((15 * 60 + 15) - (h * 60 + m), 5) / (6.25 * 60)
    return max((dte + left) / 365.0, 1.0 / (365 * 24))


def sharpe_daily(days, pnls):
    daily = pd.DataFrame({"d": days, "p": pnls}).groupby("d")["p"].sum()
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        return float("nan")
    r = daily / 50.0
    return float(r.mean() / r.std(ddof=1) * np.sqrt(252))


def metrics(days, pnls):
    s = pd.Series(pnls, dtype=float)
    n = len(s)
    empty = dict(n=0, wr=float("nan"), pf=float("nan"), total=0.0, sharpe=float("nan"),
                 avg=float("nan"), n1=0, wr1=float("nan"), sh1=float("nan"),
                 n2=0, wr2=float("nan"), sh2=float("nan"))
    if n == 0:
        return empty
    gw, gl = float(s[s > 0].sum()), abs(float(s[s <= 0].sum()))
    d = pd.Series(days)
    m = dict(
        n=n,
        wr=round(float((s > 0).mean() * 100), 1),
        pf=round(float(gw / gl) if gl > 0 else float("inf"), 2),
        total=round(float(s.sum()), 1),
        avg=round(float(s.mean()), 2),
        sharpe=round(sharpe_daily(days, pnls), 2),
    )
    for tag, mask in (("1", d < SPLIT), ("2", d >= SPLIT)):
        sub_p, sub_d = s[mask].tolist(), d[mask].tolist()
        m[f"n{tag}"] = len(sub_p)
        if sub_p:
            m[f"wr{tag}"] = round(float((np.array(sub_p) > 0).mean() * 100), 1)
            m[f"sh{tag}"] = round(sharpe_daily(sub_d, sub_p), 2)
        else:
            m[f"wr{tag}"] = m[f"sh{tag}"] = float("nan")
    return m


def _check_strikes():
    assert floor50(24272) == 24250
    assert fly_strikes(24272, False) == (24250.0, 24400.0, 24500.0)
    assert fly_strikes(24272, True) == (24150.0, 24000.0, 23900.0)


_check_strikes()

df = trading = day_map = prev_map = prev_info = None


def load(cache_path=None):
    global df, trading, day_map, prev_map, prev_info
    path = Path(cache_path) if cache_path is not None else CACHE
    print("Loading cache", path, flush=True)
    df = pd.read_pickle(path)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("Asia/Kolkata")
    else:
        df.index = df.index.tz_convert("Asia/Kolkata")
    df["_day"] = [ts.date() for ts in df.index]
    trading = set(df["_day"].unique())
    day_map = {d: g for d, g in df.groupby("_day", sort=False)}
    daily = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    daily["prev_net"] = daily["close"].pct_change().shift(1)
    daily["p_open"] = daily["open"].shift(1)
    daily["p_close"] = daily["close"].shift(1)
    daily["p_high"] = daily["high"].shift(1)
    daily["p_low"] = daily["low"].shift(1)
    prev_map = {d.date(): v for d, v in daily["prev_net"].items()}
    prev_info = {}
    for ts, row in daily.iterrows():
        prev_info[ts.date()] = dict(
            net=row.prev_net, p_open=row.p_open, p_close=row.p_close,
            p_high=row.p_high, p_low=row.p_low, day_open=float(row.open),
        )


def expiry_date(day):
    # From 2025-09-01: Nifty weekly Tuesday, Sensex weekly Thursday.
    # Before that: Nifty Thursday, Sensex Tuesday.
    if INDEX == "SENSEX":
        wd = 3 if day >= EXPIRY_SWITCH else 1
    else:
        wd = 1 if day >= EXPIRY_SWITCH else 3
    cand = day + timedelta(days=(wd - day.weekday()) % 7)
    end = day + timedelta(days=10)
    while cand not in trading:
        cand += timedelta(days=1)
        if cand > end:
            return None
    return cand


def expiry_dte(day):
    cand = expiry_date(day)
    return 3 if cand is None else (cand - day).days


def close_1515(day):
    g = day_map.get(day)
    if g is None or g.empty:
        return None
    at = g.between_time("15:15", "15:15")
    if not at.empty:
        return float(at["close"].iloc[0])
    return float(g["close"].iloc[-1])


def generate(or_end="09:44", quiet=False, dir_aware=False, skip_range=True,
             hold=HOLD, entry_end=ENTRY_END, or_min=OR_MIN, or_max=OR_MAX, require_mid=False,
             quiet_thr=QUIET):
    rec = []
    for day, grp in day_map.items():
        orw = grp.between_time("09:15", or_end)
        if len(orw) < 3:
            continue
        oh, ol = float(orw["high"].max()), float(orw["low"].min())
        if not (or_min <= oh - ol <= or_max):
            continue
        if skip_range:
            at = grp.between_time("10:15", "10:15")
            if not at.empty:
                c = float(at["close"].iloc[0])
                if ol < c < oh:
                    continue
        pn = prev_map.get(day)
        if quiet and (pn is None or abs(pn) > quiet_thr):
            continue
        mid = (oh + ol) / 2.0
        sig = grp.between_time("09:45", "12:00")
        if len(sig) < hold:
            continue
        h, lo, c, ix = sig["high"].values, sig["low"].values, sig["close"].values, sig.index
        bear = bull = False
        for i in range(hold - 1, len(sig)):
            if ix[i].time() > entry_end:
                break
            lh = all(h[i - k] < h[i - k - 1] for k in range(hold - 1)) and c[i] < oh
            hl = all(lo[i - k] > lo[i - k - 1] for k in range(hold - 1)) and c[i] > ol
            if require_mid:
                lh = lh and c[i] <= mid
                hl = hl and c[i] >= mid
            if (not bear) and lh:
                rec.append(dict(day=day, entry_time=ix[i], direction="SHORT",
                                entry_price=float(c[i]), signal="LowerHigh", dte=expiry_dte(day)))
                bear = True
            allow_bull = not (dir_aware and pn is not None and abs(pn) > QUIET)
            if allow_bull and (not bull) and hl:
                rec.append(dict(day=day, entry_time=ix[i], direction="LONG",
                                entry_price=float(c[i]), signal="HigherLow", dte=expiry_dte(day)))
                bull = True
    return pd.DataFrame(rec)


def bars_np(day, et, hard):
    g = day_map[day]
    g = g[(g.index > et) & (g.index.time <= hard)]
    if g.empty:
        return None
    t = g.index
    return dict(
        close=g["close"].to_numpy(float),
        high=g["high"].to_numpy(float),
        low=g["low"].to_numpy(float),
        hh=[(ts.hour, ts.minute) for ts in t],
        tm=[ts.time() for ts in t],
    )


def path_trade(row, hard):
    ep, put, dte = float(row.entry_price), row.direction == "SHORT", int(row.dte)
    b = bars_np(row.day, row.entry_time, hard)
    if b is None:
        return None
    call = not put
    fl, fb, ff = fly_strikes(ep, put)
    vl, vo = vert_strikes(ep, put)
    T0 = years(dte, (row.entry_time.hour, row.entry_time.minute))
    n0f = fly_net(ep, T0, fl, fb, ff, call)
    n0v = vert_net(ep, T0, vl, vo, call)
    n = len(b["close"])
    mtm_f = np.empty(n)
    mtm_v = np.empty(n)
    fav = np.empty(n)
    adv = np.empty(n)
    for i in range(n):
        T = years(dte, b["hh"][i])
        S = b["close"][i]
        mtm_f[i] = fly_net(S, T, fl, fb, ff, call) - n0f
        mtm_v[i] = vert_net(S, T, vl, vo, call) - n0v
        if put:
            fav[i] = ep - b["low"][i]
            adv[i] = b["high"][i] - ep
        else:
            fav[i] = b["high"][i] - ep
            adv[i] = ep - b["low"][i]
    return dict(mtm_f=mtm_f, mtm_v=mtm_v, fav=fav, adv=adv, close=b["close"],
                tm=b["tm"], n0f=n0f, debit=n0f > 0.0, ep=ep, put=put, dte=dte)


def apply_mtm(p, arr, tgt, stp):
    m = p[arr]
    if tgt is None:
        return float(m[-1]), "HARD"
    for i, x in enumerate(m):
        if x >= tgt:
            return float(x), "MTM_TGT"
        if x <= -stp:
            return float(x), "MTM_STP"
    return float(m[-1]), "HARD"


def apply_spot(p, arr, tgt, stp):
    m, fav, adv = p[arr], p["fav"], p["adv"]
    for i in range(len(m)):
        if fav[i] >= tgt:
            return float(m[i]), "SPOT_TGT"
        if adv[i] >= stp:
            return float(m[i]), "SPOT_STP"
    return float(m[-1]), "HARD"


def apply_trail(p, arr, arm, give):
    m = p[arr]
    peak = -1e9
    armed = False
    for x in m:
        if x > peak:
            peak = x
        if (not armed) and peak >= arm:
            armed = True
        if armed and x <= peak - give:
            return float(x), "TRAIL"
        if x <= -arm:
            return float(x), "MTM_STP"
    return float(m[-1]), "HARD"


def apply_spread(p, tgt, stp, width=50, cost=7.5):
    for i in range(len(p["fav"])):
        if p["fav"][i] >= tgt:
            return min(tgt, width) - cost, "TARGET"
        if p["adv"][i] >= stp:
            return -cost, "STOP"
    dlt = (p["ep"] - float(p["close"][-1])) if p["put"] else (float(p["close"][-1]) - p["ep"])
    return float(np.clip(dlt, 0, width) - cost), "HARD"


def apply_frac(p, tgt_f, stp_f):
    n0 = abs(p["n0f"]) if abs(p["n0f"]) > 1e-6 else 1e-6
    return apply_mtm(p, "mtm_f", tgt_f * n0, stp_f * n0)


def apply_side_mtm(p, tgt_ce, stp_ce, tgt_pe, stp_pe):
    if p["put"]:
        return apply_mtm(p, "mtm_f", tgt_pe, stp_pe)
    return apply_mtm(p, "mtm_f", tgt_ce, stp_ce)


def score_one(p, kind, kw):
    if kind == "fly_hold":
        return float(p["mtm_f"][-1]), "HARD"
    if kind == "fly_mtm":
        return apply_mtm(p, "mtm_f", kw["tgt"], kw["stp"])
    if kind == "fly_spot":
        return apply_spot(p, "mtm_f", kw["tgt"], kw["stp"])
    if kind == "fly_trail":
        return apply_trail(p, "mtm_f", kw["arm"], kw["give"])
    if kind == "fly_frac":
        return apply_frac(p, kw["tgt"], kw["stp"])
    if kind == "fly_side":
        return apply_side_mtm(p, kw["tgt_ce"], kw["stp_ce"], kw["tgt_pe"], kw["stp_pe"])
    if kind == "spread":
        return apply_spread(p, kw["tgt"], kw["stp"])
    if kind == "vert_mtm":
        return apply_mtm(p, "mtm_v", kw["tgt"], kw["stp"])
    return apply_spot(p, "mtm_v", kw["tgt"], kw["stp"])


def main():
    load()
    print("Signals...", flush=True)
    specs = [
        ("lhhl", dict()),
        ("quiet", dict(quiet=True)),
        ("diraware", dict(dir_aware=True)),
        ("or29_quiet", dict(or_end="09:29", quiet=True)),
        ("quiet_h2", dict(quiet=True, hold=2)),
        ("quiet_e1030", dict(quiet=True, entry_end=dtime(10, 30))),
        ("quiet_e1100", dict(quiet=True, entry_end=dtime(11, 0))),
        ("quiet_mid", dict(quiet=True, require_mid=True)),
        ("quiet_or3055", dict(quiet=True, or_min=30, or_max=55)),
        ("quiet_or4070", dict(quiet=True, or_min=40, or_max=70)),
        ("quiet_norange", dict(quiet=True, skip_range=False)),
        ("quiet_h2_e1030", dict(quiet=True, hold=2, entry_end=dtime(10, 30))),
        ("lhhl_h2", dict(hold=2)),
        ("or29_quiet_h2", dict(or_end="09:29", quiet=True, hold=2)),
    ]
    sets = {}
    for name, kw in specs:
        sets[name] = generate(**kw)
        print(f"  {name}: {len(sets[name])}", flush=True)

    fly_cfgs = [("fly_hold", "fly_hold", {})]
    for mt, ms in product((6, 8, 10, 12, 14, 16, 20, 25, 30), (6, 8, 10, 12, 15, 20)):
        fly_cfgs.append((f"fly_mtm_{mt}_{ms}", "fly_mtm", dict(tgt=mt, stp=ms)))
    for tf, sf in product((0.4, 0.6, 0.8, 1.0), (0.5, 0.8, 1.0, 1.2)):
        fly_cfgs.append((f"fly_frac_{tf}_{sf}", "fly_frac", dict(tgt=tf, stp=sf)))
    for tgt, stp in ((30, 20), (40, 25), (50, 25), (80, 35)):
        fly_cfgs.append((f"fly_spot_{tgt}_{stp}", "fly_spot", dict(tgt=tgt, stp=stp)))
    for arm, give in product((6, 8, 12, 16), (3, 5, 8)):
        fly_cfgs.append((f"fly_trail_{arm}_{give}", "fly_trail", dict(arm=arm, give=give)))
    for tgt_ce, stp_ce, tgt_pe, stp_pe in (
        (12, 8, 6, 8), (16, 10, 8, 8), (20, 10, 8, 10),
        (10, 8, 6, 6), (14, 10, 8, 10), (25, 12, 10, 10),
    ):
        fly_cfgs.append((f"fly_side_{tgt_ce}_{stp_ce}_{tgt_pe}_{stp_pe}", "fly_side",
                         dict(tgt_ce=tgt_ce, stp_ce=stp_ce, tgt_pe=tgt_pe, stp_pe=stp_pe)))

    ctrl = [
        ("spread_50_15_t40", "spread", dict(tgt=40, stp=25)),
        ("vert_spot_40_25", "vert_spot", dict(tgt=40, stp=25)),
    ]

    rows, best = [], None

    def consider(met, extra):
        nonlocal best
        met.update(extra)
        met["pass"] = bool(met["sharpe"] > SHARPE_GATE and met["wr"] > WR_GATE and met["n"] >= N_MIN)
        rows.append(met)
        if met["pass"] and str(met.get("vehicle", "")).startswith("fly"):
            key = (met["sharpe"], met["wr"], met["pf"], met["n"])
            if best is None or key > best[0]:
                best = (key, met)

    def run_grid(sig_name, paths, days0, htag, side, first_only, dte_min, cfgs):
        for name, kind, kw in cfgs:
            pnls, reasons, days, seen = [], [], [], set()
            for p, day in zip(paths, days0):
                if side == "CE" and p["put"]:
                    continue
                if side == "PE" and not p["put"]:
                    continue
                if first_only:
                    if day in seen:
                        continue
                    seen.add(day)
                if p["dte"] < dte_min:
                    continue
                pnl, reason = score_one(p, kind, kw)
                pnls.append(pnl)
                reasons.append(reason)
                days.append(day)
            if len(pnls) < N_MIN:
                continue
            met = metrics(days, pnls)
            consider(met, dict(
                signals=sig_name, vehicle=name, hard=htag, side=side,
                first_only=first_only, dte_min=dte_min,
                n_tgt=reasons.count("TARGET") + reasons.count("SPOT_TGT") + reasons.count("MTM_TGT"),
                n_stp=reasons.count("STOP") + reasons.count("SPOT_STP") + reasons.count("MTM_STP"),
                n_hard=reasons.count("HARD"), n_trail=reasons.count("TRAIL"),
            ))

    hards = (dtime(15, 15), dtime(14, 30))
    for sig_name, entries in sets.items():
        if entries.empty:
            continue
        for hard in hards:
            htag = "1515" if hard == dtime(15, 15) else "1430"
            print(f"Paths {sig_name} hard={htag}...", flush=True)
            paths, days0 = [], []
            n0s, n0_ce, n0_pe = [], [], []
            for _, row in entries.iterrows():
                p = path_trade(row, hard)
                if p is None:
                    continue
                paths.append(p)
                days0.append(row.day)
                n0s.append(p["n0f"])
                (n0_pe if p["put"] else n0_ce).append(p["n0f"])
            if len(paths) < N_MIN:
                print(f"  skip n={len(paths)}", flush=True)
                continue
            print(f"  n={len(paths)} debit_mean={np.mean(n0s):.1f} CE={np.mean(n0_ce):.1f}(n={len(n0_ce)}) PE={np.mean(n0_pe):.1f}(n={len(n0_pe)})", flush=True)
            run_grid(sig_name, paths, days0, htag, "BOTH", False, 0, fly_cfgs + ctrl)
            if sig_name in ("quiet", "quiet_h2", "quiet_e1030", "or29_quiet"):
                run_grid(sig_name, paths, days0, htag, "CE", False, 0, fly_cfgs)
                run_grid(sig_name, paths, days0, htag, "PE", False, 0, fly_cfgs)
            if sig_name == "quiet" and hard == dtime(15, 15):
                run_grid(sig_name, paths, days0, htag, "BOTH", True, 0, fly_cfgs)
                run_grid(sig_name, paths, days0, htag, "BOTH", False, 1, fly_cfgs)

    res = pd.DataFrame(rows).sort_values(["pass", "sharpe", "wr"], ascending=[False, False, False])
    res.to_csv(OUT / "orb_asym_v3_search.csv", index=False)
    fly_only = res[res["vehicle"].astype(str).str.startswith("fly")]
    print("\nTOP 20 fly", flush=True)
    print(fly_only.head(20).to_string(index=False), flush=True)
    print(f"\nPassing all: {int(res['pass'].sum())}  fly: {int(fly_only['pass'].sum())}", flush=True)
    if best:
        print("BEST FLY", best[1], flush=True)
    else:
        print("No fly passer", flush=True)
        near = fly_only[(fly_only["wr"] > 50) & (fly_only["sharpe"] > 1.5)].head(10)
        print("Near-miss WR>50 & Sh>1.5\n", near.to_string(index=False) if not near.empty else "(none)", flush=True)
    print("wrote", OUT / "orb_asym_v3_search.csv", flush=True)


def wf_ok(met):
    try:
        return (
            met["n"] >= N_MIN and met["wr"] > WR_GATE and met["sharpe"] > SHARPE_GATE
            and met["n1"] >= 40 and met["n2"] >= 40
            and met["wr1"] > WR_GATE and met["wr2"] > WR_GATE
            and met["sh1"] > SHARPE_GATE and met["sh2"] > SHARPE_GATE
            and met["avg"] > 0
        )
    except (TypeError, KeyError):
        return False


def apply_trail_stp(p, arr, arm, give, stp):
    m = p[arr]
    peak, armed = -1e9, False
    for x in m:
        if x > peak:
            peak = x
        if (not armed) and peak >= arm:
            armed = True
        if armed and x <= peak - give:
            return float(x), "TRAIL"
        if x <= -stp:
            return float(x), "MTM_STP"
    return float(m[-1]), "HARD"


def friction_main():
    """Score fly grids at 0/1/2 pt flat cost. Gates applied after friction, including H1/H2."""
    load()
    print("Signals (friction search)...", flush=True)
    specs = [
        ("quiet", dict(quiet=True)),
        ("quiet_mid", dict(quiet=True, require_mid=True)),
        ("quiet_or4070", dict(quiet=True, or_min=40, or_max=70)),
        ("quiet_or4070_mid", dict(quiet=True, or_min=40, or_max=70, require_mid=True)),
        ("quiet_or4070_e1030", dict(quiet=True, or_min=40, or_max=70, entry_end=dtime(10, 30))),
        ("quiet_or4070_e1100", dict(quiet=True, or_min=40, or_max=70, entry_end=dtime(11, 0))),
        ("quiet_or4070_h2", dict(quiet=True, or_min=40, or_max=70, hold=2)),
        ("quiet_mid_e1100", dict(quiet=True, require_mid=True, entry_end=dtime(11, 0))),
        ("quiet_or4575", dict(quiet=True, or_min=45, or_max=75)),
        ("quiet_or3565", dict(quiet=True, or_min=35, or_max=65)),
        ("or29_quiet_or4070", dict(or_end="09:29", quiet=True, or_min=40, or_max=70)),
        ("quiet_or4070_h2_e1100", dict(quiet=True, or_min=40, or_max=70, hold=2, entry_end=dtime(11, 0))),
        ("quiet_e1100", dict(quiet=True, entry_end=dtime(11, 0))),
        ("lhhl_or4070", dict(or_min=40, or_max=70)),
    ]
    sets = {}
    for name, kw in specs:
        sets[name] = generate(**kw)
        print(f"  {name}: {len(sets[name])}", flush=True)

    fly_cfgs = []
    for mt, ms in product((12, 16, 20, 25, 30, 40, 50), (10, 12, 15, 20, 25)):
        fly_cfgs.append((f"fly_mtm_{mt}_{ms}", "fly_mtm", dict(tgt=mt, stp=ms)))
    for tf, sf in product((0.6, 0.8, 1.0, 1.2, 1.5), (0.4, 0.5, 0.6, 0.8)):
        fly_cfgs.append((f"fly_frac_{tf}_{sf}", "fly_frac", dict(tgt=tf, stp=sf)))
    for tgt, stp in ((25, 15), (30, 15), (30, 20), (40, 20), (40, 25), (50, 20), (50, 25), (60, 25)):
        fly_cfgs.append((f"fly_spot_{tgt}_{stp}", "fly_spot", dict(tgt=tgt, stp=stp)))
    for arm, give in product((8, 12, 16, 20, 25, 30), (5, 8, 10, 12)):
        fly_cfgs.append((f"fly_trail_{arm}_{give}", "fly_trail", dict(arm=arm, give=give)))
    for arm, give, stp in ((12, 6, 12), (16, 8, 12), (16, 8, 15), (20, 8, 12), (20, 10, 15),
                           (25, 8, 15), (25, 10, 15), (25, 10, 20), (30, 10, 15), (30, 12, 20)):
        fly_cfgs.append((f"fly_trstp_{arm}_{give}_{stp}", "fly_trstp", dict(arm=arm, give=give, stp=stp)))
    for tgt_ce, stp_ce, tgt_pe, stp_pe in (
        (16, 10, 10, 8), (20, 12, 12, 10), (25, 12, 12, 10), (25, 15, 16, 12),
        (30, 12, 12, 10), (30, 15, 16, 12), (30, 15, 20, 12), (40, 15, 16, 12),
        (40, 20, 20, 15), (20, 10, 16, 10), (25, 10, 12, 8), (50, 20, 20, 12),
    ):
        fly_cfgs.append((f"fly_side_{tgt_ce}_{stp_ce}_{tgt_pe}_{stp_pe}", "fly_side",
                         dict(tgt_ce=tgt_ce, stp_ce=stp_ce, tgt_pe=tgt_pe, stp_pe=stp_pe)))

    def score_ex(p, kind, kw):
        if kind == "fly_trstp":
            return apply_trail_stp(p, "mtm_f", kw["arm"], kw["give"], kw["stp"])
        return score_one(p, kind, kw)

    rows, best1, best2 = [], None, None
    debit_floors = (0.0, 12.0, 18.0, 25.0)
    pe_floors = (0.0, 15.0)
    hards = (dtime(15, 15), dtime(14, 30), dtime(13, 15))

    def pack(days, pnls, extra):
        nonlocal best1, best2
        raw = metrics(days, pnls)
        if raw["avg"] <= 0.93:
            return
        rec = dict(raw)
        rec.update(extra)
        rec["avg0"] = raw["avg"]
        for cost in (0, 1, 2):
            m = metrics(days, [x - cost for x in pnls]) if cost else raw
            rec[f"wr_c{cost}"] = m["wr"]
            rec[f"sh_c{cost}"] = m["sharpe"]
            rec[f"pf_c{cost}"] = m["pf"]
            rec[f"avg_c{cost}"] = m["avg"]
            rec[f"wr1_c{cost}"] = m["wr1"]
            rec[f"sh1_c{cost}"] = m["sh1"]
            rec[f"wr2_c{cost}"] = m["wr2"]
            rec[f"sh2_c{cost}"] = m["sh2"]
            rec[f"pass_c{cost}"] = wf_ok(m)
        rec["pass1"] = rec["pass_c1"]
        rec["pass2"] = rec["pass_c2"]
        rows.append(rec)
        if rec["pass_c1"]:
            key = (rec["pass_c2"], rec["sh_c2"] if rec["pass_c2"] else rec["sh_c1"], rec["avg_c1"], rec["n"])
            if best1 is None or key > best1[0]:
                best1 = (key, rec)
        if rec["pass_c2"]:
            key = (rec["sh_c2"], rec["avg_c2"], rec["wr_c2"], rec["n"])
            if best2 is None or key > best2[0]:
                best2 = (key, rec)

    for sig_name, entries in sets.items():
        if entries.empty or len(entries) < N_MIN:
            continue
        for hard in hards:
            htag = {dtime(15, 15): "1515", dtime(14, 30): "1430", dtime(13, 15): "1315"}[hard]
            print(f"Paths {sig_name} {htag}...", flush=True)
            paths, days0 = [], []
            for _, row in entries.iterrows():
                p = path_trade(row, hard)
                if p is None:
                    continue
                paths.append(p)
                days0.append(row.day)
            if len(paths) < N_MIN:
                print(f"  skip n={len(paths)}", flush=True)
                continue
            print(f"  n={len(paths)}", flush=True)
            n0 = np.array([p["n0f"] for p in paths])
            is_put = np.array([p["put"] for p in paths])
            for name, kind, kw in fly_cfgs:
                raw_pnl = np.array([score_ex(p, kind, kw)[0] for p in paths], float)
                for dmin in debit_floors:
                    for pef in pe_floors:
                        mask = (n0 >= dmin) & ((~is_put) | (n0 >= pef))
                        if int(mask.sum()) < N_MIN:
                            continue
                        ds = [d for d, keep in zip(days0, mask) if keep]
                        ps = raw_pnl[mask].tolist()
                        pack(ds, ps, dict(
                            signals=sig_name, vehicle=name, hard=htag,
                            debit_min=dmin, pe_floor=pef, n_raw=len(paths),
                        ))

    if not rows:
        print("No rows with avg0 > 0.93", flush=True)
        return
    res = pd.DataFrame(rows)
    res = res.sort_values(["pass2", "pass1", "sh_c1", "avg_c1"], ascending=[False, False, False, False])
    outp = OUT / "orb_asym_v3_friction.csv"
    res.to_csv(outp, index=False)
    p1 = res[res["pass1"] == True]
    p2 = res[res["pass2"] == True]
    cols = ("n wr pf avg sharpe wr_c1 sh_c1 avg_c1 wr1_c1 sh1_c1 wr2_c1 sh2_c1 "
            "wr_c2 sh_c2 avg_c2 signals vehicle hard debit_min pe_floor").split()
    print(f"\nrows {len(res)}  pass@1pt {len(p1)}  pass@2pt {len(p2)}", flush=True)
    print("\nTOP 15 @1pt WF", flush=True)
    show = p1 if len(p1) else res.head(15)
    print(show[cols].head(15).to_string(index=False), flush=True)
    if len(p2):
        print("\nTOP 10 @2pt WF", flush=True)
        print(p2[cols].head(10).to_string(index=False), flush=True)
    if best1:
        print("BEST 1pt", {k: best1[1][k] for k in cols if k in best1[1]}, flush=True)
    else:
        print("No 1pt WF passer", flush=True)
        near = res[(res["wr_c1"] > 48) & (res["sh_c1"] > 1.5)].head(12)
        print("near 1pt\n", near[cols].to_string(index=False) if len(near) else "(none)", flush=True)
    if best2:
        print("BEST 2pt", {k: best2[1][k] for k in cols if k in best2[1]}, flush=True)
    print("wrote", outp, flush=True)


def hunt2_main():
    """Second pass: new signal knobs aimed at 2pt WF while keeping PE."""
    load()
    print("Signals (2pt hunt)...", flush=True)
    specs = []
    for qthr, omin, omax, hold, eend, mid in product(
        (0.003, 0.0042, 0.005),
        (40,),
        (70, 75),
        (3, 4),
        (dtime(10, 15), dtime(11, 0), dtime(12, 0)),
        (False, True),
    ):
        name = f"q{int(qthr*10000)}_or{omin}{omax}_h{hold}_e{eend.hour:02d}{eend.minute:02d}{'_mid' if mid else ''}"
        specs.append((name, dict(quiet=True, quiet_thr=qthr, or_min=omin, or_max=omax,
                                 hold=hold, entry_end=eend, require_mid=mid)))
    for hold, eend in product((3, 4), (dtime(11, 0), dtime(12, 0))):
        specs.append((f"or29_q42_h{hold}_e{eend.hour:02d}{eend.minute:02d}",
                      dict(or_end="09:29", quiet=True, or_min=40, or_max=70, hold=hold, entry_end=eend)))
    sets = {}
    for name, kw in specs:
        g = generate(**kw)
        if len(g) >= N_MIN:
            sets[name] = g
            print(f"  {name}: {len(g)}", flush=True)
        else:
            print(f"  {name}: {len(g)} skip", flush=True)

    cfgs = []
    for arm, give in product((5, 6, 7, 8, 9, 10, 12), (3, 4, 5, 6)):
        cfgs.append((f"trail_{arm}_{give}", "fly_trail", dict(arm=arm, give=give)))
    for mt, ms in product((12, 16, 20, 25, 30), (8, 10, 12)):
        cfgs.append((f"mtm_{mt}_{ms}", "fly_mtm", dict(tgt=mt, stp=ms)))
    for t in ((18, 10, 10, 8), (22, 11, 12, 9), (25, 12, 12, 10), (25, 10, 12, 8),
              (28, 12, 14, 10), (20, 10, 14, 8)):
        cfgs.append((f"side_{t[0]}_{t[1]}_{t[2]}_{t[3]}", "fly_side",
                     dict(tgt_ce=t[0], stp_ce=t[1], tgt_pe=t[2], stp_pe=t[3])))

    rows, best1, best2 = [], None, None

    def pack(days, pnls, extra, nce, npe):
        nonlocal best1, best2
        if npe < 20 or nce < 20:
            return
        raw = metrics(days, pnls)
        if raw["avg"] <= 0.93:
            return
        rec = dict(raw)
        rec.update(extra)
        rec["nce"] = nce
        rec["npe"] = npe
        rec["avg0"] = raw["avg"]
        for cost in (0, 1, 2):
            m = metrics(days, [x - cost for x in pnls]) if cost else raw
            rec[f"wr_c{cost}"] = m["wr"]
            rec[f"sh_c{cost}"] = m["sharpe"]
            rec[f"pf_c{cost}"] = m["pf"]
            rec[f"avg_c{cost}"] = m["avg"]
            rec[f"wr1_c{cost}"] = m["wr1"]
            rec[f"sh1_c{cost}"] = m["sh1"]
            rec[f"wr2_c{cost}"] = m["wr2"]
            rec[f"sh2_c{cost}"] = m["sh2"]
            rec[f"n1_c{cost}"] = m["n1"]
            rec[f"n2_c{cost}"] = m["n2"]
            rec[f"pass_c{cost}"] = wf_ok(m)
        rec["pass1"] = rec["pass_c1"]
        rec["pass2"] = rec["pass_c2"]
        rows.append(rec)
        if rec["pass_c1"]:
            key = (rec["pass_c2"], rec["sh_c1"], rec["avg_c1"], rec["n"])
            if best1 is None or key > best1[0]:
                best1 = (key, rec)
        if rec["pass_c2"]:
            key = (rec["sh_c2"], rec["avg_c2"], rec["n"])
            if best2 is None or key > best2[0]:
                best2 = (key, rec)

    hards = (dtime(15, 15), dtime(14, 30))
    for sig_name, entries in sets.items():
        for hard in hards:
            htag = "1515" if hard == dtime(15, 15) else "1430"
            print(f"Paths {sig_name} {htag}...", flush=True)
            paths, days0 = [], []
            for _, row in entries.iterrows():
                p = path_trade(row, hard)
                if p is None:
                    continue
                paths.append(p)
                days0.append(row.day)
            if len(paths) < N_MIN:
                continue
            for name, kind, kw in cfgs:
                raw = [score_one(p, kind, kw)[0] for p in paths]
                for first_only in (False, True):
                    for pef in (0.0, 15.0):
                        days, pnls, nce, npe, seen = [], [], 0, 0, set()
                        for p, day, pnl in zip(paths, days0, raw):
                            if first_only:
                                if day in seen:
                                    continue
                                seen.add(day)
                            if p["put"] and p["n0f"] < pef:
                                continue
                            days.append(day)
                            pnls.append(pnl)
                            if p["put"]:
                                npe += 1
                            else:
                                nce += 1
                        if len(pnls) < N_MIN:
                            continue
                        pack(days, pnls, dict(signals=sig_name, vehicle=name, hard=htag,
                                              first_only=first_only, pe_floor=pef), nce, npe)

    if not rows:
        print("No hunt2 rows", flush=True)
        return
    res = pd.DataFrame(rows).sort_values(["pass2", "pass1", "sh_c2", "wr_c2"],
                                         ascending=[False, False, False, False])
    outp = OUT / "orb_asym_v3_friction2.csv"
    res.to_csv(outp, index=False)
    p1, p2 = res[res["pass1"] == True], res[res["pass2"] == True]
    cols = ("n nce npe wr avg wr_c1 sh_c1 avg_c1 wr_c2 sh_c2 avg_c2 wr1_c2 sh1_c2 wr2_c2 sh2_c2 "
            "signals vehicle hard first_only pe_floor").split()
    print(f"\nhunt2 rows {len(res)} pass1 {len(p1)} pass2 {len(p2)}", flush=True)
    if len(p2):
        print("2PT PASSERS\n", p2[cols].head(15).to_string(index=False), flush=True)
    else:
        print("No 2pt. Top by sh_c2 among wr_c2>50\n",
              res[res.wr_c2 > 50].sort_values("sh_c2", ascending=False)[cols].head(12).to_string(index=False)
              if (res.wr_c2 > 50).any() else "none with wr_c2>50", flush=True)
        near = res[(res.wr_c2 >= 49) & (res.sh_c2 >= 1.9) & (res.n2 >= 40)]
        print("near 2pt n2>=40\n", near[cols].head(12).to_string(index=False) if len(near) else "(none)", flush=True)
    if best1:
        print("BEST 1pt", {k: best1[1].get(k) for k in cols}, flush=True)
    print("wrote", outp, flush=True)


def _bar(grp, hhmm):
    x = grp.between_time(hhmm, hhmm)
    if x.empty:
        return None
    r = x.iloc[0]
    return x.index[0], float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"])


def _row(day, ts, direction, px, signal):
    return dict(day=day, entry_time=ts, direction=direction, entry_price=px,
                signal=signal, dte=expiry_dte(day))


def _sess_vwap_close(grp, hhmm):
    """Equal-weight typical VWAP — cache volume is always 0."""
    # ponytail: volume missing; upgrade to vol-weighted if the cache ever has it
    sess = grp.between_time("09:15", hhmm)
    if sess.empty:
        return None
    typ = (sess["high"] + sess["low"] + sess["close"]) / 3.0
    vol = sess["volume"] if "volume" in sess.columns else None
    if vol is not None and float(vol.sum()) > 0:
        vwap = float((typ * vol).sum() / vol.sum())
    else:
        vwap = float(typ.mean())  # ponytail: Nifty cache volume is 0
    last = sess.iloc[-1]
    return sess.index[-1], float(last["close"]), vwap


def _to_15m(grp):
    g = grp.between_time("09:15", "12:00")
    rows = list(g.itertuples())
    out = []
    for i in range(0, len(rows) - 2, 3):
        ch = rows[i:i + 3]
        out.append((ch[-1].Index, ch[0].open, max(x.high for x in ch),
                    min(x.low for x in ch), ch[-1].close))
    return out


def gen_dir_engines():
    """Direction engines that do not use OR width, LH/HL vs OR, skip-range, or DefensiveHold."""
    sets = {}

    def put(name, recs):
        sets[name] = pd.DataFrame(recs) if recs else pd.DataFrame()

    buckets = {k: [] for k in (
        "pd_0945", "pd_1000", "pd_0945_m30", "pd_1000_m30",
        "gap_0945_15", "gap_1000_15", "gap_0945_30", "gap_1000_30",
        "gapf_0945_15", "gapf_1000_15",
        "vwap_0945", "vwap_1000", "vwap_1015", "vwap_1000_d20",
        "drive_0930", "drive_0945", "drive_0930_25", "drive_0945_25",
        "yday_1100", "yday_1200",
        "s15_1100", "s15_1200",
        "agree_pd_gap_0945", "agree_drv_vwap_1000",
        "gap_pd_fade_0945",
    )}

    for day, grp in day_map.items():
        inf = prev_info.get(day)
        if inf is None or pd.isna(inf.get("p_close")) or pd.isna(inf.get("p_open")):
            continue
        p_open, p_close = float(inf["p_open"]), float(inf["p_close"])
        p_high, p_low = float(inf["p_high"]), float(inf["p_low"])
        day_open = float(inf["day_open"])
        if p_close <= 0:
            continue
        pd_bull = p_close > p_open
        pd_bear = p_close < p_open
        pd_move = abs(p_close / p_open - 1.0) if p_open else 0.0
        gap = day_open / p_close - 1.0

        b0945 = _bar(grp, "09:45")
        b1000 = _bar(grp, "10:00")
        b1015 = _bar(grp, "10:15")
        b0930 = _bar(grp, "09:30")

        def add_fixed(key, bar, direction, signal):
            if bar is None:
                return
            buckets[key].append(_row(day, bar[0], direction, bar[4], signal))

        if pd_bull or pd_bear:
            dirc = "LONG" if pd_bull else "SHORT"
            add_fixed("pd_0945", b0945, dirc, "pd_oc")
            add_fixed("pd_1000", b1000, dirc, "pd_oc")
            if pd_move >= 0.003:
                add_fixed("pd_0945_m30", b0945, dirc, "pd_oc_m30")
                add_fixed("pd_1000_m30", b1000, dirc, "pd_oc_m30")

        if gap >= 0.0015:
            add_fixed("gap_0945_15", b0945, "LONG", "gap_cont")
            add_fixed("gap_1000_15", b1000, "LONG", "gap_cont")
        elif gap <= -0.0015:
            add_fixed("gap_0945_15", b0945, "SHORT", "gap_cont")
            add_fixed("gap_1000_15", b1000, "SHORT", "gap_cont")
        if gap >= 0.003:
            add_fixed("gap_0945_30", b0945, "LONG", "gap_cont30")
            add_fixed("gap_1000_30", b1000, "LONG", "gap_cont30")
        elif gap <= -0.003:
            add_fixed("gap_0945_30", b0945, "SHORT", "gap_cont30")
            add_fixed("gap_1000_30", b1000, "SHORT", "gap_cont30")
        if gap >= 0.0015:
            add_fixed("gapf_0945_15", b0945, "SHORT", "gap_fade")
            add_fixed("gapf_1000_15", b1000, "SHORT", "gap_fade")
        elif gap <= -0.0015:
            add_fixed("gapf_0945_15", b0945, "LONG", "gap_fade")
            add_fixed("gapf_1000_15", b1000, "LONG", "gap_fade")

        for key, hhmm, bar, mind in (
            ("vwap_0945", "09:45", b0945, 0.0),
            ("vwap_1000", "10:00", b1000, 0.0),
            ("vwap_1015", "10:15", b1015, 0.0),
            ("vwap_1000_d20", "10:00", b1000, 20.0),
        ):
            vw = _sess_vwap_close(grp, hhmm)
            if vw is None or bar is None:
                continue
            ts, cl, v = vw
            if abs(cl - v) < mind:
                continue
            if cl > v:
                buckets[key].append(_row(day, ts, "LONG", cl, "vwap"))
            elif cl < v:
                buckets[key].append(_row(day, ts, "SHORT", cl, "vwap"))

        if b0930 is not None:
            dlt = b0930[4] - day_open
            if dlt > 0:
                add_fixed("drive_0930", b0930, "LONG", "drive")
            elif dlt < 0:
                add_fixed("drive_0930", b0930, "SHORT", "drive")
            if dlt >= 25:
                add_fixed("drive_0930_25", b0930, "LONG", "drive25")
            elif dlt <= -25:
                add_fixed("drive_0930_25", b0930, "SHORT", "drive25")
        if b0945 is not None:
            dlt = b0945[4] - day_open
            if dlt > 0:
                add_fixed("drive_0945", b0945, "LONG", "drive")
            elif dlt < 0:
                add_fixed("drive_0945", b0945, "SHORT", "drive")
            if dlt >= 25:
                add_fixed("drive_0945_25", b0945, "LONG", "drive25")
            elif dlt <= -25:
                add_fixed("drive_0945_25", b0945, "SHORT", "drive25")

        sig = grp.between_time("09:45", "12:00")
        y_long = y_short = False
        for ts, r in sig.iterrows():
            cl = float(r["close"])
            tm = ts.time()
            if (not y_long) and cl > p_high:
                rec = _row(day, ts, "LONG", cl, "yday_hi")
                if tm <= dtime(11, 0):
                    buckets["yday_1100"].append(rec)
                buckets["yday_1200"].append(rec)
                y_long = True
            if (not y_short) and cl < p_low:
                rec = _row(day, ts, "SHORT", cl, "yday_lo")
                if tm <= dtime(11, 0):
                    buckets["yday_1100"].append(rec)
                buckets["yday_1200"].append(rec)
                y_short = True
            if y_long and y_short:
                break

        m15 = _to_15m(grp)
        s_long = s_short = False
        for i in range(1, len(m15)):
            ts, _o, h, lo, cl = m15[i]
            ph, pl = m15[i - 1][2], m15[i - 1][3]
            tm = ts.time()
            if tm < dtime(9, 45):
                continue
            hh_hl = h > ph and lo > pl
            lh_ll = h < ph and lo < pl
            if (not s_long) and hh_hl:
                rec = _row(day, ts, "LONG", cl, "s15_hhhl")
                if tm <= dtime(11, 0):
                    buckets["s15_1100"].append(rec)
                buckets["s15_1200"].append(rec)
                s_long = True
            if (not s_short) and lh_ll:
                rec = _row(day, ts, "SHORT", cl, "s15_lhll")
                if tm <= dtime(11, 0):
                    buckets["s15_1100"].append(rec)
                buckets["s15_1200"].append(rec)
                s_short = True
            if s_long and s_short:
                break

        if b0945 is not None:
            gap_long = gap >= 0.0015
            gap_short = gap <= -0.0015
            if (pd_bull and gap_long) or (pd_bear and gap_short):
                dirc = "LONG" if pd_bull else "SHORT"
                add_fixed("agree_pd_gap_0945", b0945, dirc, "pd_gap")
            if (pd_bull and gap_short) or (pd_bear and gap_long):
                dirc = "LONG" if gap_short else "SHORT"  # fade gap against yesterday
                add_fixed("gap_pd_fade_0945", b0945, dirc, "gap_vs_pd")

        vw10 = _sess_vwap_close(grp, "10:00")
        if vw10 is not None and b1000 is not None:
            cl, v = vw10[1], vw10[2]
            drv = cl - day_open
            if cl > v and drv > 0:
                add_fixed("agree_drv_vwap_1000", b1000, "LONG", "drv_vwap")
            elif cl < v and drv < 0:
                add_fixed("agree_drv_vwap_1000", b1000, "SHORT", "drv_vwap")

    for k, recs in buckets.items():
        put(k, recs)
    return sets


def _side_counts(paths, days, pnls, first_only, pef):
    out_d, out_p, nce, npe, seen = [], [], 0, 0, set()
    for p, day, pnl in zip(paths, days, pnls):
        if first_only:
            if day in seen:
                continue
            seen.add(day)
        if p["put"] and p["n0f"] < pef:
            continue
        out_d.append(day)
        out_p.append(pnl)
        if p["put"]:
            npe += 1
        else:
            nce += 1
    return out_d, out_p, nce, npe


def engine_main():
    """Screenshot 1-2-1 BWB with non-ORB direction engines. Friction 0/1/2."""
    load()
    print("Dir engines (no OR box)...", flush=True)
    sets = gen_dir_engines()
    for name, g in sets.items():
        print(f"  {name}: {len(g)}", flush=True)

    cfgs = [
        ("trail_8_4", "fly_trail", dict(arm=8, give=4)),
        ("trail_8_5", "fly_trail", dict(arm=8, give=5)),
        ("trail_10_4", "fly_trail", dict(arm=10, give=4)),
        ("trail_12_5", "fly_trail", dict(arm=12, give=5)),
        ("trail_6_3", "fly_trail", dict(arm=6, give=3)),
        ("mtm_16_10", "fly_mtm", dict(tgt=16, stp=10)),
        ("mtm_20_12", "fly_mtm", dict(tgt=20, stp=12)),
        ("mtm_12_8", "fly_mtm", dict(tgt=12, stp=8)),
        ("hold", "fly_hold", {}),
    ]

    rows, best0, best1, best2 = [], None, None, None
    ORB_AVG0, ORB_AVG2 = 4.05, 2.05
    MIN_SIDE = 20

    def pack(days, pnls, extra, nce, npe):
        nonlocal best0, best1, best2
        raw = metrics(days, pnls)
        rec = dict(raw)
        rec.update(extra)
        rec["nce"], rec["npe"] = nce, npe
        rec["both_sides"] = nce >= MIN_SIDE and npe >= MIN_SIDE
        rec["avg0"] = raw["avg"]
        rec["inr0"] = round(raw["avg"] * 65, 0) if raw["avg"] == raw["avg"] else float("nan")
        for cost in (0, 1, 2):
            m = metrics(days, [x - cost for x in pnls]) if cost else raw
            rec[f"wr_c{cost}"] = m["wr"]
            rec[f"sh_c{cost}"] = m["sharpe"]
            rec[f"pf_c{cost}"] = m["pf"]
            rec[f"avg_c{cost}"] = m["avg"]
            rec[f"inr_c{cost}"] = round(m["avg"] * 65, 0) if m["avg"] == m["avg"] else float("nan")
            rec[f"wr1_c{cost}"] = m["wr1"]
            rec[f"sh1_c{cost}"] = m["sh1"]
            rec[f"n1_c{cost}"] = m["n1"]
            rec[f"wr2_c{cost}"] = m["wr2"]
            rec[f"sh2_c{cost}"] = m["sh2"]
            rec[f"n2_c{cost}"] = m["n2"]
            rec[f"pass_c{cost}"] = bool(wf_ok(m) and rec["both_sides"])
            rec[f"pass_core_c{cost}"] = wf_ok(m)
        rec["beat0"] = rec["pass_c0"] and rec["avg_c0"] > ORB_AVG0
        rec["beat2"] = rec["pass_c2"] and rec["avg_c2"] > ORB_AVG2
        rows.append(rec)
        if rec["both_sides"] and rec["pass_core_c0"]:
            key = (rec["pass_c2"], rec["pass_c1"], rec["avg_c0"], rec["sh_c0"], rec["n"])
            if best0 is None or key > best0[0]:
                best0 = (key, rec)
        if rec["pass_c1"]:
            key = (rec["pass_c2"], rec["avg_c1"], rec["sh_c1"], rec["n"])
            if best1 is None or key > best1[0]:
                best1 = (key, rec)
        if rec["pass_c2"]:
            key = (rec["avg_c2"], rec["sh_c2"], rec["n"])
            if best2 is None or key > best2[0]:
                best2 = (key, rec)

    path_cache = {}
    hards = (dtime(15, 15), dtime(14, 30))
    for sig_name, entries in sets.items():
        if entries.empty or len(entries) < 40:
            continue
        multi = sig_name.startswith("yday_") or sig_name.startswith("s15_")
        for hard in hards:
            htag = "1515" if hard == dtime(15, 15) else "1430"
            print(f"Paths {sig_name} {htag} n={len(entries)}...", flush=True)
            paths, days0 = [], []
            for _, row in entries.iterrows():
                key = (row.day, row.entry_time, row.direction, htag)
                p = path_cache.get(key)
                if p is None:
                    p = path_trade(row, hard)
                    path_cache[key] = p
                if p is None:
                    continue
                paths.append(p)
                days0.append(row.day)
            if len(paths) < N_MIN:
                print(f"  skip n={len(paths)}", flush=True)
                continue
            nce0 = sum(1 for p in paths if not p["put"])
            npe0 = len(paths) - nce0
            print(f"  n={len(paths)} CE={nce0} PE={npe0}", flush=True)
            for name, kind, kw in cfgs:
                raw = [score_one(p, kind, kw)[0] for p in paths]
                firsts = (False, True) if multi else (False,)
                for first_only in firsts:
                    for pef in (0.0,):
                        ds, ps, nce, npe = _side_counts(paths, days0, raw, first_only, pef)
                        if len(ps) < N_MIN:
                            continue
                        pack(ds, ps, dict(signals=sig_name, vehicle=name, hard=htag,
                                          first_only=first_only, pe_floor=pef), nce, npe)

    if not rows:
        print("No engine rows", flush=True)
        return
    res = pd.DataFrame(rows).sort_values(
        ["pass_c2", "pass_c1", "pass_c0", "avg_c0", "sh_c0"],
        ascending=[False, False, False, False, False],
    )
    outp = OUT / "orb_asym_dir_engine.csv"
    res.to_csv(outp, index=False)
    cols = ("n nce npe wr avg inr0 wr_c1 sh_c1 avg_c1 inr_c1 wr_c2 sh_c2 avg_c2 inr_c2 "
            "wr1_c1 sh1_c1 n1_c1 wr2_c1 sh2_c1 n2_c1 wr1_c2 sh1_c2 wr2_c2 sh2_c2 "
            "signals vehicle hard first_only both_sides pass_c0 pass_c1 pass_c2 beat0 beat2").split()
    p0 = res[res["pass_c0"] == True]
    p1 = res[res["pass_c1"] == True]
    p2 = res[res["pass_c2"] == True]
    print(f"\nrows {len(res)} pass0 {len(p0)} pass1 {len(p1)} pass2 {len(p2)} beat0 {int(res.beat0.sum())} beat2 {int(res.beat2.sum())}", flush=True)
    print("\nTOP 20 by avg0 (both-sides WF @0 if any else raw)", flush=True)
    show = p0 if len(p0) else res.head(20)
    use = [c for c in cols if c in show.columns]
    print(show[use].head(20).to_string(index=False), flush=True)
    if len(p1):
        print("\n1PT PASSERS\n", p1[use].head(15).to_string(index=False), flush=True)
    if len(p2):
        print("\n2PT PASSERS\n", p2[use].head(15).to_string(index=False), flush=True)
    near = res[(res.both_sides) & (res.wr_c1 > 50) & (res.sh_c1 > 1.5) & (res.n >= 80)]
    print("\nNEAR 1pt both-sides WR>50 Sh>1.5\n",
          near[use].head(15).to_string(index=False) if len(near) else "(none)", flush=True)
    one_side = res[(res.nce < MIN_SIDE) | (res.npe < MIN_SIDE)]
    if len(one_side):
        print(f"\nOne-side-starved rows: {len(one_side)} (not 'the strategy')", flush=True)
    if best0:
        print("BEST @0 both+WF", {k: best0[1].get(k) for k in use if k in best0[1]}, flush=True)
    else:
        print("No both-sides WF passer at 0 cost", flush=True)
    if best1:
        print("BEST 1pt", {k: best1[1].get(k) for k in use if k in best1[1]}, flush=True)
    if best2:
        print("BEST 2pt", {k: best2[1].get(k) for k in use if k in best2[1]}, flush=True)
    print("wrote", outp, flush=True)


def engine_refine():
    """Second pass if the lean grid misses: extra times, bands, trails; still no OR box."""
    load()
    print("Refine engines...", flush=True)
    extra = {k: [] for k in (
        "pd_1030", "pd_0945_m50", "pd_0945_band20_80",
        "gap_0945_10", "gap_0945_20_80", "gap_1015_15",
        "vwap_1030", "vwap_1000_d10", "vwap_1000_d40",
        "drive_1000", "drive_0945_40", "drive_0945_15",
        "yday_1015", "yday_1030",
        "s15_1030", "s15_1015",
        "agree_pd_vwap_1000", "agree_pd_drive_0945", "agree_gap_vwap_1000",
        "pd_0945_max80",
    )}
    for day, grp in day_map.items():
        inf = prev_info.get(day)
        if inf is None or pd.isna(inf.get("p_close")) or pd.isna(inf.get("p_open")):
            continue
        p_open, p_close = float(inf["p_open"]), float(inf["p_close"])
        p_high, p_low = float(inf["p_high"]), float(inf["p_low"])
        day_open = float(inf["day_open"])
        if p_close <= 0 or p_open <= 0:
            continue
        pd_bull, pd_bear = p_close > p_open, p_close < p_open
        pd_move = abs(p_close / p_open - 1.0)
        gap = day_open / p_close - 1.0
        b0945, b1000, b1015, b1030 = _bar(grp, "09:45"), _bar(grp, "10:00"), _bar(grp, "10:15"), _bar(grp, "10:30")

        def add(key, bar, dirc, sig):
            if bar is None:
                return
            extra[key].append(_row(day, bar[0], dirc, bar[4], sig))

        dirc = "LONG" if pd_bull else ("SHORT" if pd_bear else None)
        if dirc:
            add("pd_1030", b1030, dirc, "pd_oc")
            if pd_move >= 0.005:
                add("pd_0945_m50", b0945, dirc, "pd_m50")
            if 0.002 <= pd_move <= 0.008:
                add("pd_0945_band20_80", b0945, dirc, "pd_band")
            if pd_move <= 0.008:
                add("pd_0945_max80", b0945, dirc, "pd_max80")
            vw = _sess_vwap_close(grp, "10:00")
            if vw is not None:
                cl, v = vw[1], vw[2]
                if (dirc == "LONG" and cl > v) or (dirc == "SHORT" and cl < v):
                    extra["agree_pd_vwap_1000"].append(_row(day, vw[0], dirc, cl, "pd_vwap"))
            if b0945 is not None:
                drv = b0945[4] - day_open
                if (dirc == "LONG" and drv > 0) or (dirc == "SHORT" and drv < 0):
                    add("agree_pd_drive_0945", b0945, dirc, "pd_drv")

        if abs(gap) >= 0.001:
            add("gap_0945_10", b0945, "LONG" if gap > 0 else "SHORT", "gap10")
        if 0.002 <= abs(gap) <= 0.008:
            add("gap_0945_20_80", b0945, "LONG" if gap > 0 else "SHORT", "gap_band")
        if abs(gap) >= 0.0015:
            add("gap_1015_15", b1015, "LONG" if gap > 0 else "SHORT", "gap15")
        vw10 = _sess_vwap_close(grp, "10:00")
        if vw10 is not None and abs(gap) >= 0.0015:
            cl, v = vw10[1], vw10[2]
            want = "LONG" if gap > 0 else "SHORT"
            if (want == "LONG" and cl > v) or (want == "SHORT" and cl < v):
                extra["agree_gap_vwap_1000"].append(_row(day, vw10[0], want, cl, "gap_vwap"))

        for key, hhmm, bar, mind in (
            ("vwap_1030", "10:30", b1030, 0.0),
            ("vwap_1000_d10", "10:00", b1000, 10.0),
            ("vwap_1000_d40", "10:00", b1000, 40.0),
        ):
            vw = _sess_vwap_close(grp, hhmm)
            if vw is None:
                continue
            ts, cl, v = vw
            if abs(cl - v) < mind:
                continue
            extra[key].append(_row(day, ts, "LONG" if cl > v else "SHORT", cl, "vwap"))

        if b1000 is not None:
            dlt = b1000[4] - day_open
            if dlt != 0:
                add("drive_1000", b1000, "LONG" if dlt > 0 else "SHORT", "drive")
        if b0945 is not None:
            dlt = b0945[4] - day_open
            if abs(dlt) >= 40:
                add("drive_0945_40", b0945, "LONG" if dlt > 0 else "SHORT", "d40")
            if abs(dlt) >= 15:
                add("drive_0945_15", b0945, "LONG" if dlt > 0 else "SHORT", "d15")

        sig = grp.between_time("09:45", "10:30")
        y_long = y_short = False
        for ts, r in sig.iterrows():
            cl = float(r["close"])
            tm = ts.time()
            if (not y_long) and cl > p_high:
                rec = _row(day, ts, "LONG", cl, "yday_hi")
                if tm <= dtime(10, 15):
                    extra["yday_1015"].append(rec)
                extra["yday_1030"].append(rec)
                y_long = True
            if (not y_short) and cl < p_low:
                rec = _row(day, ts, "SHORT", cl, "yday_lo")
                if tm <= dtime(10, 15):
                    extra["yday_1015"].append(rec)
                extra["yday_1030"].append(rec)
                y_short = True

        m15 = _to_15m(grp)
        s_long = s_short = False
        for i in range(1, len(m15)):
            ts, _o, h, lo, cl = m15[i]
            ph, pl = m15[i - 1][2], m15[i - 1][3]
            tm = ts.time()
            if tm < dtime(9, 45) or tm > dtime(10, 30):
                continue
            if (not s_long) and h > ph and lo > pl:
                rec = _row(day, ts, "LONG", cl, "s15")
                if tm <= dtime(10, 15):
                    extra["s15_1015"].append(rec)
                extra["s15_1030"].append(rec)
                s_long = True
            if (not s_short) and h < ph and lo < pl:
                rec = _row(day, ts, "SHORT", cl, "s15")
                if tm <= dtime(10, 15):
                    extra["s15_1015"].append(rec)
                extra["s15_1030"].append(rec)
                s_short = True

    sets = {k: pd.DataFrame(v) if v else pd.DataFrame() for k, v in extra.items()}
    for name, g in sets.items():
        print(f"  {name}: {len(g)}", flush=True)

    cfgs = []
    for arm, give in product((6, 8, 10, 12, 16), (3, 4, 5, 6, 8)):
        cfgs.append((f"trail_{arm}_{give}", "fly_trail", dict(arm=arm, give=give)))
    for mt, ms in product((10, 14, 16, 20, 25), (8, 10, 12)):
        cfgs.append((f"mtm_{mt}_{ms}", "fly_mtm", dict(tgt=mt, stp=ms)))
    cfgs.append(("hold", "fly_hold", {}))

    rows = []
    MIN_SIDE = 20
    path_cache = {}

    def pack(days, pnls, extra_meta, nce, npe):
        raw = metrics(days, pnls)
        rec = dict(raw)
        rec.update(extra_meta)
        rec["nce"], rec["npe"] = nce, npe
        rec["both_sides"] = nce >= MIN_SIDE and npe >= MIN_SIDE
        rec["avg0"] = raw["avg"]
        rec["inr0"] = round(raw["avg"] * 65, 0) if raw["avg"] == raw["avg"] else float("nan")
        for cost in (0, 1, 2):
            m = metrics(days, [x - cost for x in pnls]) if cost else raw
            rec[f"wr_c{cost}"] = m["wr"]
            rec[f"sh_c{cost}"] = m["sharpe"]
            rec[f"pf_c{cost}"] = m["pf"]
            rec[f"avg_c{cost}"] = m["avg"]
            rec[f"inr_c{cost}"] = round(m["avg"] * 65, 0) if m["avg"] == m["avg"] else float("nan")
            rec[f"wr1_c{cost}"] = m["wr1"]
            rec[f"sh1_c{cost}"] = m["sh1"]
            rec[f"n1_c{cost}"] = m["n1"]
            rec[f"wr2_c{cost}"] = m["wr2"]
            rec[f"sh2_c{cost}"] = m["sh2"]
            rec[f"n2_c{cost}"] = m["n2"]
            rec[f"pass_c{cost}"] = bool(wf_ok(m) and rec["both_sides"])
        rec["beat0"] = rec["pass_c0"] and rec["avg_c0"] > 4.05
        rec["beat2"] = rec["pass_c2"] and rec["avg_c2"] > 2.05
        rows.append(rec)

    hards = (dtime(15, 15), dtime(14, 30))
    for sig_name, entries in sets.items():
        if entries.empty or len(entries) < N_MIN:
            continue
        multi = sig_name.startswith("yday_") or sig_name.startswith("s15_")
        for hard in hards:
            htag = "1515" if hard == dtime(15, 15) else "1430"
            print(f"Paths {sig_name} {htag}...", flush=True)
            paths, days0 = [], []
            for _, row in entries.iterrows():
                key = (row.day, row.entry_time, row.direction, htag)
                p = path_cache.get(key)
                if p is None:
                    p = path_trade(row, hard)
                    path_cache[key] = p
                if p is None:
                    continue
                paths.append(p)
                days0.append(row.day)
            if len(paths) < N_MIN:
                continue
            print(f"  n={len(paths)}", flush=True)
            for name, kind, kw in cfgs:
                raw = [score_one(p, kind, kw)[0] for p in paths]
                for first_only in ((False, True) if multi else (False,)):
                    ds, ps, nce, npe = _side_counts(paths, days0, raw, first_only, 0.0)
                    if len(ps) < N_MIN:
                        continue
                    pack(ds, ps, dict(signals=sig_name, vehicle=name, hard=htag,
                                      first_only=first_only, pe_floor=0.0), nce, npe)

    if not rows:
        print("No refine rows", flush=True)
        return
    res = pd.DataFrame(rows).sort_values(
        ["pass_c2", "pass_c1", "pass_c0", "avg_c0", "sh_c0"],
        ascending=[False, False, False, False, False],
    )
    outp = OUT / "orb_asym_dir_engine2.csv"
    res.to_csv(outp, index=False)
    p0, p1, p2 = res[res.pass_c0 == True], res[res.pass_c1 == True], res[res.pass_c2 == True]
    cols = ("n nce npe wr avg inr0 wr_c1 sh_c1 avg_c1 inr_c1 wr_c2 sh_c2 avg_c2 inr_c2 "
            "signals vehicle hard first_only both_sides pass_c0 pass_c1 pass_c2 beat0 beat2").split()
    use = [c for c in cols if c in res.columns]
    print(f"\nrefine rows {len(res)} pass0 {len(p0)} pass1 {len(p1)} pass2 {len(p2)}", flush=True)
    print("TOP\n", (p0 if len(p0) else res).head(15)[use].to_string(index=False), flush=True)
    if len(p1):
        print("1PT\n", p1[use].head(10).to_string(index=False), flush=True)
    if len(p2):
        print("2PT\n", p2[use].head(10).to_string(index=False), flush=True)
    print("wrote", outp, flush=True)


def gen_selective():
    """Tighter non-ORB filters: net displacement vs open (not high-low OR width)."""
    keys = (
        "drv_0945_40", "drv_0945_60", "drv_0945_80", "drv_0945_100",
        "drv_1000_40", "drv_1000_60", "drv_1000_80",
        "drv_0945_40_120", "drv_0945_60_150",
        "hold_drv_40", "hold_drv_60",
        "drv_vwap_40", "drv_vwap_60",
        "drv_pd_60", "drv_gap_40",
        "triple_40", "triple_60",
        "vwap_d40", "vwap_d60",
        "gap_50", "gap_80",
        "fade_fail_25",
        "drv_0945_40_debitish",
    )
    buckets = {k: [] for k in keys}
    for day, grp in day_map.items():
        inf = prev_info.get(day)
        if inf is None or pd.isna(inf.get("p_close")) or pd.isna(inf.get("p_open")):
            continue
        p_open, p_close = float(inf["p_open"]), float(inf["p_close"])
        day_open = float(inf["day_open"])
        if p_close <= 0 or p_open <= 0:
            continue
        pd_bull, pd_bear = p_close > p_open, p_close < p_open
        gap = day_open / p_close - 1.0
        b0930, b0945, b1000 = _bar(grp, "09:30"), _bar(grp, "09:45"), _bar(grp, "10:00")
        vw10 = _sess_vwap_close(grp, "10:00")
        vw45 = _sess_vwap_close(grp, "09:45")

        def add(key, bar, dirc, sig):
            if bar is None:
                return
            buckets[key].append(_row(day, bar[0], dirc, bar[4], sig))

        def side(dlt):
            if dlt > 0:
                return "LONG"
            if dlt < 0:
                return "SHORT"
            return None

        if b0945 is not None:
            d45 = b0945[4] - day_open
            s45 = side(d45)
            a45 = abs(d45)
            if s45 and a45 >= 40:
                add("drv_0945_40", b0945, s45, "drv40")
            if s45 and a45 >= 60:
                add("drv_0945_60", b0945, s45, "drv60")
            if s45 and a45 >= 80:
                add("drv_0945_80", b0945, s45, "drv80")
            if s45 and a45 >= 100:
                add("drv_0945_100", b0945, s45, "drv100")
            if s45 and 40 <= a45 <= 120:
                add("drv_0945_40_120", b0945, s45, "drv_band")
            if s45 and 60 <= a45 <= 150:
                add("drv_0945_60_150", b0945, s45, "drv_band")
            if b0930 is not None:
                d30 = b0930[4] - day_open
                s30 = side(d30)
                if s30 and s45 == s30 and a45 >= 40:
                    add("hold_drv_40", b0945, s45, "hold40")
                if s30 and s45 == s30 and a45 >= 60:
                    add("hold_drv_60", b0945, s45, "hold60")
                # failed drive: 09:30 ≥25 then 09:45 back through open
                if abs(d30) >= 25 and s30 and s45 and s45 != s30:
                    add("fade_fail_25", b0945, s45, "fade_fail")
            if s45 and a45 >= 60:
                if (s45 == "LONG" and pd_bull) or (s45 == "SHORT" and pd_bear):
                    add("drv_pd_60", b0945, s45, "drv_pd")
            if s45 and a45 >= 40:
                if (s45 == "LONG" and gap >= 0.0015) or (s45 == "SHORT" and gap <= -0.0015):
                    add("drv_gap_40", b0945, s45, "drv_gap")
            if vw45 is not None and s45 and a45 >= 40:
                cl, v = vw45[1], vw45[2]
                if (s45 == "LONG" and cl > v) or (s45 == "SHORT" and cl < v):
                    add("drv_0945_40_debitish", b0945, s45, "drv_vwap45")

        if b1000 is not None:
            d10 = b1000[4] - day_open
            s10 = side(d10)
            a10 = abs(d10)
            if s10 and a10 >= 40:
                add("drv_1000_40", b1000, s10, "drv40")
            if s10 and a10 >= 60:
                add("drv_1000_60", b1000, s10, "drv60")
            if s10 and a10 >= 80:
                add("drv_1000_80", b1000, s10, "drv80")
            if vw10 is not None:
                cl, v = vw10[1], vw10[2]
                vs = "LONG" if cl > v else ("SHORT" if cl < v else None)
                if vs and abs(cl - v) >= 40:
                    buckets["vwap_d40"].append(_row(day, vw10[0], vs, cl, "vwap40"))
                if vs and abs(cl - v) >= 60:
                    buckets["vwap_d60"].append(_row(day, vw10[0], vs, cl, "vwap60"))
                if s10 and vs == s10 and a10 >= 40:
                    add("drv_vwap_40", b1000, s10, "drv_vwap")
                if s10 and vs == s10 and a10 >= 60:
                    add("drv_vwap_60", b1000, s10, "drv_vwap")
                if s10 and vs == s10 and a10 >= 40:
                    pd_ok = (s10 == "LONG" and pd_bull) or (s10 == "SHORT" and pd_bear)
                    if pd_ok:
                        add("triple_40", b1000, s10, "triple")
                if s10 and vs == s10 and a10 >= 60:
                    pd_ok = (s10 == "LONG" and pd_bull) or (s10 == "SHORT" and pd_bear)
                    if pd_ok:
                        add("triple_60", b1000, s10, "triple")

        if abs(gap) >= 0.005 and b0945 is not None:
            add("gap_50", b0945, "LONG" if gap > 0 else "SHORT", "gap50")
        if abs(gap) >= 0.008 and b0945 is not None:
            add("gap_80", b0945, "LONG" if gap > 0 else "SHORT", "gap80")

    return {k: pd.DataFrame(v) if v else pd.DataFrame() for k, v in buckets.items()}


def _pack_engine_rows(sets, cfgs, out_name, pe_floors=(0.0, 15.0), skip_credit=True):
    rows = []
    MIN_SIDE = 20
    path_cache = {}

    def pack(days, pnls, extra, nce, npe):
        raw = metrics(days, pnls)
        rec = dict(raw)
        rec.update(extra)
        rec["nce"], rec["npe"] = nce, npe
        rec["both_sides"] = nce >= MIN_SIDE and npe >= MIN_SIDE
        rec["avg0"] = raw["avg"]
        rec["inr0"] = round(raw["avg"] * 65, 0) if raw["avg"] == raw["avg"] else float("nan")
        for cost in (0, 1, 2):
            m = metrics(days, [x - cost for x in pnls]) if cost else raw
            rec[f"wr_c{cost}"] = m["wr"]
            rec[f"sh_c{cost}"] = m["sharpe"]
            rec[f"pf_c{cost}"] = m["pf"]
            rec[f"avg_c{cost}"] = m["avg"]
            rec[f"inr_c{cost}"] = round(m["avg"] * 65, 0) if m["avg"] == m["avg"] else float("nan")
            rec[f"wr1_c{cost}"] = m["wr1"]
            rec[f"sh1_c{cost}"] = m["sh1"]
            rec[f"n1_c{cost}"] = m["n1"]
            rec[f"wr2_c{cost}"] = m["wr2"]
            rec[f"sh2_c{cost}"] = m["sh2"]
            rec[f"n2_c{cost}"] = m["n2"]
            rec[f"pass_c{cost}"] = bool(wf_ok(m) and rec["both_sides"])
        rec["beat0"] = rec.get("pass_c0") and rec["avg_c0"] > 4.05
        rec["beat2"] = rec.get("pass_c2") and rec["avg_c2"] > 2.05
        rows.append(rec)

    hards = (dtime(15, 15), dtime(14, 30))
    for sig_name, entries in sets.items():
        if entries.empty or len(entries) < 50:
            print(f"  {sig_name}: {len(entries)} skip", flush=True)
            continue
        print(f"  {sig_name}: {len(entries)}", flush=True)
        for hard in hards:
            htag = "1515" if hard == dtime(15, 15) else "1430"
            paths, days0 = [], []
            for _, row in entries.iterrows():
                key = (row.day, row.entry_time, row.direction, htag)
                p = path_cache.get(key)
                if p is None:
                    p = path_trade(row, hard)
                    path_cache[key] = p
                if p is None:
                    continue
                paths.append(p)
                days0.append(row.day)
            if len(paths) < N_MIN:
                continue
            print(f"    {htag} n={len(paths)}", flush=True)
            for name, kind, kw in cfgs:
                raw = [score_one(p, kind, kw)[0] for p in paths]
                for first_only in (False,):
                    for pef in pe_floors:
                        for sc in ((False, True) if skip_credit else (False,)):
                            ds, ps, nce, npe, seen = [], [], 0, 0, set()
                            for p, day, pnl in zip(paths, days0, raw):
                                if first_only:
                                    if day in seen:
                                        continue
                                    seen.add(day)
                                if p["put"] and p["n0f"] < pef:
                                    continue
                                if sc and p["n0f"] <= 0:
                                    continue
                                ds.append(day)
                                ps.append(pnl)
                                if p["put"]:
                                    npe += 1
                                else:
                                    nce += 1
                            if len(ps) < N_MIN:
                                continue
                            pack(ds, ps, dict(signals=sig_name, vehicle=name, hard=htag,
                                              first_only=first_only, pe_floor=pef,
                                              skip_credit=sc), nce, npe)
    if not rows:
        print("No selective rows", flush=True)
        return None
    res = pd.DataFrame(rows).sort_values(
        ["pass_c2", "pass_c1", "pass_c0", "avg_c0", "sh_c0"],
        ascending=[False, False, False, False, False],
    )
    outp = OUT / out_name
    res.to_csv(outp, index=False)
    p0, p1, p2 = res[res.pass_c0 == True], res[res.pass_c1 == True], res[res.pass_c2 == True]
    cols = ("n nce npe wr avg inr0 wr_c1 sh_c1 avg_c1 inr_c1 wr_c2 sh_c2 avg_c2 inr_c2 "
            "wr1_c0 sh1_c0 n1_c0 wr2_c0 sh2_c0 n2_c0 "
            "signals vehicle hard skip_credit pe_floor both_sides pass_c0 pass_c1 pass_c2 beat0 beat2").split()
    use = [c for c in cols if c in res.columns]
    print(f"\nrows {len(res)} pass0 {len(p0)} pass1 {len(p1)} pass2 {len(p2)} beat0 {int(res.beat0.sum())} beat2 {int(res.beat2.sum())}", flush=True)
    print("TOP 15\n", (p0 if len(p0) else res)[use].head(15).to_string(index=False), flush=True)
    if len(p1):
        print("1PT\n", p1[use].head(10).to_string(index=False), flush=True)
    if len(p2):
        print("2PT\n", p2[use].head(10).to_string(index=False), flush=True)
    near = res[(res.both_sides) & (res.wr > 50) & (res.sharpe > 1.5) & (res.n >= 80)]
    print("NEAR 0pt WF-ish\n", near[use].head(12).to_string(index=False) if len(near) else "(none)", flush=True)
    print("wrote", outp, flush=True)
    return res


def engine_selective():
    load()
    print("Selective non-ORB engines...", flush=True)
    sets = gen_selective()
    cfgs = [
        ("trail_8_4", "fly_trail", dict(arm=8, give=4)),
        ("trail_8_5", "fly_trail", dict(arm=8, give=5)),
        ("trail_10_4", "fly_trail", dict(arm=10, give=4)),
        ("trail_12_6", "fly_trail", dict(arm=12, give=6)),
        ("trail_6_3", "fly_trail", dict(arm=6, give=3)),
        ("mtm_16_10", "fly_mtm", dict(tgt=16, stp=10)),
        ("mtm_20_12", "fly_mtm", dict(tgt=20, stp=12)),
        ("mtm_25_12", "fly_mtm", dict(tgt=25, stp=12)),
        ("hold", "fly_hold", {}),
    ]
    return _pack_engine_rows(sets, cfgs, "orb_asym_dir_engine3.csv")


def gen_gap_grid():
    """Gap-continuation grid: |gap| bands, entry time, confirm — no OR box."""
    gmins = (0.006, 0.007, 0.008, 0.009, 0.010)
    gmaxs = (None, 0.012, 0.015, 0.020)
    times = ("09:45", "10:00", "10:15")
    confirms = ("none", "drive", "vwap", "hold")
    buckets = {}
    for day, grp in day_map.items():
        inf = prev_info.get(day)
        if inf is None or pd.isna(inf.get("p_close")):
            continue
        p_close = float(inf["p_close"])
        day_open = float(inf["day_open"])
        if p_close <= 0:
            continue
        gap = day_open / p_close - 1.0
        ag = abs(gap)
        if ag < 0.006:
            continue
        dirc = "LONG" if gap > 0 else "SHORT"
        b30 = _bar(grp, "09:30")
        bars = {t: _bar(grp, t) for t in times}
        vw = {t: _sess_vwap_close(grp, t) for t in times}
        for gmin in gmins:
            if ag < gmin:
                continue
            for gmax in gmaxs:
                if gmax is not None and ag > gmax:
                    continue
                for t in times:
                    bar = bars[t]
                    if bar is None:
                        continue
                    cl, o_px = bar[4], day_open
                    drv_ok = (dirc == "LONG" and cl >= o_px) or (dirc == "SHORT" and cl <= o_px)
                    v = vw[t]
                    vwap_ok = False
                    if v is not None:
                        vwap_ok = (dirc == "LONG" and v[1] > v[2]) or (dirc == "SHORT" and v[1] < v[2])
                    hold_ok = drv_ok
                    if b30 is not None:
                        c30 = b30[4]
                        hold_ok = drv_ok and (
                            (dirc == "LONG" and c30 >= o_px) or (dirc == "SHORT" and c30 <= o_px)
                        )
                    for conf, ok in (("none", True), ("drive", drv_ok), ("vwap", vwap_ok), ("hold", hold_ok)):
                        if not ok:
                            continue
                        gtag = "x" if gmax is None else str(int(gmax * 1000))
                        name = f"gap_{int(gmin*1000)}_{gtag}_{t.replace(':','')}_{conf}"
                        buckets.setdefault(name, []).append(_row(day, bar[0], dirc, cl, name))
    return {k: pd.DataFrame(v) for k, v in buckets.items()}


def engine_gap():
    load()
    print("Gap-continuation retune...", flush=True)
    sets = gen_gap_grid()
    print(f"  sets {len(sets)}", flush=True)
    cfgs = [
        ("trail_8_4", "fly_trail", dict(arm=8, give=4)),
        ("trail_8_5", "fly_trail", dict(arm=8, give=5)),
        ("trail_10_4", "fly_trail", dict(arm=10, give=4)),
        ("trail_10_5", "fly_trail", dict(arm=10, give=5)),
        ("trail_12_6", "fly_trail", dict(arm=12, give=6)),
        ("trail_16_8", "fly_trail", dict(arm=16, give=8)),
        ("mtm_16_10", "fly_mtm", dict(tgt=16, stp=10)),
        ("mtm_20_12", "fly_mtm", dict(tgt=20, stp=12)),
        ("mtm_25_12", "fly_mtm", dict(tgt=25, stp=12)),
        ("side_25_12_16_10", "fly_side", dict(tgt_ce=25, stp_ce=12, tgt_pe=16, stp_pe=10)),
        ("side_20_10_12_8", "fly_side", dict(tgt_ce=20, stp_ce=10, tgt_pe=12, stp_pe=8)),
        ("hold", "fly_hold", {}),
    ]
    return _pack_engine_rows(sets, cfgs, "orb_asym_dir_gap.csv", pe_floors=(0.0,), skip_credit=False)


def gen_gap_hold():
    """Push n2 to 40 on the gap+VWAP@10:00 hold lead, still no OR."""
    buckets = {}
    gmins = (0.005, 0.0055, 0.0058, 0.006, 0.0062, 0.0065)
    gmaxs = (0.018, 0.022, 0.025, 0.028, 0.030, 0.035, None)
    times = ("09:45", "10:00", "10:15")
    for day, grp in day_map.items():
        inf = prev_info.get(day)
        if inf is None or pd.isna(inf.get("p_close")):
            continue
        p_close = float(inf["p_close"])
        day_open = float(inf["day_open"])
        if p_close <= 0:
            continue
        gap = day_open / p_close - 1.0
        ag = abs(gap)
        if ag < 0.005:
            continue
        dirc = "LONG" if gap > 0 else "SHORT"
        for t in times:
            bar = _bar(grp, t)
            vw = _sess_vwap_close(grp, t)
            if bar is None or vw is None:
                continue
            cl, v = vw[1], vw[2]
            if not ((dirc == "LONG" and cl > v) or (dirc == "SHORT" and cl < v)):
                continue
            for gmin in gmins:
                if ag < gmin:
                    continue
                for gmax in gmaxs:
                    if gmax is not None and ag > gmax:
                        continue
                    gtag = "x" if gmax is None else str(int(round(gmax * 1000)))
                    name = f"g{int(round(gmin*10000))}_{gtag}_{t.replace(':','')}"
                    buckets.setdefault(name, []).append(_row(day, bar[0], dirc, cl, name))
    return {k: pd.DataFrame(v) for k, v in buckets.items()}


def engine_gap_hold():
    load()
    print("Gap+VWAP hold rescue (n2>=40)...", flush=True)
    sets = gen_gap_hold()
    print(f"  sets {len(sets)}", flush=True)
    cfgs = [
        ("hold", "fly_hold", {}),
        ("mtm_40_15", "fly_mtm", dict(tgt=40, stp=15)),
        ("mtm_50_20", "fly_mtm", dict(tgt=50, stp=20)),
        ("mtm_80_15", "fly_mtm", dict(tgt=80, stp=15)),
        ("mtm_80_20", "fly_mtm", dict(tgt=80, stp=20)),
        ("mtm_100_25", "fly_mtm", dict(tgt=100, stp=25)),
        ("trail_16_8", "fly_trail", dict(arm=16, give=8)),
        ("trail_20_8", "fly_trail", dict(arm=20, give=8)),
        ("trail_25_10", "fly_trail", dict(arm=25, give=10)),
        ("trail_8_5", "fly_trail", dict(arm=8, give=5)),
        ("trail_8_4", "fly_trail", dict(arm=8, give=4)),
    ]
    return _pack_engine_rows(sets, cfgs, "orb_asym_dir_gap_hold.csv", pe_floors=(0.0,), skip_credit=False)


def expiry_compare(out_csv):
    """g58_35_1000 book: 15:15 same-day vs hold to weekly expiry (intrinsic)."""
    hard = dtime(15, 15)
    sets = gen_gap_hold()
    if "g58_35_1000" not in sets or sets["g58_35_1000"].empty:
        print("no g58_35_1000 trades", flush=True)
        return pd.DataFrame()
    entries = sets["g58_35_1000"]
    rows = []
    for _, row in entries.iterrows():
        p = path_trade(row, hard)
        if p is None:
            continue
        day_pnl, _ = score_one(p, "fly_hold", {})
        ed = expiry_date(row.day)
        Sx = close_1515(ed) if ed is not None else None
        if Sx is None:
            continue
        put = row.direction == "SHORT"
        fl, fb, ff = fly_strikes(float(row.entry_price), put)
        nx = fly_net(Sx, 1.0 / (365 * 24), fl, fb, ff, not put)
        exp_pnl = nx - p["n0f"]
        rows.append(dict(
            day=row.day, dte=int(row.dte), direction=row.direction,
            entry=float(row.entry_price), n0=p["n0f"],
            S_exp=Sx, n_exp=nx, pnl_day=day_pnl, pnl_exp=exp_pnl,
        ))
    t = pd.DataFrame(rows)
    t.to_csv(out_csv, index=False)

    def dump(label, days, pnls):
        m = metrics(days, pnls)
        print(f"{label}: n={m['n']} WR={m['wr']}% PF={m['pf']} Sh={m['sharpe']} "
              f"avg={m['avg']:+.2f} (₹{m['avg']*LOT:.0f}) H1 {m['wr1']}/{m['sh1']} H2 {m['wr2']}/{m['sh2']}",
              flush=True)
        return m

    print(f"{INDEX} paired {len(t)}  dte0={(t.dte==0).sum()}  dte>0={(t.dte>0).sum()}  lot={LOT}", flush=True)
    dump("same-day 15:15", t.day.tolist(), t.pnl_day.tolist())
    dump("hold to expiry", t.day.tolist(), t.pnl_exp.tolist())
    overnight = t[t.dte > 0]
    if len(overnight):
        dump("expiry dte>0 only", overnight.day.tolist(), overnight.pnl_exp.tolist())
        dump("same-day dte>0 only", overnight.day.tolist(), overnight.pnl_day.tolist())
    for cost in (1.0, 2.0):
        dump(f"expiry -{int(cost)}pt", t.day.tolist(), (t.pnl_exp - cost).tolist())
    for side, dirc in (("CE", "LONG"), ("PE", "SHORT")):
        s = t[t.direction == dirc]
        if s.empty:
            continue
        dump(f"expiry {side}", s.day.tolist(), s.pnl_exp.tolist())
        dump(f"same-day {side}", s.day.tolist(), s.pnl_day.tolist())
    print("wrote", out_csv, flush=True)
    return t


def expiry_main():
    """Same g58_35_1000 book: 15:15 same-day vs hold to weekly expiry (intrinsic)."""
    load()
    expiry_compare(OUT / "orb_asym_expiry_hold_trades.csv")


def fetch_sensex_cache(start="2021-07-01", end="2026-06-30"):
    import os
    import requests
    if SENSEX_CACHE.exists() and SENSEX_CACHE.stat().st_size > 1000:
        print("Sensex cache exists", SENSEX_CACHE, flush=True)
        return
    key = os.getenv("OPENALGO_API_KEY", "")
    if not key:
        raise SystemExit("Set OPENALGO_API_KEY")
    start_d, end_d = ddate.fromisoformat(start), ddate.fromisoformat(end)
    frames = []
    cur = start_d
    while cur <= end_d:
        nxt = min(cur + timedelta(days=90), end_d)
        print(f"Fetching SENSEX 5m {cur}→{nxt}", flush=True)
        r = requests.post(
            "http://127.0.0.1:5000/api/v1/history",
            json={"apikey": key, "symbol": "SENSEX", "exchange": "BSE_INDEX",
                  "interval": "5m", "start_date": cur.isoformat(), "end_date": nxt.isoformat()},
            timeout=120,
        )
        r.raise_for_status()
        j = r.json()
        if j.get("status") == "success" and j.get("data"):
            frames.append(pd.DataFrame(j["data"]))
        cur = nxt + timedelta(days=1)
    if not frames:
        raise SystemExit("No Sensex history returned")
    raw = pd.concat(frames, ignore_index=True)
    ts = pd.to_datetime(raw["timestamp"], unit="s", utc=True).dt.tz_convert("Asia/Kolkata")
    raw.index = ts
    cols = [c for c in ("open", "high", "low", "close", "volume") if c in raw.columns]
    raw = raw[cols].sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    SENSEX_CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(SENSEX_CACHE)
    print(f"Cached {len(raw):,} Sensex bars {raw.index.min()}→{raw.index.max()}", flush=True)


def sensex_main():
    global INDEX, LOT
    fetch_sensex_cache()
    INDEX, LOT = "SENSEX", SENSEX_LOT
    load(SENSEX_CACHE)
    expiry_compare(OUT / "orb_asym_sensex_expiry_trades.csv")
    INDEX, LOT = "NIFTY", 65


def pe_sharpen_main():
    """PE (down-gap) exits/filters. Mix: CE hold-to-expiry + PE 15:15."""
    load()
    hard = dtime(15, 15)
    sets = gen_gap_hold()
    cfgs = [
        ("hold", "fly_hold", {}),
        ("trail_6_3", "fly_trail", dict(arm=6, give=3)),
        ("trail_8_4", "fly_trail", dict(arm=8, give=4)),
        ("trail_8_5", "fly_trail", dict(arm=8, give=5)),
        ("trail_10_4", "fly_trail", dict(arm=10, give=4)),
        ("mtm_8_6", "fly_mtm", dict(tgt=8, stp=6)),
        ("mtm_12_8", "fly_mtm", dict(tgt=12, stp=8)),
        ("mtm_16_8", "fly_mtm", dict(tgt=16, stp=8)),
        ("mtm_20_10", "fly_mtm", dict(tgt=20, stp=10)),
    ]
    rows = []
    for sig_name, entries in sets.items():
        pe = entries[entries.direction == "SHORT"]
        if len(pe) < 30:
            continue
        for name, kind, kw in cfgs:
            days, pnls = [], []
            for _, row in pe.iterrows():
                p = path_trade(row, hard)
                if p is None:
                    continue
                pnl, _ = score_one(p, kind, kw)
                days.append(row.day)
                pnls.append(pnl)
            if len(pnls) < 30:
                continue
            m = metrics(days, pnls)
            m.update(signals=sig_name, vehicle=name, side="PE",
                     passed=bool(m["wr"] > 50 and m["sharpe"] > 2 and m["n"] >= 30))
            rows.append(m)
    res = pd.DataFrame(rows).sort_values(["passed", "sharpe", "avg"], ascending=[False, False, False])
    res.to_csv(OUT / "orb_asym_pe_sharpen.csv", index=False)
    print(res.head(15).to_string(index=False), flush=True)

    # Mixed book on locked gap set
    entries = sets["g58_35_1000"]
    mix_days, mix_pnls = [], []
    pe_days, pe_pnls = [], []
    ce_days, ce_pnls = [], []
    for _, row in entries.iterrows():
        p = path_trade(row, hard)
        if p is None:
            continue
        day_pnl, _ = score_one(p, "fly_hold", {})
        ed = expiry_date(row.day)
        Sx = close_1515(ed) if ed is not None else None
        if Sx is None:
            continue
        put = row.direction == "SHORT"
        fl, fb, ff = fly_strikes(float(row.entry_price), put)
        nx = fly_net(Sx, 1.0 / (365 * 24), fl, fb, ff, not put)
        exp_pnl = nx - p["n0f"]
        if put:
            pnl = day_pnl
            pe_days.append(row.day)
            pe_pnls.append(pnl)
        else:
            pnl = exp_pnl
            ce_days.append(row.day)
            ce_pnls.append(pnl)
        mix_days.append(row.day)
        mix_pnls.append(pnl)
    print("\nMIX CE-expiry + PE-15:15", metrics(mix_days, mix_pnls), flush=True)
    print("PE 15:15 only", metrics(pe_days, pe_pnls), flush=True)
    print("CE expiry only", metrics(ce_days, ce_pnls), flush=True)
    pd.DataFrame({"day": mix_days, "pnl": mix_pnls}).to_csv(OUT / "orb_asym_pe_mix_trades.csv", index=False)
    print("wrote", OUT / "orb_asym_pe_sharpen.csv", flush=True)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "engine"
    if cmd == "friction":
        friction_main()
    elif cmd == "hunt2":
        hunt2_main()
    elif cmd == "refine":
        engine_refine()
    elif cmd == "selective":
        engine_selective()
    elif cmd == "gap":
        engine_gap()
    elif cmd == "gaphold":
        engine_gap_hold()
    elif cmd == "expiry":
        expiry_main()
    elif cmd == "sensex":
        sensex_main()
    elif cmd == "pe":
        pe_sharpen_main()
    elif cmd == "search":
        main()
    else:
        engine_main()
