"""ORB_Asym — replay locked v2 paths with BS 1-2-1 MTM (live-like).

Live P&L is (exit_net - entry_net), not intrinsic-minus-7.5.
No chain history here: Black-Scholes European, 50pt strikes, DTE from
Nifty weekly expiry calendar (Thu until 2025-08-31, Tue after).

Same entries/exits as orb_asym_v2_best_trades.csv (spot TARGET 50 / STOP 25).
"""
from datetime import date as ddate, datetime, timedelta
from math import erf, exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "orb_asym"
TRADES = OUT / "orb_asym_v2_best_trades.csv"
CACHE = OUT / "nifty_5m_cache.pkl"
BODY, FAR = 50, 150
EXPIRY_SWITCH = ddate(2025, 9, 1)


def n_cdf(x):
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs(S, K, T, sigma, call, r=0.0):
    if S <= 0 or K <= 0:
        return 0.0
    if T <= 1.0 / (365 * 24):  # <1 hour: intrinsic
        return max(S - K, 0.0) if call else max(K - S, 0.0)
    vol = sigma * sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol
    d2 = d1 - vol
    df = exp(-r * T)
    if call:
        return S * n_cdf(d1) - K * df * n_cdf(d2)
    return K * df * n_cdf(-d2) - S * n_cdf(-d1)


def round_strike(px):
    return round(px / 50.0) * 50.0


def fly_net(S, T, sigma, put):
    atm = round_strike(S)
    if put:
        body, far = atm - BODY, atm - FAR
    else:
        body, far = atm + BODY, atm + FAR
    call = not put
    return (
        bs(S, atm, T, sigma, call)
        + bs(S, far, T, sigma, call)
        - 2.0 * bs(S, body, T, sigma, call)
    )


def years(dte, hhmm, session_end=(15, 15)):
    h, m = hhmm
    left = (session_end[0] * 60 + session_end[1]) - (h * 60 + m)
    left = max(left, 5) / (6.25 * 60)  # ~6.25h session as 1 trading day fraction
    return max((dte + left) / 365.0, 1.0 / (365 * 24))


def expiry_date(day, trading):
    wd = 1 if day >= EXPIRY_SWITCH else 3
    cand = day + timedelta(days=(wd - day.weekday()) % 7)
    end = day + timedelta(days=10)
    while cand not in trading:
        cand += timedelta(days=1)
        if cand > end:
            return None
    return cand


def metrics(pnls, days):
    s = pd.Series(pnls)
    if s.empty:
        return dict(n=0, wr=float("nan"), pf=float("nan"), total=0.0, sharpe=float("nan"))
    gw, gl = s[s > 0].sum(), abs(s[s <= 0].sum())
    daily = pd.DataFrame({"d": days, "p": pnls}).groupby("d")["p"].sum()
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        sh = float("nan")
    else:
        r = daily / 50.0
        sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    return dict(
        n=int(len(s)),
        wr=round(float((s > 0).mean() * 100), 1),
        pf=round(float(gw / gl) if gl > 0 else float("inf"), 2),
        total=round(float(s.sum()), 1),
        sharpe=round(sh, 2) if sh == sh else float("nan"),
        n_credit=None,
    )


def main():
    t = pd.read_csv(TRADES)
    t["day"] = pd.to_datetime(t["day"]).dt.date
    t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True).dt.tz_convert("Asia/Kolkata")
    bars = pd.read_pickle(CACHE)
    trading = {pd.Timestamp(x).tz_convert("Asia/Kolkata").date()
               if getattr(pd.Timestamp(x), "tzinfo", None)
               else pd.Timestamp(x).date()
               for x in pd.DatetimeIndex(bars.index).normalize().unique()}
    # pickle index may already be tz-aware IST
    idx = pd.DatetimeIndex(bars.index)
    if idx.tz is None:
        trading = {ts.date() for ts in idx.normalize().unique()}
    else:
        trading = {ts.tz_convert("Asia/Kolkata").date() for ts in idx.normalize().unique()}

    rows = []
    for _, tr in t.iterrows():
        day = tr["day"]
        exp = expiry_date(day, trading)
        dte = (exp - day).days if exp else None
        S0 = float(tr["entry_price"])
        dlt = float(tr["spot_delta"])
        put = tr["direction"] == "SHORT"
        S1 = S0 - dlt if put else S0 + dlt
        et = tr["entry_time"]
        # TARGET/STOP roughly ~1h later; HARD 15:15
        if tr["reason"] == "HARD_EXIT":
            eh, em = 15, 15
        else:
            eh, em = min(et.hour + 1, 15), et.minute
        rec = dict(day=day, direction=tr["direction"], reason=tr["reason"],
                   dte=dte, S0=S0, S1=S1, intrinsic_pnl=float(tr["bf_pnl"]))
        for sig in (0.10, 0.12, 0.15, 0.20):
            T0 = years(dte if dte is not None else 3, (et.hour, et.minute))
            T1 = years(dte if dte is not None else 3, (eh, em))
            n0 = fly_net(S0, T0, sig, put)
            n1 = fly_net(S1, T1, sig, put)
            rec[f"net0_{int(sig*100)}"] = round(n0, 2)
            rec[f"net1_{int(sig*100)}"] = round(n1, 2)
            rec[f"mtm_{int(sig*100)}"] = round(n1 - n0, 2)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "orb_asym_live_mtm_trades.csv", index=False)

    summary = []
    for sig in (10, 12, 15, 20):
        col = f"mtm_{sig}"
        cred = f"net0_{sig}"
        m = metrics(df[col].tolist(), df["day"].tolist())
        m["sigma"] = f"{sig}%"
        m["slice"] = "all"
        m["n_credit"] = int((df[cred] < 0).sum())
        m["pct_credit"] = round(100 * m["n_credit"] / m["n"], 1)
        summary.append(m)
        tgt = df[df["reason"] == "TARGET"]
        m2 = metrics(tgt[col].tolist(), tgt["day"].tolist())
        m2["sigma"] = f"{sig}%"
        m2["slice"] = "TARGET only"
        m2["n_credit"] = int((tgt[cred] < 0).sum())
        m2["pct_credit"] = round(100 * m2["n_credit"] / max(m2["n"], 1), 1)
        summary.append(m2)
        near = df[df["dte"] <= 1]
        m3 = metrics(near[col].tolist(), near["day"].tolist())
        m3["sigma"] = f"{sig}%"
        m3["slice"] = "DTE<=1"
        m3["n_credit"] = int((near[cred] < 0).sum())
        m3["pct_credit"] = round(100 * m3["n_credit"] / max(m3["n"], 1), 1)
        summary.append(m3)
        far = df[df["dte"] > 1]
        m4 = metrics(far[col].tolist(), far["day"].tolist())
        m4["sigma"] = f"{sig}%"
        m4["slice"] = "DTE>1"
        m4["n_credit"] = int((far[cred] < 0).sum())
        m4["pct_credit"] = round(100 * m4["n_credit"] / max(m4["n"], 1), 1)
        summary.append(m4)

    m0 = metrics(df["intrinsic_pnl"].tolist(), df["day"].tolist())
    m0.update(sigma="intrinsic-7.5", slice="all", n_credit=0, pct_credit=0.0)
    summary.insert(0, m0)
    mt = metrics(df.loc[df["reason"] == "TARGET", "intrinsic_pnl"].tolist(),
                 df.loc[df["reason"] == "TARGET", "day"].tolist())
    mt.update(sigma="intrinsic-7.5", slice="TARGET only", n_credit=0, pct_credit=0.0)
    summary.insert(1, mt)

    res = pd.DataFrame(summary)
    cols = ["sigma", "slice", "n", "wr", "pf", "sharpe", "total", "n_credit", "pct_credit"]
    res = res[cols]
    res.to_csv(OUT / "orb_asym_live_mtm_summary.csv", index=False)
    print(res.to_string(index=False))
    print("\n15% TARGET MTM vs intrinsic: "
          f"mtm mean={df.loc[df.reason=='TARGET','mtm_15'].mean():.2f}  "
          f"intr mean={df.loc[df.reason=='TARGET','intrinsic_pnl'].mean():.2f}")
    print("wrote", OUT / "orb_asym_live_mtm_summary.csv")


if __name__ == "__main__":
    main()
