"""
EMA Regime Crossover -- Daily Circuit Breaker Trade Log, Charges Overlay

Applies the same NSE equity intraday (MIS) charges model used in
ema_regime_crossover_backtest_4pos_35k.py to the CURRENT LIVE CONFIG's trade
log (6 concurrent positions, Rs 25,000/trade = 10% of Rs 2,50,000 allocated
capital, 5x leverage, both circuit-breaker mechanics active -- portfolio-wide
2% daily halt + per-symbol same-day loss-block).

Reuses ema_regime_crossover_daily_circuit_breaker_trades.csv (8,759 trades,
already generated 2026-07-28 by ema_regime_crossover_backtest_daily_circuit_
breaker.py) rather than re-fetching 5 years of data -- the trade SELECTION
(which candidates the breaker/cap accept or reject) is unchanged; this only
overlays realistic transaction costs on top to get net P&L and a net-of-
charges drawdown, matching how ema_regime_crossover_backtest_4pos_35k.py
isolated the sizing effect for the 4-position/Rs 35,000 variant.

Charges model (NSE equity intraday / MIS, Dhan) -- identical to the 4pos_35k
script:
  - Brokerage: Rs 20 flat or 0.03% of turnover, whichever lower, per executed order
  - STT: 0.025% on sell-side turnover only (intraday equity)
  - Exchange transaction charges (NSE): 0.00297% of turnover (buy + sell)
  - SEBI turnover fee: 0.0001% of turnover (buy + sell)
  - Stamp duty: 0.003% on buy-side turnover only
  - GST: 18% on (brokerage + exchange charges + SEBI fee)
"""

import os

import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
TRADES_CSV = os.path.join(BASE, "ema_regime_crossover_daily_circuit_breaker",
                           "ema_regime_crossover_daily_circuit_breaker_trades.csv")
ALLOCATED_CAPITAL = 250_000

BROKERAGE_FLAT = 20.0
BROKERAGE_PCT = 0.0003     # 0.03%
STT_PCT = 0.00025          # 0.025%, sell-side only
EXCHANGE_PCT = 0.0000297   # 0.00297%, both sides
SEBI_PCT = 0.000001        # 0.0001%, both sides
STAMP_DUTY_PCT = 0.00003   # 0.003%, buy-side only
GST_PCT = 0.18


def compute_charges(entry_price, exit_price, qty):
    buy_turnover = entry_price * qty
    sell_turnover = exit_price * qty
    total_turnover = buy_turnover + sell_turnover

    brokerage = 2 * min(BROKERAGE_FLAT, BROKERAGE_PCT * buy_turnover)
    stt = STT_PCT * sell_turnover
    exchange_chg = EXCHANGE_PCT * total_turnover
    sebi_chg = SEBI_PCT * total_turnover
    stamp_duty = STAMP_DUTY_PCT * buy_turnover
    gst = GST_PCT * (brokerage + exchange_chg + sebi_chg)

    total = brokerage + stt + exchange_chg + sebi_chg + stamp_duty + gst
    return {
        "brokerage": round(brokerage, 2),
        "stt": round(stt, 2),
        "exchange_chg": round(exchange_chg, 2),
        "sebi_chg": round(sebi_chg, 2),
        "stamp_duty": round(stamp_duty, 2),
        "gst": round(gst, 2),
        "total_charges": round(total, 2),
    }


df = pd.read_csv(TRADES_CSV)
df["exit_time"] = pd.to_datetime(df["exit_time"])

charges = df.apply(lambda r: compute_charges(r["entry_price"], r["exit_price"], r["qty"]), axis=1)
charges_df = pd.DataFrame(list(charges))
df = pd.concat([df, charges_df], axis=1)
df["net_pnl_rupees"] = df["pnl_rupees"] - df["total_charges"]

n = len(df)
wr_gross = (df["pnl_rupees"] > 0).mean() * 100
wr_net = (df["net_pnl_rupees"] > 0).mean() * 100

total_gross = df["pnl_rupees"].sum()
total_charges = df["total_charges"].sum()
total_net = df["net_pnl_rupees"].sum()

gw = df[df["pnl_rupees"] > 0]["pnl_rupees"].sum()
gl = abs(df[df["pnl_rupees"] <= 0]["pnl_rupees"].sum())
pf_gross = gw / gl if gl > 0 else float("inf")

nw = df[df["net_pnl_rupees"] > 0]["net_pnl_rupees"].sum()
nl = abs(df[df["net_pnl_rupees"] <= 0]["net_pnl_rupees"].sum())
pf_net = nw / nl if nl > 0 else float("inf")

print(f"=== EMA Regime Crossover -- Current Live Config (6 pos / Rs 25,000/trade / both breakers) + Charges ===")
print(f"Trades: {n} | WR (gross): {wr_gross:.1f}% | WR (net): {wr_net:.1f}%")
print(f"\nGross P&L : Rs {total_gross:+,.0f}  | PF (gross): {pf_gross:.2f}")
print(f"Charges   : Rs {total_charges:,.0f}  ({total_charges/n:.0f}/trade avg, {total_charges/abs(total_gross)*100:.1f}% of gross P&L)")
print(f"Net P&L   : Rs {total_net:+,.0f}  | PF (net):   {pf_net:.2f}")

print(f"\nCharges breakdown:")
print(f"  Brokerage     : Rs {df['brokerage'].sum():,.0f}")
print(f"  STT           : Rs {df['stt'].sum():,.0f}")
print(f"  Exchange chg  : Rs {df['exchange_chg'].sum():,.0f}")
print(f"  SEBI chg      : Rs {df['sebi_chg'].sum():,.0f}")
print(f"  Stamp duty    : Rs {df['stamp_duty'].sum():,.0f}")
print(f"  GST           : Rs {df['gst'].sum():,.0f}")


def _pf(g, col):
    w = g.loc[g[col] > 0, col].sum()
    losses = -g.loc[g[col] < 0, col].sum()
    return w / losses if losses > 0 else float("inf")


print("\nBy direction (net of charges):")
for d, g in df.groupby("direction"):
    print(f"  {d}: n={len(g)}, WR={((g.net_pnl_rupees>0).mean()*100):.1f}%, "
          f"net_pnl={g.net_pnl_rupees.sum():+,.0f}, PF(net)={_pf(g, 'net_pnl_rupees'):.2f}, "
          f"PF(gross)={_pf(g, 'pnl_rupees'):.2f}")

# ---- Drawdown, net of charges (the realistic equity curve) ----
dd_df = df.sort_values("exit_time").reset_index(drop=True)
dd_df["cum_pnl_gross"] = dd_df["pnl_rupees"].cumsum()
dd_df["cum_pnl_net"] = dd_df["net_pnl_rupees"].cumsum()
dd_df["running_peak_net"] = dd_df["cum_pnl_net"].cummax()
dd_df["drawdown_net"] = dd_df["cum_pnl_net"] - dd_df["running_peak_net"]
max_dd_net = dd_df["drawdown_net"].min()
max_dd_idx = dd_df["drawdown_net"].idxmin()
peak_before = dd_df.loc[:max_dd_idx, "cum_pnl_net"].idxmax()
after = dd_df.loc[max_dd_idx:]
recovery = after[after["cum_pnl_net"] >= dd_df.loc[peak_before, "cum_pnl_net"]]

print(f"\n=== Drawdown (net of charges) ===")
print(f"Max drawdown (net): Rs {max_dd_net:,.0f} ({abs(max_dd_net)/ALLOCATED_CAPITAL*100:.1f}% of allocated capital)")
print(f"Drawdown window: {dd_df.loc[peak_before, 'exit_time']} -> {dd_df.loc[max_dd_idx, 'exit_time']}")
if len(recovery):
    print(f"Recovered by: {recovery.iloc[0]['exit_time']}")
else:
    print("Never recovered by end of backtest")

out_path = os.path.join(BASE, "ema_regime_crossover_daily_circuit_breaker",
                         "ema_regime_crossover_daily_circuit_breaker_trades_with_charges.csv")
df.to_csv(out_path, index=False)
print(f"\nTrade log with charges -> {out_path}")
