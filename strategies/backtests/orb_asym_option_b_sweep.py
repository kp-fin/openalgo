"""Option B overlay: skip BS fly if entry net <= 0. Flatten-on-fail is not a path test."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "orb_asym"
df = pd.read_csv(OUT / "orb_asym_live_mtm_trades.csv")
df["day"] = pd.to_datetime(df["day"]).dt.date


def metrics(sub, col):
    if sub.empty:
        return dict(n=0, wr=float("nan"), pf=float("nan"), total=0.0, sharpe=float("nan"),
                    n_target=0, n_stop=0, n_hard=0)
    s = sub[col]
    gw, gl = s[s > 0].sum(), abs(s[s <= 0].sum())
    daily = sub.groupby("day")[col].sum()
    if len(daily) < 2 or daily.std(ddof=1) == 0:
        sh = float("nan")
    else:
        r = daily / 50.0
        sh = float(r.mean() / r.std(ddof=1) * np.sqrt(252))
    return dict(
        n=int(len(sub)),
        wr=round(float((s > 0).mean() * 100), 1),
        pf=round(float(gw / gl) if gl > 0 else float("inf"), 2),
        total=round(float(s.sum()), 1),
        sharpe=round(sh, 2) if sh == sh else float("nan"),
        n_target=int((sub.reason == "TARGET").sum()),
        n_stop=int((sub.reason == "STOP").sum()),
        n_hard=int((sub.reason == "HARD_EXIT").sum()),
    )


rows = []
m = metrics(df, "intrinsic_pnl")
m.update(sigma="intrinsic-7.5", rule="locked v2 (cannot skip credits)")
rows.append(m)

for sig in (10, 12, 15, 20):
    n0, mtm = f"net0_{sig}", f"mtm_{sig}"
    a = metrics(df, mtm)
    a.update(sigma=f"{sig}%", rule="B off")
    rows.append(a)
    kept = df[df[n0] > 0]
    skipped = int((df[n0] <= 0).sum())
    b = metrics(kept, mtm)
    b.update(sigma=f"{sig}%", rule=f"B on skip net<=0 skipped={skipped}")
    rows.append(b)
    bi = metrics(kept, "intrinsic_pnl")
    bi.update(sigma=f"{sig}%", rule=f"B on, intrinsic of kept skipped={skipped}")
    rows.append(bi)

res = pd.DataFrame(rows)
res.to_csv(OUT / "orb_asym_option_b_sweep.csv", index=False)
print(res.to_string(index=False))
for sig in (10, 12):
    kept = df[df[f"net0_{sig}"] > 0]
    print(f"\nsigma {sig}% kept n={len(kept)} DTE:\n", kept["dte"].value_counts().sort_index().to_string())
    cut = pd.Timestamp("2024-01-01").date()
    print("H1", metrics(kept[kept.day < cut], f"mtm_{sig}"))
    print("H2", metrics(kept[kept.day >= cut], f"mtm_{sig}"))
