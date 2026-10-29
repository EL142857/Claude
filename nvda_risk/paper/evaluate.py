"""Summarise paper-trading logs: real fills vs. the backtest's assumed prices.

  python paper/evaluate.py      -> prints a per-day and overall summary

"Signal P&L" uses the decision-time prices (what the backtest assumes);
"fill P&L" uses Alpaca's actual fill prices. The gap is slippage.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LOGS = Path(__file__).parent / "logs"


def load():
    files = sorted(LOGS.glob("*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York")
    return df.sort_values("time")


def per_trade(df):
    """One row per held position: from one decision to the next (or the flatten)."""
    rows = []
    for day, g in df.groupby(df["time"].dt.date):
        g = g.reset_index(drop=True)
        for i in range(len(g) - 1):
            a, b = g.iloc[i], g.iloc[i + 1]
            if a["action"] != "decide" or a["target"] == 0:
                continue
            side = np.sign(a["target"])
            sig_ret = side * np.log(b["p_now"] / a["p_now"]) if b["action"] == "decide" else np.nan
            fa, fb = a.get("fill_price"), b.get("fill_price")
            fill_ret = side * np.log(fb / fa) if pd.notna(fa) and pd.notna(fb) else np.nan
            slip = side * (fa / a["p_now"] - 1) if pd.notna(fa) else np.nan     # >0 = paid up
            rows.append({"day": day, "entry": a["time"], "side": int(side), "shares": abs(int(a["target"])),
                         "signal_bp": sig_ret * 1e4, "fill_bp": fill_ret * 1e4, "entry_slip_bp": slip * 1e4})
    return pd.DataFrame(rows)


def main():
    df = load()
    if df.empty:
        print("no paper-trading logs yet")
        return
    t = per_trade(df)
    eq = df.groupby(df["time"].dt.date)["equity"].agg(["first", "last"])
    print("per day:")
    for day, g in t.groupby("day"):
        print(f"  {day}  trades {len(g):2d}  hit {np.mean(g['fill_bp'] > 0):.0%}  "
              f"signal {g['signal_bp'].mean():+.2f}bp  fills {g['fill_bp'].mean():+.2f}bp  "
              f"entry slippage {g['entry_slip_bp'].mean():+.2f}bp  "
              f"equity {eq.loc[day, 'first']:,.0f} -> {eq.loc[day, 'last']:,.0f}")
    print(f"overall: {len(t)} trades over {t['day'].nunique()} days, "
          f"mean signal {t['signal_bp'].mean():+.2f}bp, mean fill {t['fill_bp'].mean():+.2f}bp, "
          f"hit {np.mean(t['fill_bp'].dropna() > 0):.1%}, backtest expectation ≈ +3.8bp")


if __name__ == "__main__":
    main()
