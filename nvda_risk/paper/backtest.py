"""Backtest every rule in rules.py on Yahoo 30-minute bars (last ~60 days),
through the same decision code the live bot uses.

  python paper/backtest.py   -> paper/backtest_results.json + printed table

Costs: COST_BP of traded notional per side. Significance: t-stat of daily
P&L (one observation per day, so intraday correlation is not double counted).
"""
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import intraday  # noqa: E402
import rules  # noqa: E402

COST_BP = 0.5


def load_bars():
    out = {}
    for s in rules.UNIVERSE + [rules.MARKET]:
        try:
            b = intraday.load(s, "30m", refresh=not (intraday.RAW / f"intra_30m_{s}.csv").exists())
            out[s] = b
        except Exception as e:
            print("skip", s, e)
    return out


def day_tables(bars):
    """Per symbol: dict day -> Series of prices at each bar END time, plus open."""
    ends, opens = {}, {}
    for s, b in bars.items():
        e = b["Close"].copy()
        e.index = e.index + timedelta(minutes=30)
        ends[s] = {d: g for d, g in e.groupby(e.index.normalize())}
        opens[s] = b["Open"].groupby(b.index.normalize()).first()
    return ends, opens


def run():
    bars = load_bars()
    ends, opens = day_tables(bars)
    days = sorted(set.intersection(*[set(v) for v in ends.values()]))
    rec = {k: [] for k in rules.RULES}             # (day, ret, notional_traded_fraction)
    daily = {k: {} for k in rules.RULES}
    hist = {s: [] for s in bars}                   # rolling 30m returns for sigma/beta
    for i, day in enumerate(days):
        # sigma / beta from the previous 10 sessions
        sb = {}
        for s in bars:
            if s == rules.MARKET:
                continue
            past = [r for r in hist[s][-10:]]
            pm = [r for r in hist[rules.MARKET][-10:]]
            if len(past) == 10:
                sb[s] = rules.sigma_beta(np.concatenate(past), np.concatenate(pm))
        open_t = day + timedelta(hours=9, minutes=30)
        close_t = max(ends["NVDA"][day].index)          # 16:00, or 13:00 on half days
        grid = []
        t = open_t + timedelta(minutes=30)
        while t <= close_t - timedelta(minutes=30):
            grid.append(t)
            t += timedelta(minutes=30)
        prev_day = days[i - 1] if i else None
        prev_t = {}
        pos = {k: {} for k in rules.RULES}               # rule -> sym -> shares
        pnl_day = {k: 0.0 for k in rules.RULES}
        for j, T in enumerate(grid):
            snap = {}
            for s in bars:
                e = ends[s].get(day)
                if e is None:
                    continue
                g = lambda t: float(e.get(t)) if t in e.index else None
                snap[s] = {"p": g(T), "p30": g(T - timedelta(minutes=30)), "p60": g(T - timedelta(minutes=60)),
                           "prev": float(ends[s][prev_day].iloc[-1]) if prev_day in ends[s] else None,
                           "open": float(opens[s].get(day)) if day in opens[s].index else None,
                           "p1000": g(open_t + timedelta(minutes=30)),
                           "sigma": sb.get(s, (None, None))[0], "beta": sb.get(s, (None, None))[1]}
            slot = {"minutes_from_open": int((T - open_t).total_seconds() // 60),
                    "minutes_to_close": int((close_t - T).total_seconds() // 60),
                    "key": f"{day:%Y-%m-%d}T{T:%H:%M}"}
            dirs = rules.decide(slot, snap, prev_t)
            prev_t = dirs
            T2 = grid[j + 1] if j + 1 < len(grid) else close_t
            for k, dd in dirs.items():
                new = {}
                for s, d in dd.items():
                    p, p2 = snap.get(s, {}).get("p"), ends[s][day].get(T2) if s in ends else None
                    if not d or p is None or p2 is None or np.isnan(p2):
                        continue
                    q = rules.shares(d, p, rules.NOTIONAL[k])
                    new[s] = q
                    r = np.sign(q) * (p2 / p - 1)
                    rec[k].append((day, r, s))
                    pnl_day[k] += q * (p2 - p)
                # trading cost on the change in shares
                traded = sum(abs(new.get(s, 0) - pos[k].get(s, 0)) * (snap.get(s, {}).get("p") or 0)
                             for s in set(new) | set(pos[k]))
                pnl_day[k] -= traded * COST_BP / 1e4
                pos[k] = new
        for k in rules.RULES:                            # flatten cost at the close
            last = {s: ends[s][day].iloc[-1] for s in pos[k] if day in ends[s]}
            pnl_day[k] -= sum(abs(q) * last.get(s, 0) for s, q in pos[k].items()) * COST_BP / 1e4
            daily[k][day] = pnl_day[k]
        for s in bars:                                   # update history with today's 30m returns
            e = ends[s].get(day)
            if e is not None and len(e) > 2:
                hist[s].append(e.pct_change().iloc[2:].to_numpy())   # drop the first bar's return
    out = {}
    half = days[len(days) // 2]
    for k in rules.RULES:
        r = pd.DataFrame(rec[k], columns=["day", "ret", "sym"])
        dp = pd.Series(daily[k])
        h1, h2 = dp[dp.index < half], dp[dp.index >= half]
        out[k] = {"label": rules.LABEL[k], "positions": int(len(r)),
                  "gross_bp": float(r["ret"].mean() * 1e4) if len(r) else np.nan,
                  "hit": float((r["ret"] > 0).mean()) if len(r) else np.nan,
                  "daily_pnl_mean": float(dp.mean()), "daily_pnl_t": float(dp.mean() / dp.std() * np.sqrt(len(dp))),
                  "sharpe_ann": float(dp.mean() / dp.std() * np.sqrt(252)),
                  "pnl_first_half": float(h1.sum()), "pnl_second_half": float(h2.sum()),
                  "days": len(dp), "daily": {f"{d:%Y-%m-%d}": float(v) for d, v in dp.items()},
                  "by_symbol_bp": {s: float(v * 1e4) for s, v in r.groupby("sym")["ret"].mean().items()}}
    return out, days


def main():
    out, days = run()
    print(f"{len(days)} days {days[0]:%Y-%m-%d} .. {days[-1]:%Y-%m-%d}, cost {COST_BP}bp/side")
    print(f"{'rule':14s} {'positions':>9s} {'gross bp':>9s} {'hit':>6s} {'$/day net':>10s} {'t':>6s} "
          f"{'Sharpe':>7s} {'1st half $':>11s} {'2nd half $':>11s}")
    for k, v in out.items():
        print(f"{k:14s} {v['positions']:9d} {v['gross_bp']:+9.2f} {v['hit']:6.1%} {v['daily_pnl_mean']:+10.1f} "
              f"{v['daily_pnl_t']:+6.2f} {v['sharpe_ann']:+7.2f} {v['pnl_first_half']:+11.0f} {v['pnl_second_half']:+11.0f}")
    (HERE / "backtest_results.json").write_text(json.dumps(
        {"days": [f"{d:%Y-%m-%d}" for d in days], "cost_bp": COST_BP, "rules": out}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
