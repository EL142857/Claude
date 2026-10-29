"""Per-rule results from the multi-bot logs (virtual book per rule).

  python paper/multi_eval.py [YYYY-MM-DD ...]   (default: every day logged)

For each rule and each held interval: P&L = shares x (price at next decision -
price at this decision), at the bot's decision prices. Execution cost is then
charged from the real fills: when a rule's own share change has the same sign
as the netted order for that stock, it pays that order's slippage per share
(fill - decision price); trades that crossed with other rules inside the
netting pay nothing. The account's equity change is printed next to the sum
of all rules as a check.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402

LOGS = HERE / "logs"


def _slips(rec):
    """symbol -> (order direction, slippage per share; >0 = paid)."""
    out = {}
    marks = rec.get("marks") or {s: d.get("p") for s, d in (rec.get("snap") or {}).items()}
    for o in rec.get("orders", []):
        fp, q, s = o.get("fill_price"), o.get("qty"), o["sym"]
        if q is None and o.get("filled_qty"):
            q = -rec["positions_before"].get(s, 0)                       # flatten orders
        if fp is None or not q or not marks.get(s):
            continue
        out[s] = (np.sign(q), (float(fp) - marks[s]) * np.sign(q))
    return out


def evaluate_day(path):
    recs = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    dec = [r for r in recs if r["action"] == "decide"]
    flat = next((r for r in recs if r["action"] == "flatten"), None)
    if not dec:
        return None, None
    seq = dec + ([flat] if flat else [])
    rows = []
    prev_q = {k: {} for k in rules.RULES}
    for i, r in enumerate(seq):
        slip = _slips(r)
        for k in rules.RULES:
            q_now = (r.get("per_rule") or {}).get(k, {}) if r["action"] == "decide" else {}
            # cost of moving from prev_q to q_now
            cost = 0.0
            for s in set(q_now) | set(prev_q[k]):
                dq = q_now.get(s, 0) - prev_q[k].get(s, 0)
                if dq and s in slip and np.sign(dq) == slip[s][0]:
                    cost += abs(dq) * slip[s][1]
            # P&L of the position held from this decision to the next one
            if i + 1 < len(seq) and r["action"] == "decide":
                nxt = seq[i + 1]
                p1map = nxt.get("marks") or {s: d.get("p") for s, d in nxt["snap"].items()}
                for s, q in q_now.items():
                    p0, p1 = r["snap"].get(s, {}).get("p"), p1map.get(s)
                    if p0 and p1:
                        rows.append({"rule": k, "time": r["time"], "sym": s, "q": q,
                                     "pnl": q * (p1 - p0), "ret": np.sign(q) * (p1 / p0 - 1),
                                     "notional": abs(q) * p0, "cost": 0.0})
            if cost:
                rows.append({"rule": k, "time": r["time"], "sym": "_cost", "q": 0, "pnl": 0.0,
                             "ret": np.nan, "notional": 0.0, "cost": cost})
            prev_q[k] = q_now
    df = pd.DataFrame(rows)
    eq = (dec[0]["equity"], (flat or dec[-1])["equity"])
    return df, eq


def summarise(days):
    out = []
    for p in days:
        df, eq = evaluate_day(p)
        if df is None:
            continue
        day = p.stem.replace("multi_", "")
        pos = df[df["sym"] != "_cost"]
        tot = 0.0
        print(f"\n{day}   account equity {eq[0]:,.0f} -> {eq[1]:,.0f} ({eq[1] - eq[0]:+,.0f})")
        print(f"  {'rule':26s} {'positions':>9s} {'gross bp':>9s} {'hit':>6s} {'gross $':>9s} {'cost $':>8s} {'net $':>9s}")
        for k in rules.RULES:
            g = pos[pos["rule"] == k]
            c = df[df["rule"] == k]["cost"].sum()
            net = g["pnl"].sum() - c
            tot += net
            out.append({"day": day, "rule": k, "positions": len(g), "gross_bp": g["ret"].mean() * 1e4 if len(g) else np.nan,
                        "hit": (g["ret"] > 0).mean() if len(g) else np.nan, "gross": g["pnl"].sum(), "cost": c, "net": net})
            print(f"  {rules.LABEL[k]:26s} {len(g):9d} {out[-1]['gross_bp']:+9.2f} {out[-1]['hit']:6.1%} "
                  f"{g['pnl'].sum():+9.1f} {c:8.1f} {net:+9.1f}")
        print(f"  sum of rules (net) {tot:+,.1f}  vs account {eq[1] - eq[0]:+,.1f}  "
              "(difference = share rounding, caps, and orders that crossed inside the netting)")
    return pd.DataFrame(out)


if __name__ == "__main__":
    wanted = sys.argv[1:]
    files = sorted(LOGS.glob("multi_*.jsonl"))
    if wanted:
        files = [f for f in files if f.stem.replace("multi_", "") in wanted]
    if not files:
        print("no multi-bot logs yet")
    else:
        res = summarise(files)
        if res["day"].nunique() > 1:
            print("\nall days:")
            print(res.groupby("rule")[["positions", "gross", "cost", "net"]].sum().round(1).to_string())
