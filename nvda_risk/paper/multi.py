"""Alpaca PAPER bot: many stocks x many rules, one account, a virtual book per rule.

At each half-hour boundary every rule in rules.py sets a direction per stock;
directions become shares (rules.NOTIONAL dollars each), the shares of all rules
are netted per stock, and only the net difference to the account position is
traded. Each rule's own P&L is reconstructed afterwards from the log
(paper/multi_eval.py). Everything is flat at close - 1 minute.

  python paper/multi.py                    # trade today's session
  python paper/multi.py --dry-run          # decide and log, send no orders
  python paper/multi.py --simulate HH:MM   # one decision at a past time today, print it, no orders
  python paper/multi.py --flatten          # emergency: close every position now

Credentials: APCA_API_KEY_ID / APCA_API_SECRET_KEY (paper). Stop: touch paper/STOP.
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import rules  # noqa: E402
from bot import DATA_URL, ET, FEED, STOP, TRADE_URL, Alpaca  # noqa: E402  (paper endpoint, retries)

LOGS = HERE / "logs"
HEARTBEAT = LOGS / "heartbeat"
GROSS_CAP = 350_000          # $ cap on the account's total gross position
SYMBOL_CAP = 60_000          # $ cap per stock
UTC = ZoneInfo("UTC")


def _iso(t):
    return t.astimezone(UTC).isoformat().replace("+00:00", "Z")


class Multi(Alpaca):
    def bars(self, symbols, start, end, timeframe="1Min"):
        out, token = {}, None
        while True:
            params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": _iso(start), "end": _iso(end),
                      "feed": FEED, "limit": 10000}
            if token:
                params["page_token"] = token
            d = self._req("GET", f"{DATA_URL}/v2/stocks/bars", params=params) or {}
            for s, bs in (d.get("bars") or {}).items():
                out.setdefault(s, []).extend(bs)
            token = d.get("next_page_token")
            if not token:
                return out

    def positions(self):
        return {p["symbol"]: int(float(p["qty"])) for p in (self._req("GET", f"{TRADE_URL}/v2/positions") or [])}

    def asset(self, s):
        return self._req("GET", f"{TRADE_URL}/v2/assets/{s}") or {}

    def order_sym(self, sym, qty, cid):
        return self._req("POST", f"{TRADE_URL}/v2/orders", json={
            "symbol": sym, "qty": str(abs(qty)), "side": "buy" if qty > 0 else "sell",
            "type": "market", "time_in_force": "day", "client_order_id": cid})

    def close_everything(self):
        return self._req("DELETE", f"{TRADE_URL}/v2/positions", params={"cancel_orders": "true"}) or []


def price_before(bs, t, max_age_min=10):
    """Close of the last minute bar that ended by t (bar time = its start)."""
    best = None
    for b in bs or []:
        start = datetime.fromisoformat(b["t"].replace("Z", "+00:00"))
        if start + timedelta(minutes=1) <= t and start + timedelta(minutes=1) > t - timedelta(minutes=max_age_min):
            best = b
    return float(best["c"]) if best else None


def daily_context(symbols):
    """prev close, sigma, beta from cached/refreshed Yahoo 30m bars (same as the backtest)."""
    import intraday
    ctx, hist = {}, {}
    today = datetime.now(ET).date()
    for s in symbols + [rules.MARKET]:
        try:
            b = intraday.load(s, "30m", refresh=True)
        except Exception as e:
            print("context: skip", s, e, flush=True)
            continue
        b = b[b.index.date < today]
        days = sorted(set(b.index.date))
        if not days:
            continue
        ctx[s] = {"prev": float(b["Close"][b.index.date == days[-1]].iloc[-1])}
        rets = []
        for d in days[-10:]:
            c = b["Close"][b.index.date == d]
            rets.append(c.pct_change().iloc[2:].to_numpy())       # drop the first bar's return, as in the backtest
        hist[s] = np.concatenate(rets) if rets else np.array([])
    for s in symbols:
        if s in hist and rules.MARKET in hist and len(hist[s]) == len(hist[rules.MARKET]):
            sg, bt = rules.sigma_beta(hist[s], hist[rules.MARKET])
            ctx[s].update(sigma=sg, beta=bt)
    return ctx


def snapshot(api, symbols, T, open_t, ctx, open_px):
    bs = api.bars(symbols + [rules.MARKET], T - timedelta(minutes=62), T + timedelta(seconds=1))
    snap = {}
    for s in symbols + [rules.MARKET]:
        b = bs.get(s)
        c = ctx.get(s, {})
        snap[s] = {"p": price_before(b, T), "p30": price_before(b, T - timedelta(minutes=30)),
                   "p60": price_before(b, T - timedelta(minutes=60)), "prev": c.get("prev"),
                   "open": open_px.get(s), "p1000": c.get("p1000"),
                   "sigma": c.get("sigma"), "beta": c.get("beta")}
    return snap


def targets(dirs, snap, shortable):
    """Per-rule shares and the netted, capped account target."""
    per_rule = {}
    for k, dd in dirs.items():
        per_rule[k] = {}
        for s, d in dd.items():
            if d < 0 and not shortable.get(s, False):
                d = 0
            q = rules.shares(d, snap.get(s, {}).get("p"), rules.NOTIONAL[k])
            if q:
                per_rule[k][s] = q
    net = {}
    for k, qs in per_rule.items():
        for s, q in qs.items():
            net[s] = net.get(s, 0) + q
    for s, q in list(net.items()):                                  # per-stock cap
        cap = int(SYMBOL_CAP // snap[s]["p"])
        net[s] = max(min(q, cap), -cap)
    gross = sum(abs(q) * snap[s]["p"] for s, q in net.items())
    scale = min(1.0, GROSS_CAP / gross) if gross else 1.0
    if scale < 1:
        net = {s: int(q * scale) for s, q in net.items()}
    return per_rule, {s: q for s, q in net.items() if q}, gross, scale


def execute(api, net, T, dry_run):
    cur = api.positions()
    syms = set(cur) | set(net)
    phase1, phase2 = [], []
    for s in syms:
        a, b = cur.get(s, 0), net.get(s, 0)
        if a == b:
            continue
        if a != 0 and (b == 0 or np.sign(a) != np.sign(b)):           # close (or close before flipping)
            phase1.append((s, -a))
            if b:
                phase2.append((s, b))
        elif abs(b) < abs(a):
            phase1.append((s, b - a))
        else:
            phase2.append((s, b - a))
    done = []
    for ph, orders in (("1", phase1), ("2", phase2)):
        ids = []
        for s, q in orders:
            if dry_run:
                done.append({"sym": s, "qty": q, "status": "dry_run"})
                continue
            try:
                o = api.order_sym(s, q, f"m-{T:%Y%m%d%H%M}-{s}-{ph}")
                ids.append((s, q, o["id"]))
            except Exception as e:
                done.append({"sym": s, "qty": q, "status": f"error: {str(e)[:120]}"})
        for s, q, oid in ids:                                          # wait for this phase's fills
            o = None
            for _ in range(30):
                o = api.get_order(oid)
                if o and o["status"] in ("filled", "canceled", "rejected", "expired"):
                    break
                time.sleep(1)
            done.append({"sym": s, "qty": q, "id": oid, "status": (o or {}).get("status"),
                         "fill_price": (o or {}).get("filled_avg_price"), "filled_qty": (o or {}).get("filled_qty")})
    return cur, done


def _write(rec):
    LOGS.mkdir(parents=True, exist_ok=True)
    with (LOGS / f"multi_{datetime.now(ET):%Y-%m-%d}.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def _restore_prev():
    p = LOGS / f"multi_{datetime.now(ET):%Y-%m-%d}.jsonl"
    if not p.exists():
        return {}
    last = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    last = [r for r in last if r.get("action") == "decide"]
    return last[-1]["directions"] if last else {}


def _sleep_until(t):
    while True:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(datetime.now(ET).isoformat())
        dt = (t - datetime.now(ET)).total_seconds()
        if dt <= 0 or STOP.exists():
            return
        time.sleep(min(dt, 20))


def decide_once(api, T, open_t, close_t, ctx, open_px, shortable, prev, dry_run):
    symbols = [s for s in rules.UNIVERSE if s in ctx]
    snap = snapshot(api, symbols, T, open_t, ctx, open_px)
    slot = {"minutes_from_open": int((T - open_t).total_seconds() // 60),
            "minutes_to_close": int((close_t - T).total_seconds() // 60),
            "key": f"{T:%Y-%m-%d}T{T:%H:%M}"}
    dirs = rules.decide(slot, snap, prev)
    per_rule, net, gross, scale = targets(dirs, snap, shortable)
    cur, orders = execute(api, net, T, dry_run)
    rec = {"time": T.isoformat(), "action": "decide", "slot": slot, "snap": {s: {"p": d["p"], "p30": d["p30"]}
           for s, d in snap.items()}, "directions": dirs, "per_rule": per_rule, "net_target": net,
           "gross_before_cap": gross, "cap_scale": scale, "positions_before": cur, "orders": orders,
           "equity": float(api.account()["equity"]), "dry_run": dry_run}
    _write(rec)
    n_err = sum(1 for o in orders if str(o.get("status", "")).startswith(("error", "rejected")))
    print(f"{T:%H:%M} decide: {len(net)} stocks, gross ${gross:,.0f} (scale {scale:.2f}), "
          f"{len(orders)} orders, {n_err} errors, equity ${rec['equity']:,.0f}", flush=True)
    return dirs


def run(dry_run=False):
    api = Multi()
    acct = api.account()
    print(f"paper account {acct['account_number']} equity ${float(acct['equity']):,.0f}"
          f"{' (DRY RUN)' if dry_run else ''}", flush=True)
    sess = api.session_today()
    if sess is None:
        print("market closed today", flush=True)
        return
    open_t, close_t = sess
    ctx = daily_context(rules.UNIVERSE)
    shortable = {}
    for s in rules.UNIVERSE:
        a = api.asset(s)
        shortable[s] = bool(a.get("shortable") and a.get("easy_to_borrow"))
    print(f"universe {len([s for s in rules.UNIVERSE if s in ctx])} stocks, "
          f"{sum(shortable.values())} shortable", flush=True)
    grid, t = [], open_t + timedelta(minutes=30)
    while t <= close_t - timedelta(minutes=30):
        grid.append(t)
        t += timedelta(minutes=30)
    prev = _restore_prev()
    open_px = {}
    for T in grid:
        if datetime.now(ET) > T + timedelta(minutes=5):
            continue
        _sleep_until(T + timedelta(seconds=4))
        if STOP.exists():
            break
        if not open_px:                                  # open price and open+30m price, from minute bars
            bs = api.bars(rules.UNIVERSE + [rules.MARKET], open_t, open_t + timedelta(minutes=31))
            for s, b in bs.items():
                if b:
                    open_px[s] = float(b[0]["o"])
                    p = price_before(b, open_t + timedelta(minutes=30))
                    if s in ctx and p:
                        ctx[s]["p1000"] = p
        try:
            prev = decide_once(api, T, open_t, close_t, ctx, open_px, shortable, prev, dry_run)
        except Exception as e:
            print(f"{T:%H:%M} decision failed: {e}", flush=True)
            _write({"time": T.isoformat(), "action": "error", "error": str(e)[:500]})
    _sleep_until(close_t - timedelta(minutes=1))
    flatten(api, dry_run)


def flatten(api, dry_run=False):
    now = datetime.now(ET)
    symbols = rules.UNIVERSE + [rules.MARKET]
    bs = api.bars(symbols, now - timedelta(minutes=5), now)
    marks = {s: price_before(b, now) for s, b in bs.items()}
    cur = api.positions()
    orders = []
    if cur and not dry_run:
        res = api.close_everything()
        time.sleep(5)
        for r in res:
            oid = (r.get("body") or {}).get("id")
            o = api.get_order(oid) if oid else None
            orders.append({"sym": r.get("symbol"), "id": oid, "status": (o or {}).get("status"),
                           "fill_price": (o or {}).get("filled_avg_price"), "filled_qty": (o or {}).get("filled_qty")})
    _write({"time": now.isoformat(), "action": "flatten", "positions_before": cur, "orders": orders,
            "marks": marks, "equity": float(api.account()["equity"]), "dry_run": dry_run})
    print(f"flatten: {len(cur)} positions closed, equity ${float(api.account()['equity']):,.0f}", flush=True)


def simulate(hhmm):
    """One decision at a past time today using real data; prints targets, sends nothing."""
    api = Multi()
    sess = api.session_today()
    if sess is None:
        sys.exit("no session today")
    open_t, close_t = sess
    T = datetime.now(ET).replace(hour=int(hhmm[:2]), minute=int(hhmm[3:]), second=0, microsecond=0)
    ctx = daily_context(rules.UNIVERSE)
    bs = api.bars(rules.UNIVERSE + [rules.MARKET], open_t, open_t + timedelta(minutes=31))
    open_px = {s: float(b[0]["o"]) for s, b in bs.items() if b}
    for s, b in bs.items():
        p = price_before(b, open_t + timedelta(minutes=30))
        if s in ctx and p:
            ctx[s]["p1000"] = p
    symbols = [s for s in rules.UNIVERSE if s in ctx]
    snap = snapshot(api, symbols, T, open_t, ctx, open_px)
    slot = {"minutes_from_open": int((T - open_t).total_seconds() // 60),
            "minutes_to_close": int((close_t - T).total_seconds() // 60), "key": f"{T:%Y-%m-%d}T{T:%H:%M}"}
    dirs = rules.decide(slot, snap, {})
    per_rule, net, gross, scale = targets(dirs, snap, {s: True for s in rules.UNIVERSE})
    missing = [s for s, d in snap.items() if d["p"] is None or d["p30"] is None]
    print(f"simulated {T:%H:%M}: prices missing for {missing or 'none'}")
    for k, qs in per_rule.items():
        long_, short_ = sum(q > 0 for q in qs.values()), sum(q < 0 for q in qs.values())
        print(f"  {k:13s} {long_:2d} long {short_:2d} short")
    print(f"  net: {len(net)} stocks, gross before cap ${gross:,.0f}, scale {scale:.2f}")
    return snap, per_rule, net


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--simulate")
    ap.add_argument("--flatten", action="store_true")
    a = ap.parse_args()
    if a.simulate:
        simulate(a.simulate)
    elif a.flatten:
        flatten(Multi())
    else:
        run(dry_run=a.dry_run)
