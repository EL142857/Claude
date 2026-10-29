"""Alpaca PAPER trading bot for the 30-minute short-term reversal rule on NVDA.

Rule (zoo.MODELS["rule_reversal"], no fitting): at each half-hour boundary T
from 10:30 to 15:30 ET, look at the last 30 minutes' return r = p(T)/p(T-30m) - 1
and hold -sign(r) x NOTIONAL of NVDA until T+30m. Flat at 15:59, never overnight.

  python paper/bot.py              # trade today's session on the paper account
  python paper/bot.py --dry-run    # same loop, log decisions, send no orders
  python paper/bot.py --replay     # check the decision logic against the backtest
  python paper/bot.py --flatten    # emergency: close the NVDA position now

Credentials: APCA_API_KEY_ID / APCA_API_SECRET_KEY (paper keys) from the
environment. The trading host is fixed to the paper endpoint.
Stop at any time by creating paper/STOP.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

TRADE_URL = "https://paper-api.alpaca.markets"     # paper only; never the live endpoint
DATA_URL = "https://data.alpaca.markets"
SYMBOL = "NVDA"
NOTIONAL = float(os.environ.get("PAPER_NOTIONAL", "10000"))   # $ per position
MAX_SHARES_FACTOR = 1.5                                     # cap: 1.5 x notional
FEED = os.environ.get("ALPACA_FEED", "iex")                  # free plan: IEX
ET = ZoneInfo("America/New_York")
HERE = Path(__file__).parent
LOGS = HERE / "logs"
STOP = HERE / "STOP"


# ---------- pure logic (shared with --replay) ----------

def boundaries(open_t, close_t):
    """Decision times for one session: every 30 minutes from open+60m to close-30m.
    open+30m only records a price (the first bar's return is not used, as in the backtest)."""
    out, t = [], open_t + timedelta(minutes=60)
    while t <= close_t - timedelta(minutes=30):
        out.append(t)
        t += timedelta(minutes=30)
    return out


def target_shares(p_prev, p_now, notional=NOTIONAL):
    """Reversal: short after a rise, long after a fall, flat if unchanged."""
    if not (p_prev and p_now) or p_now == p_prev:
        return 0
    side = -1 if p_now > p_prev else 1
    return side * int(notional // p_now)


# ---------- Alpaca REST ----------

class Alpaca:
    def __init__(self):
        kid, sec = os.environ.get("APCA_API_KEY_ID"), os.environ.get("APCA_API_SECRET_KEY")
        if not (kid and sec):
            sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys) in the environment.")
        self.s = requests.Session()
        self.s.headers.update({"APCA-API-KEY-ID": kid, "APCA-API-SECRET-KEY": sec})

    def _req(self, method, url, **kw):
        for attempt in range(4):
            try:
                r = self.s.request(method, url, timeout=15, **kw)
                if r.status_code == 404:
                    return None
                if r.status_code == 429 or r.status_code >= 500:
                    raise requests.HTTPError(f"{r.status_code} {r.text[:200]}")
                r.raise_for_status()
                return r.json() if r.content else {}
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                if attempt == 3:
                    raise
                time.sleep(2 ** attempt)

    def account(self):
        return self._req("GET", f"{TRADE_URL}/v2/account")

    def session_today(self):
        today = datetime.now(ET).date().isoformat()
        cal = self._req("GET", f"{TRADE_URL}/v2/calendar", params={"start": today, "end": today})
        if not cal or cal[0]["date"] != today:
            return None
        o, c = cal[0]["open"], cal[0]["close"]
        mk = lambda hm: datetime.fromisoformat(f"{today}T{hm}").replace(tzinfo=ET)
        return mk(o), mk(c)

    def position(self):
        p = self._req("GET", f"{TRADE_URL}/v2/positions/{SYMBOL}")
        return int(float(p["qty"])) if p else 0

    def price_at(self, t):
        """Close of the 1-minute bar that ends at t (i.e. the last price before t)."""
        start = (t - timedelta(minutes=1)).astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
        end = t.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
        d = self._req("GET", f"{DATA_URL}/v2/stocks/{SYMBOL}/bars",
                      params={"timeframe": "1Min", "start": start, "end": end, "feed": FEED, "limit": 5})
        bars = (d or {}).get("bars") or []
        if bars:
            return float(bars[-1]["c"]), "bar"
        tr = self._req("GET", f"{DATA_URL}/v2/stocks/{SYMBOL}/trades/latest", params={"feed": FEED})
        return (float(tr["trade"]["p"]), "last_trade") if tr else (None, "none")

    def order(self, qty, client_id):
        side = "buy" if qty > 0 else "sell"
        return self._req("POST", f"{TRADE_URL}/v2/orders", json={
            "symbol": SYMBOL, "qty": str(abs(qty)), "side": side, "type": "market",
            "time_in_force": "day", "client_order_id": client_id})

    def get_order(self, oid):
        return self._req("GET", f"{TRADE_URL}/v2/orders/{oid}")

    def close_all(self):
        return self._req("DELETE", f"{TRADE_URL}/v2/positions/{SYMBOL}")


# ---------- live loop ----------

FIELDS = ["time", "action", "p_prev", "p_now", "src", "ret_30m", "pos_before", "target", "order_qty",
          "order_id", "fill_price", "fill_qty", "status", "equity"]


def _log(row):
    LOGS.mkdir(parents=True, exist_ok=True)
    path = LOGS / f"{datetime.now(ET):%Y-%m-%d}.csv"
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})
    print(json.dumps(row, default=str), flush=True)


HEARTBEAT = HERE / "logs" / "heartbeat"


def _sleep_until(t):
    while True:
        HEARTBEAT.parent.mkdir(parents=True, exist_ok=True)
        HEARTBEAT.write_text(datetime.now(ET).isoformat())
        dt = (t - datetime.now(ET)).total_seconds()
        if dt <= 0:
            return
        if STOP.exists():
            return
        time.sleep(min(dt, 30))


def _fill(api, oid, wait=20):
    for _ in range(wait):
        o = api.get_order(oid)
        if o and o["status"] in ("filled", "canceled", "rejected", "expired"):
            return o
        time.sleep(1)
    return api.get_order(oid)


def run(dry_run=False):
    api = Alpaca()
    acct = api.account()
    print(f"paper account {acct['account_number']} equity ${float(acct['equity']):,.0f} "
          f"{'(DRY RUN)' if dry_run else ''}", flush=True)
    sess = api.session_today()
    if sess is None:
        print("market closed today", flush=True)
        return
    open_t, close_t = sess
    flat_t = close_t - timedelta(minutes=1)
    for T in boundaries(open_t, close_t):
        if datetime.now(ET) > T + timedelta(minutes=5):
            continue                                   # restarted late: skip passed decisions
        _sleep_until(T + timedelta(seconds=3))         # let the minute bar ending at T publish
        if STOP.exists():
            break
        p_prev, _ = api.price_at(T - timedelta(minutes=30))
        p_now, src = api.price_at(T)
        tgt = target_shares(p_prev, p_now)
        cur = api.position()
        tgt = max(min(tgt, int(MAX_SHARES_FACTOR * NOTIONAL // p_now)), -int(MAX_SHARES_FACTOR * NOTIONAL // p_now))
        row = {"time": T.isoformat(), "action": "decide", "p_prev": p_prev, "p_now": p_now, "src": src,
               "ret_30m": (p_now / p_prev - 1) if p_prev else "", "pos_before": cur, "target": tgt,
               "order_qty": tgt - cur}
        if tgt != cur and not dry_run:
            o = api.order(tgt - cur, f"rev-{T:%Y%m%d%H%M}")
            o = _fill(api, o["id"])
            row.update(order_id=o["id"], fill_price=o.get("filled_avg_price"), fill_qty=o.get("filled_qty"),
                       status=o["status"])
        row["equity"] = api.account()["equity"]
        _log(row)
    _sleep_until(flat_t)
    cur = api.position()
    row = {"time": datetime.now(ET).isoformat(), "action": "flatten", "pos_before": cur, "target": 0}
    if cur and not dry_run:
        o = api.close_all()
        if o and o.get("id"):
            o = _fill(api, o["id"])
            row.update(order_id=o["id"], fill_price=o.get("filled_avg_price"), fill_qty=o.get("filled_qty"),
                       status=o["status"])
    row["equity"] = api.account()["equity"]
    _log(row)


# ---------- replay: the live decision logic against the backtest ----------

def replay():
    """Apply boundaries() + target_shares() to cached 30m bars and check every
    decision matches the backtest's rule_reversal sign (study30 setup A)."""
    sys.path.insert(0, str(HERE.parent))
    import numpy as np
    import pandas as pd
    import intraday

    bars = intraday.load("NVDA", "30m", refresh=False)
    close_at = {ts + timedelta(minutes=30): c for ts, c in bars["Close"].items()}    # price at bar end
    preds = pd.read_csv(HERE.parent / "output" / "study30_preds_A.csv", parse_dates=["ts"]).set_index("ts")
    n = agree = 0
    pnl = []
    for day, g in bars.groupby(bars.index.normalize()):
        o = day + timedelta(hours=9, minutes=30)
        c = day + timedelta(hours=16)
        for T in boundaries(o, c):
            T = T.replace(tzinfo=None)
            p_prev, p_now = close_at.get(T - timedelta(minutes=30)), close_at.get(T)
            p_next = close_at.get(T + timedelta(minutes=30))
            if p_prev is None or p_now is None or p_next is None:
                continue
            side = np.sign(target_shares(p_prev, p_now, notional=1e9))
            bar_ts = T - timedelta(minutes=30)                      # backtest row = bar ending at T
            if bar_ts in preds.index and pd.notna(preds.loc[bar_ts, "rule_reversal"]):   # backtest's test half
                n += 1
                agree += int(side == np.sign(preds.loc[bar_ts, "rule_reversal"]))
            pnl.append(side * math.log(p_next / p_now))
    print(f"decisions matched to backtest rows: {n}, same direction: {agree} ({agree / max(n, 1):.1%})")
    print(f"replay over all {len(pnl)} decisions: mean {np.mean(pnl) * 1e4:+.2f}bp per trade, "
          f"hit {np.mean(np.array(pnl) > 0):.1%}")
    return n, agree


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--replay", action="store_true")
    ap.add_argument("--flatten", action="store_true")
    a = ap.parse_args()
    if a.replay:
        replay()
    elif a.flatten:
        api = Alpaca()
        cur = api.position()
        o = api.close_all() if cur else None
        _log({"time": datetime.now(ET).isoformat(), "action": "manual_flatten", "pos_before": cur, "target": 0,
              "order_id": (o or {}).get("id", ""), "equity": api.account()["equity"]})
    else:
        run(dry_run=a.dry_run)
