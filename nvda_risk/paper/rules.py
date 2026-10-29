"""Intraday rules on a half-hour grid, shared by the live bot and the backtest.

Each rule is a pure function of a snapshot of prices at a decision time T and
returns {symbol: direction} with direction in {-1, 0, +1}. Positions are held
until the next decision time; everything is flat by the close. No rule is fitted.

Snapshot fields per symbol (None when unavailable):
  p      price at T (close of the minute / bar ending at T)
  p30    price at T-30m        p60   price at T-60m
  prev   previous session close    open  today's open
  p1000  price at open+30m (10:00 on a normal day)
  sigma  std of 30m returns over the last 10 sessions (first bar excluded)
  beta   beta of 30m returns to QQQ over the last 10 sessions
"""
import hashlib

import numpy as np

MARKET = "QQQ"
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "ADBE", "CRM", "ORCL", "INTC", "QCOM", "MU", "TXN", "CSCO", "IBM", "JPM",
    "BAC", "WFC", "GS", "MS", "V", "MA", "UNH", "JNJ", "LLY", "PFE",
    "MRK", "ABBV", "XOM", "CVX", "WMT", "COST", "HD", "KO", "PEP", "MCD",
    "DIS", "BA", "CAT", "PLTR", "UBER",
]

# dollars per (rule, stock) position
NOTIONAL = {"rev30": 1000, "resid_rev30": 1000, "xs_rev30": 2000, "big_rev30": 1000,
            "rev60": 1000, "intraday_mom": 1000, "gap_fade": 1000, "placebo": 1000}

LABEL = {"rev30": "30 分钟反转", "resid_rev30": "残差反转(剔除 QQQ)", "xs_rev30": "截面反转(多跌最多/空涨最多)",
         "big_rev30": "大幅波动才反转", "rev60": "60 分钟反转", "intraday_mom": "开盘动量→最后半小时",
         "gap_fade": "跳空回补", "placebo": "随机(安慰剂)"}


def _ret(a, b):
    return a / b - 1 if a and b else None


def _sign(x):
    return 0 if x is None or x == 0 else (1 if x > 0 else -1)


def rev30(slot, snap):
    if slot["minutes_from_open"] < 60:          # first bar's return is not used (as in the backtest)
        return {}
    return {s: -_sign(_ret(d["p"], d["p30"])) for s, d in snap.items() if s != MARKET}


def resid_rev30(slot, snap):
    if slot["minutes_from_open"] < 60:
        return {}
    rm = _ret(snap.get(MARKET, {}).get("p"), snap.get(MARKET, {}).get("p30"))
    out = {}
    for s, d in snap.items():
        r = _ret(d["p"], d["p30"])
        if s == MARKET or r is None or rm is None or d.get("beta") is None:
            continue
        out[s] = -_sign(r - d["beta"] * rm)
    return out


def xs_rev30(slot, snap, frac=0.2):
    if slot["minutes_from_open"] < 60:
        return {}
    r = {s: _ret(d["p"], d["p30"]) for s, d in snap.items() if s != MARKET}
    r = {s: v for s, v in r.items() if v is not None}
    if len(r) < 10:
        return {}
    k = max(1, int(len(r) * frac))
    order = sorted(r, key=r.get)
    out = {s: 0 for s in r}
    out.update({s: 1 for s in order[:k]})       # fell most -> long
    out.update({s: -1 for s in order[-k:]})     # rose most -> short
    return out


def big_rev30(slot, snap, z=1.5):
    if slot["minutes_from_open"] < 60:
        return {}
    out = {}
    for s, d in snap.items():
        r = _ret(d["p"], d["p30"])
        if s == MARKET or r is None or not d.get("sigma"):
            continue
        out[s] = -_sign(r) if abs(r) > z * d["sigma"] else 0
    return out


def rev60(slot, snap):
    m = slot["minutes_from_open"]
    if m < 90 or m % 60 != 30:                  # 11:00, 12:00, ... (on the hour); keep position at :30
        return None if m >= 90 else {}
    return {s: -_sign(_ret(d["p"], d["p60"])) for s, d in snap.items() if s != MARKET}


def intraday_mom(slot, snap):
    if slot["minutes_to_close"] != 30:          # only the last half hour
        return {s: 0 for s in snap if s != MARKET}
    return {s: _sign(_ret(d.get("p1000"), d.get("prev"))) for s, d in snap.items() if s != MARKET}


def gap_fade(slot, snap, min_gap=0.005):
    if slot["minutes_from_open"] != 30:         # enter at open+30m, out at the next decision
        return {s: 0 for s in snap if s != MARKET}
    out = {}
    for s, d in snap.items():
        g = _ret(d.get("open"), d.get("prev"))
        if s == MARKET or g is None:
            continue
        out[s] = -_sign(g) if abs(g) > min_gap else 0
    return out


def placebo(slot, snap):
    """Random +/-1, reproducible from (date, time, symbol). Expected P&L = minus costs."""
    if slot["minutes_from_open"] < 60:
        return {}
    out = {}
    for s in snap:
        if s == MARKET:
            continue
        h = hashlib.sha256(f"{slot['key']}|{s}".encode()).digest()[0]
        out[s] = 1 if h % 2 else -1
    return out


RULES = {"rev30": rev30, "resid_rev30": resid_rev30, "xs_rev30": xs_rev30, "big_rev30": big_rev30,
         "rev60": rev60, "intraday_mom": intraday_mom, "gap_fade": gap_fade, "placebo": placebo}


def decide(slot, snap, prev_targets):
    """Directions for every rule. A rule returning None keeps its previous
    directions; {} means flat."""
    out = {}
    for name, fn in RULES.items():
        d = fn(slot, snap)
        out[name] = dict(prev_targets.get(name, {})) if d is None else d
    return out


def shares(direction, price, notional):
    if not direction or not price:
        return 0
    return direction * max(1, int(round(notional / price)))


def sigma_beta(r_stock, r_mkt):
    """From aligned arrays of past 30m returns (first bars removed)."""
    r_stock, r_mkt = np.asarray(r_stock, float), np.asarray(r_mkt, float)
    ok = np.isfinite(r_stock) & np.isfinite(r_mkt)
    if ok.sum() < 30:
        return None, None
    rs, rm = r_stock[ok], r_mkt[ok]
    return float(rs.std()), float(np.cov(rs, rm)[0, 1] / rm.var())
