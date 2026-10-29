"""Turn raw series into a dated feature matrix, every column tagged by group.

Alignment rule: a row dated t only contains information known at the US close
on day t. FRED series are shifted by their publication lag.
"""
import numpy as np
import pandas as pd

import quant_flows as qf
from config import FRED, HORIZONS, TARGET, YAHOO


def _lr(s, n):
    return np.log(s / s.shift(n))


def _z(s, n=250):
    return (s - s.rolling(n, min_periods=n // 2).mean()) / s.rolling(n, min_periods=n // 2).std()


def fred_block(fred, idx, add):
    """Rates, credit and liquidity features, each lagged by its publication delay."""
    if len(fred):
        fr = fred.copy()
        fr.index = pd.to_datetime(fr.index)
        fr = fr.reindex(fr.index.union(idx)).ffill().reindex(idx)
        for s, (group, _, lag) in FRED.items():
            if s in fr:
                fr[s] = fr[s].shift(lag)
        if {"DGS10", "DGS2"} <= set(fr):
            add("curve_10y2y", fr["DGS10"] - fr["DGS2"], "rates")
        for s in ("DGS10", "DFII10", "T10YIE", "DGS2"):
            if s in fr:
                add(f"{s}_chg5", fr[s].diff(5), "rates")
                add(f"{s}_chg20", fr[s].diff(20), "rates")
        if "DFII10" in fr:
            add("real_yield_lvl", fr["DFII10"], "rates")
        if "BAMLH0A0HYM2" in fr:
            add("hy_oas_z", _z(fr["BAMLH0A0HYM2"]), "credit")
            add("hy_oas_chg5", fr["BAMLH0A0HYM2"].diff(5), "credit")
        if "BAA10Y" in fr:
            add("baa_spread_z", _z(fr["BAA10Y"]), "credit")
            add("baa_spread_chg5", fr["BAA10Y"].diff(5), "credit")
        if "NFCI" in fr:
            add("nfci", fr["NFCI"], "credit")
        if {"WALCL", "RRPONTSYD", "WTREGEN"} <= set(fr):
            # Net liquidity = Fed balance sheet - reverse repo - TGA (all $bn)
            net = fr["WALCL"] / 1e3 - fr["RRPONTSYD"] - fr["WTREGEN"] / 1e3
            add("net_liq_chg20", net.diff(20) / net.shift(20), "liquidity")
            add("net_liq_z", _z(net), "liquidity")


def earnings_block(idx, earnings, add):
    """Earnings calendar (dates are announced in advance, so not leakage)."""
    pos = pd.Series(np.arange(len(idx)), index=idx)
    e_pos = pos.reindex(pd.DatetimeIndex(earnings)[pd.DatetimeIndex(earnings).isin(idx)]).values
    nxt = np.array([e_pos[e_pos >= p].min() if (e_pos >= p).any() else np.nan for p in pos.values])
    days_to = pd.Series(nxt - pos.values, index=idx)
    add("days_to_earnings", days_to.clip(upper=70).fillna(70), "events")
    for h in HORIZONS:
        add(f"earnings_in_next{h}", (days_to < h).astype(int), "events")


def build(ohlcv, fred, earnings):
    nv = ohlcv[TARGET]
    idx = nv.index
    close = {t: d["Close"].reindex(idx).ffill(limit=5) for t, d in ohlcv.items()}
    c = close[TARGET]
    F, G = {}, {}

    def add(name, series, group):
        F[name] = series.reindex(idx)
        G[name] = group

    r1 = np.log(c).diff()

    # --- NVDA's own state (the baseline the external factors must beat) ---
    for n in (1, 5, 20, 60, 120):
        add(f"nvda_ret{n}", _lr(c, n), "own_technical")
    add("nvda_rv20", r1.rolling(20).std() * qf.ANN, "own_technical")
    add("nvda_rv_ratio", r1.rolling(5).std() / r1.rolling(60).std(), "own_technical")
    add("nvda_drawdown", c / c.rolling(250, min_periods=60).max() - 1, "own_technical")
    add("nvda_dist_ma50", c / c.rolling(50).mean() - 1, "own_technical")

    # --- Every other market series: short and medium returns, relative strength ---
    for t, (group, _) in YAHOO.items():
        if t in (TARGET, "NVDL") or t not in close:
            continue
        s = close[t]
        if group == "volatility":
            add(f"{t}_lvl_z", _z(s), group)
            add(f"{t}_chg5", _lr(s, 5), group)
            continue
        add(f"{t}_ret1", _lr(s, 1), group)
        add(f"{t}_ret5", _lr(s, 5), group)
        add(f"{t}_ret20", _lr(s, 20), group)
        if group in ("semis", "supply_chain", "ai_demand"):
            add(f"nvda_vs_{t}_rs20", _lr(c, 20) - _lr(s, 20), group)

    # --- Volatility structure ---
    if "^VIX" in close and "^VIX3M" in close:
        add("vix_term", close["^VIX"] / close["^VIX3M"], "volatility")
    if "^VIX" in close:
        spx_rv = np.log(close["^GSPC"]).diff().rolling(20).std() * qf.ANN * 100
        add("vrp", close["^VIX"] - spx_rv, "volatility")  # variance risk premium / hedging demand

    fred_block(fred, idx, add)

    # --- Quant / systematic fund behaviour ---
    cta_n = qf.cta_trend(c)
    add("cta_nvda_pos", cta_n["pos"], "quant_flows")
    add("cta_nvda_flow5", cta_n["flow5"], "quant_flows")
    cta_q = qf.cta_trend(close["QQQ"])
    add("cta_ndx_pos", cta_q["pos"], "quant_flows")
    add("cta_ndx_flow5", cta_q["flow5"], "quant_flows")
    vc = qf.vol_control(close["^GSPC"])
    add("volctl_expo", vc["expo"], "quant_flows")
    add("volctl_flow5", vc["flow5"], "quant_flows")
    if "TLT" in close:
        rp = qf.risk_parity(close["^GSPC"], close["TLT"])
        add("riskpar_lev", rp["lev"], "quant_flows")
        add("riskpar_flow5", rp["flow5"], "quant_flows")
        add("stock_bond_corr", rp["sb_corr"], "quant_flows")
    nvdl = ohlcv.get("NVDL")
    lev = qf.leveraged_etf_rebalance(
        c, None if nvdl is None else nvdl["Close"].reindex(idx),
        nv["Volume"], None if nvdl is None else nvdl["Volume"].reindex(idx),
    )
    add("letf_rebal", lev, "quant_flows")
    add("letf_rebal5", lev.rolling(5).sum(), "quant_flows")
    if {"MTUM", "USMV"} <= set(close):
        for k, v in qf.momentum_crowding(close["MTUM"], close["USMV"], c).items():
            add(k, v, "quant_flows")
    for k, v in qf.stat_arb_residual(c, close["SMH"]).items():
        add(k, v, "quant_flows")
    for k, v in qf.calendar_flows(idx).items():
        add(k, v, "quant_flows")
    for k, v in qf.microstructure(nv["Open"], nv["High"], nv["Low"], nv["Close"], nv["Volume"]).items():
        add(k, v, "quant_flows")

    earnings_block(idx, earnings, add)

    X = pd.DataFrame(F).replace([np.inf, -np.inf], np.nan)
    groups = pd.Series(G)

    # --- Targets: forward log returns from close t ---
    Y = pd.DataFrame({f"y{h}": np.log(c.shift(-h) / c) for h in HORIZONS})
    return X, groups, Y, c
