"""Panel features: the same quantified factors for every stock in UNIVERSE.

Rows are (date, ticker). Market-wide factors (macro, rates, aggregate fund
flows ...) are identical across tickers on a date; stock-level factors (own
trend, CTA positioning in the stock, stat-arb residual, microstructure,
earnings calendar) are computed per ticker. One pooled model is trained on
all rows, so the only way it can respond differently per stock is through
stock_id and the stock-level characteristics, which is what we study.
"""
import numpy as np
import pandas as pd

import quant_flows as qf
from config import PANEL_HORIZON, UNIVERSE, YAHOO
from features import _lr, _z, earnings_block, fred_block

# market-wide series that are not themselves in the stock universe
COMMON_SKIP_GROUPS = {"target", "ai_demand"}
COMMON_SKIP = set(UNIVERSE) | {"NVDL", "MTUM", "USMV"}


def common_features(close, fred, idx):
    F, G = {}, {}

    def add(name, s, group):
        F[name] = s.reindex(idx)
        G[name] = group

    for t, (group, _) in YAHOO.items():
        if group in COMMON_SKIP_GROUPS or t in COMMON_SKIP or t not in close:
            continue
        s = close[t]
        if group == "volatility":
            add(f"{t}_lvl_z", _z(s), group)
            add(f"{t}_chg5", _lr(s, 5), group)
            continue
        for n in (1, 5, 20):
            add(f"{t}_ret{n}", _lr(s, n), group)
    add("vix_term", close["^VIX"] / close["^VIX3M"], "volatility")
    spx_rv = np.log(close["^GSPC"]).diff().rolling(20).std() * qf.ANN * 100
    add("vrp", close["^VIX"] - spx_rv, "volatility")

    fred_block(fred, idx, add)

    # aggregate systematic-fund positioning (same for every stock)
    cta_q = qf.cta_trend(close["QQQ"])
    add("cta_ndx_pos", cta_q["pos"], "fund_flows")
    add("cta_ndx_flow5", cta_q["flow5"], "fund_flows")
    vc = qf.vol_control(close["^GSPC"])
    add("volctl_expo", vc["expo"], "fund_flows")
    add("volctl_flow5", vc["flow5"], "fund_flows")
    rp = qf.risk_parity(close["^GSPC"], close["TLT"])
    add("riskpar_lev", rp["lev"], "fund_flows")
    add("riskpar_flow5", rp["flow5"], "fund_flows")
    add("stock_bond_corr", rp["sb_corr"], "fund_flows")
    rm, rv = np.log(close["MTUM"]).diff(), np.log(close["USMV"]).diff()
    add("mom_spread20", (rm - rv).rolling(20).sum(), "fund_flows")
    add("mom_drawdown", close["MTUM"] / close["MTUM"].rolling(60).max() - 1, "fund_flows")
    for k, v in qf.calendar_flows(idx).items():
        add(k, v, "fund_flows")
    return pd.DataFrame(F), G


def stock_features(t, d, close, earnings, idx):
    F, G = {}, {}

    def add(name, s, group):
        F[name] = s.reindex(idx)
        G[name] = group

    c = close[t]
    r1 = np.log(c).diff()
    for n in (1, 5, 20, 60, 120):
        add(f"ret{n}", _lr(c, n), "own_technical")
    add("rv20", r1.rolling(20).std() * qf.ANN, "own_technical")
    add("rv_ratio", r1.rolling(5).std() / r1.rolling(60).std(), "own_technical")
    add("drawdown", c / c.rolling(250, min_periods=60).max() - 1, "own_technical")
    add("dist_ma50", c / c.rolling(50).mean() - 1, "own_technical")

    # peers = equal-weight of the other universe stocks (captures AI-capex / big-tech moves)
    others = [np.log(close[o]).diff() for o in UNIVERSE if o != t]
    peer = pd.concat(others, axis=1).mean(axis=1)
    add("peer_ret5", peer.rolling(5).sum(), "peers")
    add("peer_ret20", peer.rolling(20).sum(), "peers")
    add("rel_peer20", r1.rolling(20).sum() - peer.rolling(20).sum(), "peers")

    # how systematic funds are positioned in *this* stock
    cta = qf.cta_trend(c)
    add("cta_pos", cta["pos"], "stock_positioning")
    add("cta_flow5", cta["flow5"], "stock_positioning")
    for k, v in qf.stat_arb_residual(c, close["QQQ"]).items():
        add(k, v, "stock_positioning")          # resid5_z, resid20_z, sector_beta (vs QQQ)
    add("smh_beta", qf.stat_arb_residual(c, close["SMH"])["sector_beta"], "stock_positioning")
    rm = np.log(close["MTUM"]).diff() - np.log(close["USMV"]).diff()
    add("mom_beta", r1.rolling(60).cov(rm) / rm.rolling(60).var(), "stock_positioning")
    for k, v in qf.microstructure(d["Open"].reindex(idx), d["High"].reindex(idx), d["Low"].reindex(idx),
                                  c, d["Volume"].reindex(idx)).items():
        add(k, v, "stock_positioning")

    earnings_block(idx, earnings, add)
    return pd.DataFrame(F), G


def build_panel(ohlcv, fred, earnings_by_ticker, h=PANEL_HORIZON):
    """Return (X, groups, y_norm, scale, y_raw, close_wide).

    Target is the h-day forward log return divided by the stock's own ex-ante
    h-day vol, so TSLA's noise does not dominate the pooled loss. Predictions
    are converted back to returns with `scale`."""
    idx = ohlcv["NVDA"].index
    close = {t: d["Close"].reindex(idx).ffill(limit=5) for t, d in ohlcv.items()}
    C, G = common_features(close, fred, idx)
    frames, ys, scales = [], [], []
    for i, t in enumerate(UNIVERSE):
        S, Gs = stock_features(t, ohlcv[t], close, earnings_by_ticker[t], idx)
        S["stock_id"] = i
        Gs["stock_id"] = "identity"
        G.update(Gs)
        df = pd.concat([C, S], axis=1)
        df["ticker"] = t
        frames.append(df.set_index("ticker", append=True))
        c = close[t]
        scale = np.log(c).diff().rolling(20).std() * np.sqrt(h)
        y = np.log(c.shift(-h) / c)
        ys.append(pd.Series(y.values, index=frames[-1].index))
        scales.append(pd.Series(scale.values, index=frames[-1].index))
    X = pd.concat(frames).replace([np.inf, -np.inf], np.nan)
    y_raw, scale = pd.concat(ys), pd.concat(scales)
    X.index.names = ["Date", "ticker"]
    y_raw.index.names = scale.index.names = ["Date", "ticker"]
    # rows before a stock has enough history are dropped
    ok = scale.notna() & X["ret120"].notna()
    X, y_raw, scale = X[ok], y_raw[ok], scale[ok]
    close_wide = pd.DataFrame({t: close[t] for t in UNIVERSE})
    return X, pd.Series(G)[X.columns], y_raw / scale, scale, y_raw, close_wide
