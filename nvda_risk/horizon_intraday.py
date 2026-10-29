"""One table: forecast accuracy from 1 week down to 1 minute (and 30 seconds, if
tick data is ever added).

  python horizon_intraday.py   -> output/horizon_intraday.{csv,html}

Rows for 1 week / 1 day reuse the daily model's walk-forward forecasts from
horizon_table.py; intraday rows come from intraday.py. All rows use the same
metrics. Overlapping multi-day windows use an effective sample n / h.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import intraday

OUT = Path(__file__).parent / "output"
COST_BP = 0.5      # rough round-trip cost of crossing NVDA's ~1 cent spread, in bp


def metrics(fc, overlap=1):
    y, mu = fc["y"], fc["mu"]
    n = len(y)
    n_eff = max(int(n / overlap), 3)
    nz = y != 0
    hits = (np.sign(mu) == np.sign(y))[nz]
    up = float((y[nz] > 0).mean())
    base = max(up, 1 - up)
    hit = float(hits.mean())
    k_eff = round(hit * n_eff)
    ic = float(stats.spearmanr(mu, y).statistic)
    return {
        "n": n, "n_eff": n_eff,
        "typical_move_bp": float(y.abs().mean() * 1e4),
        "price_err_model": float(np.abs(np.exp(mu - y) - 1).mean()),
        "price_err_rw": float(np.abs(np.exp(-y) - 1).mean()),
        "hit_rate": hit, "hit_base": base, "hit_edge": hit - base,
        "hit_edge_p": float(stats.binomtest(k_eff, n_eff, base, alternative="greater").pvalue),
        "ic": ic, "ic_t": float(ic * np.sqrt(n_eff - 2) / np.sqrt(max(1 - ic ** 2, 1e-9))),
        "r2_vs_drift": float(1 - ((y - mu) ** 2).sum() / ((y - fc["train_mean"]) ** 2).sum()),
        "coverage_80": float(((y >= fc["band_q10"]) & (y <= fc["band_q90"])).mean()),
        "coverage_95": float(((y >= fc["band_q025"]) & (y <= fc["band_q975"])).mean()),
        "band80_half_bp": float((fc["band_q90"] - fc["band_q10"]).mean() / 2 * 1e4),
        # trade the sign of the forecast for one horizon, gross of costs, minus always-long
        "edge_per_trade_bp": float(((np.sign(mu) * y).mean() - y.mean()) * 1e4),
        "test_window": f"{fc.index.min():%Y-%m-%d} 至 {fc.index.max():%Y-%m-%d}",
    }


def main():
    rows = []
    for h, label in ((5, "1 周(5 个交易日)"), (1, "1 天")):
        fc = pd.read_csv(OUT / f"horizon_fc_h{h}.csv", index_col=0, parse_dates=True).dropna(subset=["y"])
        rows.append({"label": label, "data": "日线,2014 起", **metrics(fc, overlap=h)})
        print(label, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)})

    daily_fc = pd.read_csv(OUT / "horizon_fc_h1.csv", index_col=0, parse_dates=True)["mu"]
    names = {"60m": "1 小时", "30m": "30 分钟", "15m": "15 分钟", "5m": "5 分钟", "2m": "2 分钟", "1m": "1 分钟"}
    for itv, label in names.items():
        X, y, c, slot = intraday.build(itv, daily_mu=daily_fc)
        fc = intraday.walk_forward(X, y, slot)
        fc.to_csv(OUT / f"intraday_fc_{itv}.csv")
        days = X.index.normalize().nunique()
        rows.append({"label": label, "data": f"{itv} K 线,{days} 个交易日", **metrics(fc)})
        print(label, {k: round(v, 4) for k, v in rows[-1].items() if isinstance(v, float)})
    rows.append({"label": "30 秒", "data": "无数据:Yahoo 最细只到 1 分钟,需要逐笔成交数据"})

    t = pd.DataFrame(rows)
    t.to_csv(OUT / "horizon_intraday.csv", index=False)
    (OUT / "horizon_intraday.json").write_text(json.dumps(rows, indent=2, default=float, ensure_ascii=False))
    import horizon_intraday_report
    horizon_intraday_report.build(t, COST_BP, OUT / "horizon_intraday.html")
    print("wrote", OUT / "horizon_intraday.html")


if __name__ == "__main__":
    main()
