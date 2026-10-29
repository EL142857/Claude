"""1 week down to 1 minute, with a separately selected and trained model per horizon.

For each horizon: choose the model (tuning.CONFIGS) by purged CV on that
horizon's own training period, then run that horizon's own walk-forward with
it. Compared against the earlier run where every horizon used the same
model settings (output/horizon_intraday.csv).

  python horizon_tuned.py   -> output/horizon_tuned.{csv,json,html}
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
import features
import horizon_intraday as hi
import intraday
import model
import tuning
from config import TEST_START

OUT = Path(__file__).parent / "output"


def main():
    rows, models = [], {}
    ohlcv, fred, earnings, _ = data.load_all()
    X, _, _, price = features.build(ohlcv, fred, earnings)
    r1 = np.log(price).diff()
    daily_mu = None
    for h, label in ((5, "1 周(5 个交易日)"), (1, "1 天")):
        y = np.log(price.shift(-h) / price)
        cfg, cv = tuning.select(X, y, pd.Timestamp(TEST_START), purge=h)
        fc = model.walk_forward(X, y, r1, TEST_START, h, config=cfg)
        fc.to_csv(OUT / f"tuned_fc_daily_h{h}.csv")
        if h == 1:
            daily_mu = fc["mu"]
        fcl = fc.dropna(subset=["y"])
        rows.append({"label": label, "data": "日线,2014 起", "config": cfg, "cv_ic": cv[cfg],
                     "n_refits": int(fcl["train_mean"].nunique()), **hi.metrics(fcl, overlap=h)})
        models[label] = cv
        print(label, cfg, {k: round(v, 4) for k, v in cv.items()})

    names = {"60m": "1 小时", "30m": "30 分钟", "15m": "15 分钟", "5m": "5 分钟", "2m": "2 分钟", "1m": "1 分钟"}
    for itv, label in names.items():
        Xi, yi, c, slot = intraday.build(itv, daily_mu=daily_mu, refresh=False)  # same bars as before
        day = pd.Series(Xi.index.normalize(), index=Xi.index)
        cfg, cv = tuning.select(Xi, yi, intraday.test_start(Xi), purge=1, group=day)
        fc = intraday.walk_forward(Xi, yi, slot, config=cfg)
        fc.to_csv(OUT / f"tuned_fc_{itv}.csv")
        days = Xi.index.normalize().nunique()
        rows.append({"label": label, "data": f"{itv} K 线,{days} 个交易日", "config": cfg, "cv_ic": cv[cfg],
                     "n_refits": intraday.N_CHUNKS, **hi.metrics(fc)})
        models[label] = cv
        print(label, cfg, {k: round(v, 4) for k, v in cv.items()})
    rows.append({"label": "30 秒", "data": "无数据:Yahoo 最细只到 1 分钟,需要逐笔成交数据"})

    t = pd.DataFrame(rows)
    shared = pd.read_csv(OUT / "horizon_intraday.csv").set_index("label")
    t["ic_shared"] = t["label"].map(shared["ic"])
    t["hit_edge_shared"] = t["label"].map(shared["hit_edge"])
    t.to_csv(OUT / "horizon_tuned.csv", index=False)
    (OUT / "horizon_tuned.json").write_text(json.dumps({"rows": rows, "cv_scores": models}, indent=2,
                                                       default=float, ensure_ascii=False))
    import horizon_intraday_report
    horizon_intraday_report.build(t, hi.COST_BP, OUT / "horizon_tuned.html")
    print("wrote", OUT / "horizon_tuned.html")


if __name__ == "__main__":
    main()
