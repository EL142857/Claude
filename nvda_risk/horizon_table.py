"""Forecast accuracy as the horizon shortens (NVDA model, same walk-forward setup).

  python horizon_table.py      -> output/horizon_table.csv, output/horizon_table.html

Overlapping h-day windows are not independent, so significance uses the
effective sample size n / h.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import data
import evaluate as ev
import features
import model
from config import TEST_START

HORIZONS = [60, 40, 20, 10, 5, 3, 2, 1]
OUT = Path(__file__).parent / "output"


def row(fc, price, h):
    m = ev.forecast_metrics(fc, price, h)
    f = fc.dropna(subset=["y"])
    n_eff = max(int(len(f) / h), 3)
    k_eff = round(m["hit_rate"] * n_eff)
    p0 = m["hit_rate_always_up"]
    ic = m["ic_spearman"]
    return {
        "horizon": h,
        "n": m["n"],
        "n_eff": n_eff,
        "price_err_model": m["price_mape_model"],
        "price_err_rw": m["price_mape_random_walk"],
        "price_err_gain": 1 - m["price_mape_model"] / m["price_mape_random_walk"],
        "r2_vs_rw": m["oos_r2_vs_random_walk"],
        "r2_vs_drift": m["oos_r2_vs_drift"],
        "hit_rate": m["hit_rate"],
        "hit_always_up": p0,
        "hit_edge": m["hit_rate"] - p0,
        "hit_edge_p": float(stats.binomtest(k_eff, n_eff, p0, alternative="greater").pvalue),
        "share_pred_up": m["share_pred_up"],
        "ic": ic,
        "ic_t_eff": float(ic * np.sqrt(n_eff - 2) / np.sqrt(max(1 - ic ** 2, 1e-9))),
        "coverage_80": m["coverage_80"],
        "coverage_95": m["coverage_95"],
        "band80_width": float((np.exp(f["band_q90"]) - np.exp(f["band_q10"])).mean()),
    }


def main():
    ohlcv, fred, earnings, _ = data.load_all()
    X, groups, _, price = features.build(ohlcv, fred, earnings)
    r1 = np.log(price).diff()
    rows = []
    for h in HORIZONS:
        y = np.log(price.shift(-h) / price)
        fc = model.walk_forward(X, y, r1, TEST_START, h)
        fc.to_csv(OUT / f"horizon_fc_h{h}.csv")
        rows.append(row(fc, price, h))
        r = rows[-1]
        print(f"h={h:2d}  price err {r['price_err_model']:.2%} vs RW {r['price_err_rw']:.2%}  "
              f"hit {r['hit_rate']:.1%} vs up {r['hit_always_up']:.1%}  IC {r['ic']:+.3f}  cov80 {r['coverage_80']:.1%}")
    t = pd.DataFrame(rows).set_index("horizon")
    t.to_csv(OUT / "horizon_table.csv")
    (OUT / "horizon_table.json").write_text(json.dumps({"test_start": TEST_START,
                                                        "last_date": str(X.index[-1].date()),
                                                        "rows": rows}, indent=2))
    import horizon_report
    horizon_report.build(t, TEST_START, str(X.index[-1].date()), OUT / "horizon_table.html")
    print("wrote", OUT / "horizon_table.html")


if __name__ == "__main__":
    main()
