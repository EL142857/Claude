"""End-to-end run: data -> factors -> walk-forward forecasts -> compare with reality.

  python run.py                 # use cached data, download what is missing
  python run.py --refresh       # re-download everything
  python run.py --no-ablation   # skip the per-group attribution (faster)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import data
import evaluate as ev
import features
import model
import report
from config import HORIZONS, MAIN_HORIZON, TEST_START

OUT = Path(__file__).parent / "output"
BASELINE_GROUPS = ("own_technical", "events")


def main(refresh=False, ablation=True, ohlcv=None, fred=None, earnings=None, out=OUT):
    out.mkdir(parents=True, exist_ok=True)
    missing = []
    if ohlcv is None:
        ohlcv, fred, earnings, missing = data.load_all(refresh=refresh)
    for t, why in missing:
        print(f"[missing] {t}: {why}")

    X, groups, Y, price = features.build(ohlcv, fred, earnings)
    r1 = np.log(price).diff()
    base_cols = groups.index[groups.isin(BASELINE_GROUPS)]
    print(f"{X.shape[1]} features in {groups.nunique()} groups, {len(X)} days "
          f"({X.index[0].date()} .. {X.index[-1].date()})")

    results = {"horizons": {}, "missing_inputs": [t for t, _ in missing],
               "n_features": int(X.shape[1]), "groups": groups.value_counts().to_dict(),
               "test_start": TEST_START, "last_date": str(X.index[-1].date())}
    forecasts = {}
    for h in HORIZONS:
        y = Y[f"y{h}"]
        full = model.walk_forward(X, y, r1, TEST_START, h)
        base = model.walk_forward(X, y, r1, TEST_START, h, cols=base_cols)
        forecasts[h] = full
        full.to_csv(out / f"forecast_h{h}.csv")
        results["horizons"][h] = {
            "full_model": ev.forecast_metrics(full, price, h),
            "technical_only": ev.forecast_metrics(base, price, h),
        }
        if h == MAIN_HORIZON:
            base_main = base
        print(f"h={h}: " + json.dumps({k: round(v, 4) for k, v in
                                        results['horizons'][h]['full_model'].items()}))

    # --- trading the forecast, with the risk-control layer ---
    fc = forecasts[MAIN_HORIZON]
    pos = model.risk_controlled_position(fc, X, MAIN_HORIZON)
    pos_base = model.risk_controlled_position(base_main, X, MAIN_HORIZON)
    r_next = r1.shift(-1)
    pnl, r = ev.strategy(pos, r_next)
    pnl_base, _ = ev.strategy(pos_base, r_next)
    results["strategy"] = {
        "risk_controlled_model": ev.perf(pnl),
        "technical_only": ev.perf(pnl_base),
        "buy_and_hold": ev.perf(r),
        "avg_abs_position": float(pos.abs().mean()),
    }
    pd.DataFrame({"pos": pos, "pnl": pnl, "pnl_tech": pnl_base, "bh": r}).to_csv(out / "strategy.csv")

    # --- which outside factors carry the signal: drop one group at a time ---
    if ablation:
        y = Y[f"y{MAIN_HORIZON}"]
        ref = results["horizons"][MAIN_HORIZON]["full_model"]
        rows = []
        for g in sorted(groups.unique()):
            if g == "target":
                continue
            cols = groups.index[groups != g]
            m = ev.forecast_metrics(model.walk_forward(X, y, r1, TEST_START, MAIN_HORIZON, cols=cols),
                                    price, MAIN_HORIZON)
            rows.append({"group": g, "n_features": int((groups == g).sum()),
                         "ic_drop": ref["ic_spearman"] - m["ic_spearman"],
                         "r2_drop": ref["oos_r2_vs_random_walk"] - m["oos_r2_vs_random_walk"]})
            print(f"  without {g:14s} IC {m['ic_spearman']:+.4f}")
        results["ablation"] = sorted(rows, key=lambda d: -d["ic_drop"])

    # --- latest live forecast (outcome not known yet) ---
    last = fc.index[-1]
    results["latest_forecast"] = {
        "date": str(last.date()), "price": float(price[last]), "horizon": MAIN_HORIZON,
        "exp_return": float(fc.loc[last, "mu"]),
        "price_mid": float(price[last] * np.exp(fc.loc[last, "mu"])),
        "price_80": [float(price[last] * np.exp(fc.loc[last, "band_q10"])),
                     float(price[last] * np.exp(fc.loc[last, "band_q90"]))],
        "position": float(pos.iloc[-1]) if len(pos) else None,
    }

    quant = X[["cta_nvda_pos", "cta_ndx_pos", "volctl_expo"] +
              (["riskpar_lev"] if "riskpar_lev" in X else [])].loc[TEST_START:]
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    report.build(results, price, fc, pnl, pnl_base, r, quant, out / "report.html")
    print(f"wrote {out / 'report.html'}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--no-ablation", action="store_true")
    a = ap.parse_args()
    main(refresh=a.refresh, ablation=not a.no_ablation)
