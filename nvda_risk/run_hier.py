"""Do stock-specific factor responses survive out of sample?

Compares, under the same walk-forward protocol as run_panel.py:
  common    : one slope per factor for all stocks (c = 0)
  hier      : common slope + shrunk per-stock deviation (c chosen by CV)
  separate  : an independent ridge per stock
  lgbm      : the pooled LightGBM from run_panel.py
then bootstraps the hierarchical model's scenario responses.

  python run_hier.py           (run run_panel.py first)
  python run_hier.py --reuse   reuse the saved CV choice and forecasts, redo the bootstrap
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import data
import panel
import panel_hier as ph
from config import PANEL_HORIZON, TEST_START, UNIVERSE

OUT = Path(__file__).parent / "output" / "panel"
H = PANEL_HORIZON


def score(fc):
    """Per-stock and cross-sectional out-of-sample scores for one variant."""
    f = fc.dropna(subset=["y"])
    per = {}
    for t, g in f.groupby(level="ticker"):
        gm = g.droplevel("ticker")
        mic = gm.groupby(gm.index.to_period("M")).apply(lambda x: stats.spearmanr(x["mu"], x["y"]).statistic)
        per[t] = {"ic": float(stats.spearmanr(g["mu"], g["y"]).statistic),
                  "monthly_ic_t": float(mic.mean() / mic.std() * np.sqrt(len(mic)))}
    yn = f["y"] / f["scale"]
    r2 = 1 - ((yn - f["pred_norm"]) ** 2).sum() / (yn ** 2).sum()
    mu, y = f["mu"].unstack("ticker"), f["y"].unstack("ticker")
    xs = pd.Series({d: stats.spearmanr(mu.loc[d], y.loc[d]).statistic for d in mu.index}).dropna()
    xs_n = xs.iloc[::H]
    return {"per_stock": per,
            "avg_ic": float(np.mean([v["ic"] for v in per.values()])),
            "n_stocks_ic_pos": int(sum(v["ic"] > 0 for v in per.values())),
            "oos_r2_norm": float(r2),
            "xs_rank_ic": float(xs.mean()),
            "xs_rank_ic_t": float(xs_n.mean() / xs_n.std() * np.sqrt(len(xs_n)))}


def main(out=OUT, reuse=False):
    ohlcv, fred, nvda_earn, _ = data.load_all(extra=UNIVERSE)
    earnings = {t: data.load_earnings_for(t) for t in UNIVERSE}
    earnings["NVDA"] = earnings["NVDA"].union(nvda_earn)
    X, groups, y_norm, scale, y_raw, _ = panel.build_panel(ohlcv, fred, earnings)
    d = X.index.get_level_values("Date")

    if reuse:
        old = json.loads((out / "hier.json").read_text())
        sel = {"alpha": old["alpha"], "c": old["c_cv"], "sep_alpha": old["sep_alpha"]}
        cv_by_c, scores = {float(k): v for k, v in old["cv_mse_by_c"].items()}, old["scores"]
        sel["sep_grid"] = {0: old["cv_mse_separate"]}
    else:
        sel, cv_by_c, scores = _select_and_score(X, y_norm, scale, y_raw, out)
    _final(X, y_norm, scale, d, sel, cv_by_c, scores, out)


def _select_and_score(X, y_norm, scale, y_raw, out):
    sel = ph.select(X, y_norm, TEST_START, H)
    print(f"CV choice: alpha={sel['alpha']:.0f}, c={sel['c']}  (separate alpha={sel['sep_alpha']:.0f})")
    cv_by_c = {c: min(v for (a, cc), v in sel["grid"].items() if cc == c) for c in ph.CS}
    print("  best CV MSE by c:", {c: round(v, 5) for c, v in cv_by_c.items()})

    alpha_common = min((v, a) for (a, c), v in sel["grid"].items() if c == 0)[1]
    variants = {
        "common": lambda: ph.Hier(alpha_common, 0.0),
        "hier": lambda: ph.Hier(sel["alpha"], sel["c"]),
        "separate": lambda: ph.Separate(sel["sep_alpha"]),
    }
    scores = {}
    for name, make in variants.items():
        fc = ph.walk_forward(make, X, y_norm, scale, y_raw, TEST_START, H)
        fc.to_csv(out / f"hier_forecast_{name}.csv")
        scores[name] = score(fc)
        print(f"  {name:9s} avg IC {scores[name]['avg_ic']:+.4f}  R2 {scores[name]['oos_r2_norm']:+.4f}  "
              f"xs IC {scores[name]['xs_rank_ic']:+.4f} (t {scores[name]['xs_rank_ic_t']:.2f})")
    lg = pd.read_csv(out / "forecast.csv", index_col=[0, 1], parse_dates=[0])
    scores["lgbm"] = score(lg)
    print(f"  lgbm      avg IC {scores['lgbm']['avg_ic']:+.4f}  R2 {scores['lgbm']['oos_r2_norm']:+.4f}")
    return sel, cv_by_c, scores


def _final(X, y_norm, scale, d, sel, cv_by_c, scores, out):

    # final hierarchical fit on all labelled data, scenario responses + bootstrap.
    # Use at least a small c so the stock-specific part is visible even if CV picked 0.
    c_show = sel["c"] if sel["c"] > 0 else 0.25
    lab = y_norm.notna()
    avg_scale = scale[d >= TEST_START].groupby(level="ticker").mean()
    m = ph.Hier(sel["alpha"], c_show).fit(X[lab], y_norm[lab])
    resp = ph.scenario_from_slopes(m.slopes(), m.sd, avg_scale)[UNIVERSE]
    draws = ph.bootstrap(X[lab], y_norm[lab], sel["alpha"], c_show, avg_scale)
    arr = np.stack([tot[UNIVERSE].to_numpy() for tot, _ in draws])      # draws x scen x stock
    dev = np.stack([sp[UNIVERSE].to_numpy() for _, sp in draws])        # stock-specific part only
    lo, hi = np.percentile(arr, [5, 95], axis=0)
    dlo, dhi = np.percentile(dev, [5, 95], axis=0)
    frame = lambda a: pd.DataFrame(a, index=resp.index, columns=UNIVERSE)
    for k, v in (("hier_response", resp), ("hier_lo", frame(lo)), ("hier_hi", frame(hi)),
                 ("hier_dev_lo", frame(dlo)), ("hier_dev_hi", frame(dhi))):
        v.to_csv(out / f"{k}.csv")
    sig_dev = int(((dlo > 0) | (dhi < 0)).sum())
    sig_tot = int(((lo > 0) | (hi < 0)).sum())

    emp = pd.read_csv(out / "empirical_beta.csv", index_col=0)[UNIVERSE]
    demean = lambda x: x.sub(x.mean(axis=1), axis=0)
    pat = float(stats.spearmanr(demean(resp).stack(), demean(emp.loc[resp.index]).stack()).statistic)
    res = {
        "alpha": sel["alpha"], "c_cv": sel["c"], "c_shown": c_show, "sep_alpha": sel["sep_alpha"],
        "cv_mse_by_c": {str(k): v for k, v in cv_by_c.items()},
        "cv_mse_separate": min(sel["sep_grid"].values()),
        "scores": scores, "n_boot": len(draws), "n_cells": int(resp.size),
        "sig_total_cells": sig_tot, "sig_stock_specific_cells": sig_dev,
        "pattern_vs_history": pat,
    }
    (out / "hier.json").write_text(json.dumps(res, indent=2, default=float))
    print(f"  bootstrap: {sig_tot}/{resp.size} cells with 90% CI excluding 0; "
          f"{sig_dev} with a stock-specific part excluding 0; pattern vs history {pat:+.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    main(reuse=ap.parse_args().reuse)
