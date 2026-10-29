"""Pooled model over the mega-cap tech universe and how each stock reacts to factor shocks.

  python run_panel.py           # full run
  python run_panel.py --reuse   # reuse saved walk-forward forecasts, redo the analysis
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import data
import evaluate as ev
import panel
import panel_model as pm
import panel_report
from config import PANEL_HORIZON, SCENARIOS, TEST_START, UNIVERSE

OUT = Path(__file__).parent / "output" / "panel"
H = PANEL_HORIZON


def cross_section(fc, close_wide, top=3):
    """Does the model rank the stocks against each other? Daily rank IC and a
    top-3 / bottom-3 portfolio rebalanced daily, against the equal-weight basket."""
    mu = fc["mu"].unstack("ticker")
    y = fc["y"].unstack("ticker")
    ic = pd.Series({d: stats.spearmanr(mu.loc[d], y.loc[d]).statistic
                    for d in mu.index if y.loc[d].notna().sum() >= 5}).dropna()
    ic_nonoverlap = ic.iloc[::H]
    r_next = (close_wide.shift(-1) / close_wide - 1).reindex(mu.index)[mu.columns]
    rank = mu.rank(axis=1)
    n = mu.notna().sum(axis=1)
    longs = rank.gt(n - top, axis=0).astype(float) / top
    shorts = rank.le(top, axis=0).astype(float) / top
    w_ls, w_eq = longs - shorts, mu.notna().astype(float).div(n, axis=0)

    def pnl(w):
        gross = (w * r_next).sum(axis=1, min_count=1)
        cost = w.diff().abs().sum(axis=1).fillna(w.abs().sum(axis=1)) * ev.COST_BP / 1e4
        return (gross - cost).dropna()

    p = {"long_short": pnl(w_ls), "long_top3": pnl(longs), "equal_weight": pnl(w_eq)}
    return {
        "rank_ic_mean": float(ic.mean()),
        "rank_ic_tstat_nonoverlap": float(ic_nonoverlap.mean() / ic_nonoverlap.std() * np.sqrt(len(ic_nonoverlap))),
        "rank_ic_hit": float((ic > 0).mean()),
        "perf": {k: ev.perf(v) for k, v in p.items()},
    }, p


def main(out=OUT, reuse=False):
    out.mkdir(parents=True, exist_ok=True)
    ohlcv, fred, nvda_earn, missing = data.load_all(extra=UNIVERSE)
    earnings = {t: data.load_earnings_for(t) for t in UNIVERSE}
    earnings["NVDA"] = earnings["NVDA"].union(nvda_earn)
    X, groups, y_norm, scale, y_raw, close_wide = panel.build_panel(ohlcv, fred, earnings)
    d = X.index.get_level_values("Date")
    print(f"panel: {len(X)} rows, {X.shape[1]} features, {len(UNIVERSE)} stocks, "
          f"{d.min().date()} .. {d.max().date()}")

    # 1) out-of-sample: is one pooled model any good, per stock?
    if reuse and (out / "forecast.csv").exists():
        fc = pd.read_csv(out / "forecast.csv", index_col=[0, 1], parse_dates=[0])
    else:
        fc = pm.walk_forward(X, y_norm, scale, y_raw, TEST_START, H)
        fc.to_csv(out / "forecast.csv")
    per_stock = {}
    for t in UNIVERSE:
        f = fc.xs(t, level="ticker")
        per_stock[t] = ev.forecast_metrics(f, close_wide[t], H)
        print(f"  {t:5s} IC {per_stock[t]['ic_spearman']:+.3f}  hit {per_stock[t]['hit_rate']:.3f} "
              f"(always-up {per_stock[t]['hit_rate_always_up']:.3f})  cov80 {per_stock[t]['coverage_80']:.3f}")
    xs, pnls = cross_section(fc, close_wide)
    pd.DataFrame(pnls).to_csv(out / "cross_section_pnl.csv")
    print(f"  cross-sectional rank IC {xs['rank_ic_mean']:+.4f} (t={xs['rank_ic_tstat_nonoverlap']:.2f})")

    # 2) the one model, trained on everything labelled, and a pre-2024 twin for stability
    final = pm.fit_final(X, y_norm)
    early = pm.fit_final(X, y_norm, until=TEST_START)
    tr = y_norm.notna()
    sd, mean = X[tr].std(), X[tr].mean()
    Xa = X[d >= TEST_START]                     # analyse responses under recent conditions

    shares = pm.shap_shares(final, Xa, groups)
    resp = pm.scenarios(final, Xa, scale, sd)
    resp_early = pm.scenarios(early, Xa, scale, sd)
    emp_b, emp_t = pm.empirical(X, y_raw, sd, mean)
    for k, v in (("shap_shares", shares), ("scenario_response", resp), ("scenario_response_pre2024", resp_early),
                 ("empirical_beta", emp_b), ("empirical_t", emp_t)):
        v.to_csv(out / f"{k}.csv")

    def _corr(a, b):
        a, b = a.stack(), b.reindex_like(a).stack()
        return float(stats.spearmanr(a, b).statistic)

    # agreement of the cross-stock *pattern* within each scenario (de-meaned rows)
    demean = lambda m: m.sub(m.mean(axis=1), axis=0)
    agree = {
        "model_vs_pre2024_all": _corr(resp, resp_early),
        "model_vs_history_all": _corr(resp, emp_b[resp.columns]),
        "model_vs_history_pattern": _corr(demean(resp), demean(emp_b[resp.columns])),
        "per_scenario_pattern": {k: float(stats.spearmanr(resp.loc[k], emp_b.loc[k, resp.columns]).statistic)
                                 for k in resp.index},
    }
    # how much of the model's cross-stock difference is just each stock's volatility?
    vol = scale[d >= TEST_START].groupby(level="ticker").mean()
    agree["vol_scaling_corr"] = {k: float(stats.spearmanr(resp.loc[k].abs(), vol[resp.columns]).statistic)
                                 for k in resp.index}
    agree["vol_scaling_corr_median"] = float(np.median(list(agree["vol_scaling_corr"].values())))
    print("  agreement:", {k: round(v, 3) for k, v in agree.items() if not isinstance(v, dict)})

    single = Path(__file__).parent / "output" / "results.json"
    nvda_single = None
    if single.exists():
        nvda_single = json.loads(single.read_text())["horizons"][str(H)]["full_model"]["ic_spearman"]

    results = {
        "universe": UNIVERSE, "horizon": H, "test_start": TEST_START, "last_date": str(d.max().date()),
        "n_rows": int(len(X)), "n_features": int(X.shape[1]), "missing_inputs": [t for t, _ in missing],
        "per_stock": per_stock, "cross_section": xs, "agreement": agree,
        "nvda_single_model_ic": nvda_single,
        "scenario_labels": {k: v[0] for k, v in SCENARIOS.items()},
    }
    (out / "results.json").write_text(json.dumps(results, indent=2, default=float))
    panel_report.build(results, resp, resp_early, emp_b, emp_t, shares, pnls, out / "report.html")
    print(f"wrote {out / 'report.html'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reuse", action="store_true")
    main(reuse=ap.parse_args().reuse)
