"""Pipeline smoke test on SYNTHETIC random data (checks the code runs; the
numbers it prints mean nothing).
Usage: python tests/smoke_test.py [out_dir] [--no-ablation]"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run  # noqa: E402
from config import EARNINGS_FALLBACK, FRED, YAHOO  # noqa: E402

rng = np.random.default_rng(0)
idx = pd.bdate_range("2014-01-02", "2026-10-28")
mkt = rng.normal(0.0004, 0.01, len(idx))
ohlcv = {}
for t in YAHOO:
    start = idx[0] if t != "NVDL" else pd.Timestamp("2022-12-13")
    r = 0.8 * mkt + rng.normal(0, 0.012, len(idx))
    c = pd.Series(50 * np.exp(np.cumsum(r)), index=idx)
    o = c.shift(1).fillna(c) * np.exp(rng.normal(0, 0.004, len(idx)))
    df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.01, "Low": np.minimum(o, c) * 0.99,
                       "Close": c, "Volume": rng.lognormal(16, 0.3, len(idx))}, index=idx)
    ohlcv[t] = df[df.index >= start]
fred = pd.DataFrame({s: 2 + np.cumsum(rng.normal(0, 0.03, len(idx))) for s in FRED}, index=idx)
fred[["WALCL", "RRPONTSYD", "WTREGEN"]] = [8e6, 500, 7e5]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
out = Path(args[0]) if args else Path(tempfile.mkdtemp())
res = run.main(ablation="--no-ablation" not in sys.argv, ohlcv=ohlcv, fred=fred,
               earnings=pd.DatetimeIndex(EARNINGS_FALLBACK), out=out)
m = res["horizons"][5]["full_model"]
assert 0.3 < m["hit_rate"] < 0.7 and abs(m["ic_spearman"]) < 0.2, m   # random data: no edge expected
assert 0.6 < m["coverage_80"] < 0.95, m
print("smoke test OK ->", out)
