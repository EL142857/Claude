"""Universe of external factors for the NVDA model.

Every series here is a *quantified* outside influence on NVDA. Each one is
tagged with a factor group so the report can attribute forecast power by group.
"""

START = "2014-01-01"
TARGET = "NVDA"

# Yahoo Finance tickers -> (group, description)
YAHOO = {
    # target
    "NVDA": ("target", "NVIDIA"),
    # broad market and risk appetite
    "^GSPC": ("market", "S&P 500"),
    "QQQ": ("market", "Nasdaq-100 ETF"),
    "IWM": ("market", "Russell 2000 ETF"),
    "^VIX": ("volatility", "VIX 30d implied vol"),
    "^VIX3M": ("volatility", "VIX 3-month"),
    "^VVIX": ("volatility", "Vol of VIX"),
    "^SKEW": ("volatility", "CBOE SKEW (tail-risk pricing)"),
    # semiconductor sector and supply chain
    "SMH": ("semis", "VanEck Semiconductor ETF"),
    "AMD": ("semis", "AMD (competitor)"),
    "AVGO": ("semis", "Broadcom (custom ASIC competitor)"),
    "TSM": ("supply_chain", "TSMC ADR (foundry)"),
    "ASML": ("supply_chain", "ASML (lithography)"),
    "MU": ("supply_chain", "Micron (HBM memory)"),
    # AI demand side: hyperscaler capex buyers
    "MSFT": ("ai_demand", "Microsoft"),
    "GOOGL": ("ai_demand", "Alphabet"),
    "AMZN": ("ai_demand", "Amazon"),
    "META": ("ai_demand", "Meta"),
    "ORCL": ("ai_demand", "Oracle"),
    # macro markets
    "DX-Y.NYB": ("macro", "US Dollar Index"),
    "TLT": ("macro", "20y+ Treasury ETF"),
    "HYG": ("credit", "High-yield bond ETF"),
    "CL=F": ("commodities", "WTI crude"),
    "HG=F": ("commodities", "Copper"),
    "GC=F": ("commodities", "Gold"),
    # geopolitics: China and Taiwan exposure
    "FXI": ("geopolitics", "China large-cap ETF"),
    "EWT": ("geopolitics", "Taiwan ETF"),
    "TWD=X": ("geopolitics", "USD/TWD"),
    "CNY=X": ("geopolitics", "USD/CNY"),
    # equity style factors used by quant funds
    "MTUM": ("quant_flows", "Momentum factor ETF"),
    "USMV": ("quant_flows", "Min-vol factor ETF"),
    "NVDL": ("quant_flows", "2x leveraged NVDA ETF (since 2022-12)"),
}

# FRED series -> (group, description, publication lag in trading days)
# The lag stops the model from seeing a number before it was published.
FRED = {
    "DGS10": ("rates", "10y Treasury yield", 1),
    "DGS2": ("rates", "2y Treasury yield", 1),
    "DFII10": ("rates", "10y real yield (TIPS)", 1),
    "T10YIE": ("rates", "10y breakeven inflation", 1),
    "DFF": ("rates", "Effective fed funds", 1),
    "BAMLH0A0HYM2": ("credit", "HY OAS spread (FRED keeps only ~3y of history)", 1),
    "BAA10Y": ("credit", "Moody's Baa minus 10y Treasury (full history)", 1),
    "NFCI": ("credit", "Chicago Fed financial conditions (weekly)", 6),
    "WALCL": ("liquidity", "Fed balance sheet (weekly)", 6),
    "RRPONTSYD": ("liquidity", "Overnight reverse repo", 1),
    "WTREGEN": ("liquidity", "Treasury General Account (weekly)", 6),
}

# Hard-coded NVDA earnings dates (after-close reports). Used when the live
# source fails. Extend this list as new reports come out.
EARNINGS_FALLBACK = [
    "2014-02-12", "2014-05-08", "2014-08-07", "2014-11-06",
    "2015-02-11", "2015-05-07", "2015-08-06", "2015-11-05",
    "2016-02-17", "2016-05-12", "2016-08-11", "2016-11-10",
    "2017-02-09", "2017-05-09", "2017-08-10", "2017-11-09",
    "2018-02-08", "2018-05-10", "2018-08-16", "2018-11-15",
    "2019-02-14", "2019-05-16", "2019-08-15", "2019-11-14",
    "2020-02-13", "2020-05-21", "2020-08-19", "2020-11-18",
    "2021-02-24", "2021-05-26", "2021-08-18", "2021-11-17",
    "2022-02-16", "2022-05-25", "2022-08-24", "2022-11-16",
    "2023-02-22", "2023-05-24", "2023-08-23", "2023-11-21",
    "2024-02-21", "2024-05-22", "2024-08-28", "2024-11-20",
    "2025-02-26", "2025-05-28", "2025-08-27", "2025-11-19",
]

# Forecast setup
HORIZONS = [1, 5, 20]          # trading days ahead
MAIN_HORIZON = 5
TEST_START = "2024-01-01"      # out-of-sample period starts here
RETRAIN_EVERY = 21             # walk-forward refit cadence (trading days)
MIN_TRAIN = 750                # ~3 years before the first fit

# --- Panel model: one pooled model over several mega-cap tech stocks ---
UNIVERSE = ["NVDA", "AMD", "AVGO", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA"]
PANEL_HORIZON = 5

# Shock scenarios fed through the trained panel model. Each entry moves the
# listed features by (sign x 1 training std) at the same time; a feature given
# as ("set", a, b) is instead switched from value a to value b.
SCENARIOS = {
    "vol_spike": ("波动率飙升", {"^VIX_lvl_z": 1, "^VIX_chg5": 1, "vix_term": 1, "^VVIX_lvl_z": 1}),
    "rates_up": ("长端利率上行", {"DGS10_chg5": 1, "DGS10_chg20": 1, "DFII10_chg5": 1, "DFII10_chg20": 1}),
    "credit_tight": ("信用收紧", {"baa_spread_z": 1, "baa_spread_chg5": 1, "nfci": 1}),
    "liquidity_drain": ("美联储流动性收缩", {"net_liq_chg20": -1, "net_liq_z": -1}),
    "dollar_up": ("美元走强", {"DX-Y.NYB_ret5": 1, "DX-Y.NYB_ret20": 1}),
    "china_taiwan": ("中国/台湾风险", {"FXI_ret5": -1, "EWT_ret5": -1, "TWD=X_ret5": 1, "CNY=X_ret5": 1}),
    "semis_rally": ("半导体板块走强", {"SMH_ret5": 1, "SMH_ret20": 1}),
    "supply_chain_rally": ("供应链(TSM/ASML/MU)走强", {"TSM_ret5": 1, "ASML_ret5": 1, "MU_ret5": 1}),
    "cta_ndx_sell": ("CTA 抛售纳指", {"cta_ndx_flow5": -1, "cta_ndx_pos": -1}),
    "volctl_sell": ("波动率目标基金减仓", {"volctl_flow5": -1, "volctl_expo": -1}),
    "riskpar_delever": ("风险平价去杠杆", {"riskpar_flow5": -1, "riskpar_lev": -1}),
    "momentum_crash": ("动量因子崩盘", {"mom_drawdown": -1, "mom_spread20": -1}),
    "stock_cta_sell": ("个股 CTA 仓位下降", {"cta_pos": -1, "cta_flow5": -1}),
    "statarb_overbought": ("个股相对大盘超涨(统计套利)", {"resid5_z": 1, "resid20_z": 1}),
    "opex_week": ("期权到期周", {"opex_week": ("set", 0, 1)}),
    "earnings_ahead": ("5日内有财报", {"earnings_in_next5": ("set", 0, 1)}),
}
