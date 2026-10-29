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
    "BAMLH0A0HYM2": ("credit", "HY OAS spread", 1),
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
