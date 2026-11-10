"""Point-in-time features for each analyst. All outputs are ratios, so no raw prices or
company names are needed in prompts (which lets the backtest anonymize the stock)."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd


def _r(x, nd=4):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    return round(float(x), nd)


# ---------------- technical ----------------
def rsi(close: pd.Series, n: int = 14) -> float | None:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    v = 100 - 100 / (1 + rs)
    return _r(v.iloc[-1], 1) if len(v) > n else None


def technical_features(px: pd.DataFrame, bench: pd.DataFrame) -> dict:
    c = px["Close"]
    b = bench["Close"].reindex(c.index).ffill()
    n = len(c)

    def ret(k):
        return c.iloc[-1] / c.iloc[-1 - k] - 1 if n > k else None

    def xret(k):
        return (c.iloc[-1] / c.iloc[-1 - k]) / (b.iloc[-1] / b.iloc[-1 - k]) - 1 if n > k else None

    ema12, ema26 = c.ewm(span=12, adjust=False).mean(), c.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    hist = macd - macd.ewm(span=9, adjust=False).mean()
    daily = c.pct_change().dropna()
    vol = px["Volume"]
    sma50 = c.rolling(50).mean().iloc[-1] if n >= 50 else None
    sma200 = c.rolling(200).mean().iloc[-1] if n >= 200 else None
    hi252 = c.iloc[-252:].max()
    vz = (vol.iloc[-5:].mean() - vol.iloc[-63:].mean()) / vol.iloc[-63:].std() if n >= 63 else None
    return {
        "ret_1w": _r(ret(5)), "ret_1m": _r(ret(21)), "ret_3m": _r(ret(63)), "ret_6m": _r(ret(126)),
        "excess_ret_1m_vs_spy": _r(xret(21)), "excess_ret_3m_vs_spy": _r(xret(63)),
        "excess_ret_12m_ex1m_vs_spy": _r((1 + xret(252)) / (1 + xret(21)) - 1) if n > 252 else None,
        "rsi_14": rsi(c),
        "macd_hist_pct_of_price": _r(hist.iloc[-1] / c.iloc[-1], 5),
        "macd_hist_rising": bool(hist.iloc[-1] > hist.iloc[-2]) if n > 30 else None,
        "price_vs_sma50": _r(c.iloc[-1] / sma50 - 1) if sma50 else None,
        "price_vs_sma200": _r(c.iloc[-1] / sma200 - 1) if sma200 else None,
        "drawdown_from_52w_high": _r(c.iloc[-1] / hi252 - 1),
        "vol_ann_3m": _r(daily.iloc[-63:].std() * np.sqrt(252)),
        "volume_zscore_1w_vs_3m": _r(vz, 2),
    }


# ---------------- fundamentals (SEC XBRL, point in time) ----------------
CONCEPTS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues",
                "RevenuesNetOfInterestExpense", "SalesRevenueNet",
                "RevenueFromContractWithCustomerIncludingAssessedTax"],
    "net_income": ["NetIncomeLoss", "ProfitLoss"],
    "operating_income": ["OperatingIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "liabilities": ["Liabilities"],
    "assets": ["Assets"],
}


def _records(facts: dict, concept: str, as_of: date, unit: str = "USD") -> pd.DataFrame:
    try:
        rows = facts["facts"]["us-gaap"][concept]["units"][unit]
    except KeyError:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["filed"] = pd.to_datetime(df["filed"])
    df = df[df["filed"] <= pd.Timestamp(as_of)].copy()       # the point-in-time cut
    if df.empty:
        return df
    df["end"] = pd.to_datetime(df["end"])
    if "start" in df:
        df["start"] = pd.to_datetime(df["start"])
        df["days"] = (df["end"] - df["start"]).dt.days
        keys = ["start", "end"]
    else:
        keys = ["end"]
    # the latest value known at as_of for each period (amendments filed before as_of win)
    return df.sort_values("filed").drop_duplicates(keys, keep="last").sort_values("end")


def _pick(facts: dict, names: list[str], as_of: date) -> pd.DataFrame:
    best = pd.DataFrame()
    for name in names:
        df = _records(facts, name, as_of)
        if not df.empty and (best.empty or df["end"].max() > best["end"].max()):
            best = df
    return best


def quarters(df: pd.DataFrame) -> pd.Series:
    """Discrete fiscal-quarter values from a mix of 3-month, year-to-date and annual facts.
    Q = YTD - previous YTD with the same start; Q4 = FY - the three quarters inside it."""
    if df.empty or "days" not in df:
        return pd.Series(dtype=float)
    q: dict[pd.Timestamp, float] = {}
    for _, r in df[df["days"].between(80, 100)].iterrows():
        q[r["end"]] = r["val"]
    for _, r in df[df["days"] > 100].sort_values("days").iterrows():
        if r["end"] in q:
            continue
        gap = (r["end"] - df["end"]).dt.days
        prev = df[(df["start"] == r["start"]) & gap.between(80, 100)]
        if not prev.empty:
            q[r["end"]] = r["val"] - prev.iloc[-1]["val"]
        elif 350 <= r["days"] <= 380:
            inside = [e for e in q if r["start"] < e < r["end"]]
            if len(inside) == 3:
                q[r["end"]] = r["val"] - sum(q[e] for e in inside)
    return pd.Series(q, dtype=float).sort_index()


def _sum4(q: pd.Series, end: pd.Timestamp) -> float | None:
    last = q[q.index <= end].iloc[-4:]
    if len(last) < 4 or not 250 <= (last.index[-1] - last.index[0]).days <= 300:
        return None
    return float(last.sum())


def ttm(df: pd.DataFrame) -> tuple[float | None, float | None, pd.Timestamp | None]:
    """Trailing-twelve-month value, its year-over-year growth, and the latest period end."""
    q = quarters(df)
    if q.empty:
        return None, None, None
    end = q.index[-1]
    value = _sum4(q, end)
    year_ago = q[(q.index - (end - pd.Timedelta(days=365))).map(abs) <= pd.Timedelta(days=20)]
    prev = _sum4(q, year_ago.index[-1]) if not year_ago.empty else None
    growth = value / prev - 1 if value is not None and prev and prev > 0 else None
    return value, growth, end


def latest_instant(df: pd.DataFrame) -> float | None:
    return float(df.iloc[-1]["val"]) if not df.empty else None


def shares_outstanding(facts: dict, as_of: date) -> float | None:
    try:
        rows = pd.DataFrame(facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"])
    except KeyError:
        return None
    rows["filed"] = pd.to_datetime(rows["filed"])
    rows = rows[rows["filed"] <= pd.Timestamp(as_of)]
    if rows.empty:
        return None
    last = rows[rows["filed"] == rows["filed"].max()]
    return float(last.groupby("end")["val"].sum().iloc[-1])  # multi-class shares are summed


def fundamental_features(facts: dict, price: float, as_of: date) -> dict:
    rev = _pick(facts, CONCEPTS["revenue"], as_of)
    ni = _pick(facts, CONCEPTS["net_income"], as_of)
    rev_ttm, rev_g, rev_end = ttm(rev)
    ni_ttm, ni_g, _ = ttm(ni)
    op_ttm, _, _ = ttm(_pick(facts, CONCEPTS["operating_income"], as_of))
    cfo_ttm, _, _ = ttm(_pick(facts, CONCEPTS["cfo"], as_of))
    capex_ttm, _, _ = ttm(_pick(facts, CONCEPTS["capex"], as_of))
    equity = latest_instant(_pick(facts, CONCEPTS["equity"], as_of))
    liab = latest_instant(_pick(facts, CONCEPTS["liabilities"], as_of))
    shares = shares_outstanding(facts, as_of)
    mcap = price * shares if shares else None
    last_filed = None
    if not rev.empty:
        last_filed = rev["filed"].max().date()

    def div(a, b):
        return a / b if a is not None and b not in (None, 0) else None

    fcf = cfo_ttm - capex_ttm if cfo_ttm is not None and capex_ttm is not None else None
    return {
        "earnings_yield": _r(div(ni_ttm, mcap)),
        "sales_yield": _r(div(rev_ttm, mcap)),
        "fcf_yield": _r(div(fcf, mcap)),
        "book_to_market": _r(div(equity, mcap)),
        "roe_ttm": _r(div(ni_ttm, equity)),
        "operating_margin_ttm": _r(div(op_ttm, rev_ttm)),
        "net_margin_ttm": _r(div(ni_ttm, rev_ttm)),
        "revenue_growth_yoy": _r(rev_g),
        "net_income_growth_yoy": _r(ni_g),
        "liabilities_to_equity": _r(div(liab, equity), 2),
        "latest_period_end": rev_end.date().isoformat() if rev_end is not None else None,
        "days_since_last_filing": (as_of - last_filed).days if last_filed else None,
    }


# ---------------- events (SEC filings) ----------------
ITEM_LABELS = {
    "1.01": "entered material agreement", "1.02": "terminated material agreement",
    "1.03": "bankruptcy or receivership", "2.01": "completed acquisition or disposal",
    "2.02": "results of operations (earnings release)", "2.03": "new material debt obligation",
    "2.05": "exit or restructuring costs", "2.06": "material impairment",
    "3.01": "delisting notice", "3.02": "unregistered equity sale",
    "4.01": "auditor change", "4.02": "prior financials no longer reliable (restatement)",
    "5.02": "director or officer departure/appointment", "5.07": "shareholder vote results",
    "7.01": "Regulation FD disclosure", "8.01": "other events",
}
RED_FLAGS = {"1.03", "3.01", "4.02"}


def event_features(filings: pd.DataFrame, as_of: date, window_days: int = 30) -> dict:
    lo = pd.Timestamp(as_of - timedelta(days=window_days))
    f = filings[(filings["filed"] <= pd.Timestamp(as_of)) & (filings["filed"] > lo)]
    eightk = f[f["form"].isin(["8-K", "8-K/A"])]
    items: list[str] = []
    for s in eightk["items"]:
        items += [i.strip() for i in str(s).split(",") if i.strip() and i.strip() != "9.01"]
    insider = int((f["form"] == "4").sum())
    prior = filings[(filings["filed"] <= lo) & (filings["filed"] > lo - pd.Timedelta(days=90))]
    insider_base = (prior["form"] == "4").sum() / 3
    covered = not filings.empty and filings["filed"].min() <= lo
    return {
        "window_days": window_days,
        "coverage_complete": bool(covered),
        "n_8k": int(len(eightk)),
        "8k_items": sorted({f"{i} {ITEM_LABELS.get(i, 'other')}" for i in items}),
        "red_flags": sorted({i for i in items if i in RED_FLAGS}),
        "periodic_report_filed": bool(f["form"].isin(["10-K", "10-Q"]).any()),
        "form4_insider_filings": insider,
        "form4_vs_3m_monthly_avg": _r(insider / insider_base, 2) if insider_base else None,
    }
