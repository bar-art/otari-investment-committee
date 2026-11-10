"""Market and filing data with point-in-time access.

Everything an agent sees is cut at `as_of`:
- prices: rows dated <= as_of
- fundamentals: SEC XBRL facts whose `filed` date <= as_of (restatements filed later are invisible)
- events: SEC filings whose filing date <= as_of
- headlines: only in live mode, published <= as_of
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Protocol

import pandas as pd
import requests

from .config import DataSettings


class DataSource(Protocol):
    def prices(self, ticker: str, as_of: date) -> pd.DataFrame: ...
    def facts(self, ticker: str) -> dict: ...
    def filings(self, ticker: str) -> pd.DataFrame: ...
    def profile(self, ticker: str) -> dict: ...
    def headlines(self, ticker: str, as_of: date) -> list[dict]: ...


def to_date(x) -> date:
    if isinstance(x, datetime):
        return x.date()
    if isinstance(x, date):
        return x
    return date.fromisoformat(str(x)[:10])


class MarketData:
    """yfinance prices + SEC EDGAR fundamentals and filings, cached on disk."""

    BENCHMARK = "SPY"

    def __init__(self, settings: DataSettings | None = None, start: str = "2023-01-01",
                 end: str | None = None):
        self.settings = settings or DataSettings()
        self.cache = self.settings.cache_dir / "data"
        self.cache.mkdir(parents=True, exist_ok=True)
        self.start = start
        self.end = end or (date.today() + timedelta(days=1)).isoformat()
        self._prices: dict[str, pd.DataFrame] = {}
        self._json: dict[str, dict] = {}
        self._cik: dict[str, int] | None = None
        self._last_sec = 0.0

    # ---------- prices ----------
    def load_prices(self, tickers: list[str]):
        import yfinance as yf
        need = [t for t in dict.fromkeys([*tickers, self.BENCHMARK]) if t not in self._prices]
        if not need:
            return
        tag = f"{self.start}_{self.end}"
        missing = []
        for t in need:
            f = self.cache / f"px_{t}_{tag}.csv"
            if f.exists():
                self._prices[t] = pd.read_csv(f, index_col=0, parse_dates=True)
            else:
                missing.append(t)
        if missing:
            raw = yf.download(missing, start=self.start, end=self.end, auto_adjust=True,
                              progress=False, group_by="ticker", threads=True)
            for t in missing:
                df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
                df = df.dropna(how="all")[["Open", "High", "Low", "Close", "Volume"]]
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.to_csv(self.cache / f"px_{t}_{tag}.csv")
                self._prices[t] = df

    def prices(self, ticker: str, as_of: date) -> pd.DataFrame:
        if ticker not in self._prices:
            self.load_prices([ticker])
        df = self._prices[ticker]
        return df[df.index <= pd.Timestamp(as_of)]

    def trading_days(self, start: date, end: date) -> pd.DatetimeIndex:
        idx = self.prices(self.BENCHMARK, end).index
        return idx[(idx >= pd.Timestamp(start)) & (idx <= pd.Timestamp(end))]

    # ---------- SEC ----------
    def _sec_get(self, url: str, cache_name: str, max_age_h: float = 24) -> dict:
        if cache_name in self._json:
            return self._json[cache_name]
        f = self.cache / cache_name
        if f.exists() and (time.time() - f.stat().st_mtime) < max_age_h * 3600:
            data = json.loads(f.read_text())
        else:
            wait = 0.12 - (time.time() - self._last_sec)  # SEC limit is 10 req/s
            if wait > 0:
                time.sleep(wait)
            r = requests.get(url, headers={"User-Agent": self.settings.sec_user_agent}, timeout=30)
            self._last_sec = time.time()
            r.raise_for_status()
            data = r.json()
            f.write_text(json.dumps(data))
        self._json[cache_name] = data
        return data

    def cik(self, ticker: str) -> int:
        if self._cik is None:
            raw = self._sec_get("https://www.sec.gov/files/company_tickers.json", "sec_tickers.json", 24 * 7)
            self._cik = {v["ticker"].upper(): int(v["cik_str"]) for v in raw.values()}
        return self._cik[ticker.upper().replace(".", "-")]

    def facts(self, ticker: str) -> dict:
        cik = self.cik(ticker)
        return self._sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                             f"facts_{cik}.json")

    def _submissions(self, ticker: str) -> dict:
        cik = self.cik(ticker)
        return self._sec_get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", f"subs_{cik}.json")

    def filings(self, ticker: str) -> pd.DataFrame:
        recent = self._submissions(ticker)["filings"]["recent"]
        n = len(recent["form"])
        cik = self.cik(ticker)
        acc = recent.get("accessionNumber", [""] * n)
        doc = recent.get("primaryDocument", [""] * n)
        df = pd.DataFrame({"form": recent["form"], "filed": pd.to_datetime(recent["filingDate"]),
                           "items": recent.get("items", [""] * n),
                           "url": [f"https://www.sec.gov/Archives/edgar/data/{cik}/{a.replace('-', '')}/{d}"
                                   if a and d else "" for a, d in zip(acc, doc)]})
        return df

    def profile(self, ticker: str) -> dict:
        s = self._submissions(ticker)
        return {"name": s.get("name", ticker), "sector": s.get("sicDescription") or "Unknown",
                "sic": s.get("sic", "")}

    # ---------- news (live only) ----------
    def headlines(self, ticker: str, as_of: date) -> list[dict]:
        """yfinance only serves recent news, so headlines are used in live mode only."""
        if (date.today() - as_of).days > 3:
            return []
        import yfinance as yf
        out = []
        for item in (yf.Ticker(ticker).news or [])[:15]:
            c = item.get("content", item)
            title = c.get("title")
            pub = c.get("pubDate") or c.get("providerPublishTime")
            if not title or pub is None:
                continue
            ts = (datetime.fromtimestamp(pub, tz=timezone.utc) if isinstance(pub, (int, float))
                  else datetime.fromisoformat(str(pub).replace("Z", "+00:00")))
            if ts.date() <= as_of:
                out.append({"date": ts.date().isoformat(), "title": title,
                            "publisher": (c.get("provider") or {}).get("displayName", c.get("publisher", ""))})
        return out
