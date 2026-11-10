"""Deterministic risk gate. Runs after the judge; the LLM risk officer can only cut size, never add."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import RiskLimits


def hist_var_es(returns: np.ndarray, var_level: float = 0.95, es_level: float = 0.975) -> tuple[float, float]:
    """1-day historical VaR and Expected Shortfall as positive loss fractions."""
    r = np.asarray(returns, dtype=float)
    r = r[~np.isnan(r)]
    if len(r) < 20:
        return float("nan"), float("nan")
    var = -np.quantile(r, 1 - var_level)
    q = np.quantile(r, 1 - es_level)
    es = -r[r <= q].mean()
    return float(var), float(es)


def beta(asset: pd.Series, bench: pd.Series) -> float:
    df = pd.concat([asset, bench], axis=1, sort=True).dropna()
    if len(df) < 60:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1])
    return float(cov[0, 1] / cov[1, 1])


@dataclass
class RiskMetrics:
    var95_1d: float
    es975_1d: float
    vol_ann: float
    beta_1y: float
    history_days: int

    def as_dict(self):
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def risk_metrics(px: pd.DataFrame, bench: pd.DataFrame, lookback: int = 252) -> RiskMetrics:
    r = px["Close"].pct_change().dropna().iloc[-lookback:]
    rb = bench["Close"].pct_change().dropna()
    var, es = hist_var_es(r.values)
    return RiskMetrics(var95_1d=var, es975_1d=es, vol_ann=float(r.std() * np.sqrt(252)),
                       beta_1y=beta(r, rb), history_days=len(r))


@dataclass
class RiskDecision:
    approved: bool
    proposed_weight: float
    final_weight: float
    checks: list[dict] = field(default_factory=list)   # {rule, status, detail}
    metrics: dict = field(default_factory=dict)

    def as_dict(self):
        return self.__dict__


def gate(proposed_weight: float, action: str, metrics: RiskMetrics, limits: RiskLimits, *,
         current_weight: float = 0.0, portfolio_drawdown: float = 0.0,
         event_red_flags: list[str] | None = None, fundamentals_age_days: int | None = None,
         llm_cap: float | None = None) -> RiskDecision:
    """Apply hard limits in a fixed order and record why size was cut."""
    checks = []
    w = max(0.0, float(proposed_weight)) if action == "BUY" else (current_weight if action == "HOLD" else 0.0)

    def cap(rule, limit, detail):
        nonlocal w
        if w > limit + 1e-12:
            checks.append({"rule": rule, "status": "CUT", "detail": f"{detail}: {w:.2%} -> {limit:.2%}"})
            w = limit
        else:
            checks.append({"rule": rule, "status": "PASS", "detail": detail})

    if metrics.history_days < limits.min_history_days:
        checks.append({"rule": "history", "status": "VETO", "detail": f"only {metrics.history_days} days of returns"})
        w = 0.0
    if event_red_flags:
        checks.append({"rule": "event_red_flag", "status": "VETO",
                       "detail": f"8-K items {', '.join(event_red_flags)} in the last 30 days"})
        w = 0.0
    if action == "BUY" and portfolio_drawdown > limits.drawdown_stop and w > current_weight:
        checks.append({"rule": "drawdown_stop", "status": "CUT",
                       "detail": f"portfolio drawdown {portfolio_drawdown:.1%} > {limits.drawdown_stop:.0%}: no adds"})
        w = current_weight

    cap("max_weight", limits.max_weight, f"single-name cap {limits.max_weight:.0%}")
    if metrics.vol_ann > 0:
        cap("vol_budget", limits.vol_budget / metrics.vol_ann,
            f"weight x vol <= {limits.vol_budget:.1%} (vol {metrics.vol_ann:.0%})")
    if metrics.es975_1d > 0:
        cap("es_budget", limits.es_budget / metrics.es975_1d,
            f"weight x ES97.5 <= {limits.es_budget:.2%} (ES {metrics.es975_1d:.2%})")
    if fundamentals_age_days is None or fundamentals_age_days > limits.stale_fundamentals_days:
        cap("stale_fundamentals", limits.max_weight / 2,
            f"fundamentals missing or older than {limits.stale_fundamentals_days} days")
    if llm_cap is not None:
        cap("risk_officer", max(0.0, llm_cap), "LLM risk officer (can only reduce)")

    vetoed = any(c["status"] == "VETO" for c in checks)
    return RiskDecision(approved=not vetoed and (w > 0 or action != "BUY"),
                        proposed_weight=round(float(proposed_weight), 4), final_weight=round(w, 4),
                        checks=checks, metrics=metrics.as_dict())


def apply_portfolio_limits(weights: dict[str, float], sectors: dict[str, str],
                           limits: RiskLimits) -> tuple[dict[str, float], list[str]]:
    """Sector cap and gross cap across names, scaled pro rata."""
    notes = []
    w = dict(weights)
    by_sector: dict[str, float] = {}
    for t, x in w.items():
        by_sector[sectors.get(t, "Unknown")] = by_sector.get(sectors.get(t, "Unknown"), 0) + x
    for s, tot in by_sector.items():
        if tot > limits.max_sector_weight:
            k = limits.max_sector_weight / tot
            for t in w:
                if sectors.get(t, "Unknown") == s:
                    w[t] *= k
            notes.append(f"sector {s} {tot:.1%} scaled to {limits.max_sector_weight:.0%}")
    gross = sum(w.values())
    if gross > limits.max_gross:
        w = {t: x * limits.max_gross / gross for t, x in w.items()}
        notes.append(f"gross {gross:.1%} scaled to {limits.max_gross:.0%}")
    return w, notes
