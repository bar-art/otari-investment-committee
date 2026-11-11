"""Walk-forward backtest of the committee, one or more model profiles side by side.

Timing (no look-ahead):
- decide on the close of rebalance day t, with data <= t only
- trade on the close of t+1 (one-day execution lag), pay `tc_bps` per unit of turnover
- weights drift with prices until the next rebalance
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Profile, RiskLimits
from .data import MarketData
from .llm import OtariClient
from .pipeline import Committee, Decision
from .risk import apply_portfolio_limits, hist_var_es


@dataclass
class BacktestConfig:
    universe: list[str]
    start: date
    end: date
    tc_bps: float = 5.0
    workers: int = 4
    anonymize: bool = True
    construction: str = "rank"   # rank: long the top_k judge outlooks; judge: follow each BUY/HOLD/SELL
    top_k: int = 3


def rebalance_dates(days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    s = pd.Series(days, index=days)
    return list(s.groupby([days.isocalendar().year, days.isocalendar().week]).max())


def perf_metrics(ret: pd.Series) -> dict:
    ret = ret.dropna()
    if ret.empty:
        return {}
    eq = (1 + ret).cumprod()
    n = len(ret)
    ann = eq.iloc[-1] ** (252 / n) - 1
    vol = ret.std() * np.sqrt(252)
    down = ret[ret < 0].std() * np.sqrt(252)
    _, es = hist_var_es(ret.values)
    return {"total_return": eq.iloc[-1] - 1, "ann_return": ann, "ann_vol": vol,
            "sharpe": ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else np.nan,
            "sortino": ret.mean() * 252 / down if down and down > 0 else np.nan,
            "max_drawdown": float((eq / eq.cummax() - 1).min()), "es975_1d": es, "days": n}


def signed_signal(d: Decision) -> float:
    """The judge's outlook (-2..+2) when it gave one, else its signed conviction."""
    if d.judge.get("outlook") is not None:
        return float(d.judge["outlook"])
    c = d.judge.get("conviction", 0)
    return {"BUY": c, "SELL": -c}.get(d.final_action, 0.0)


class Backtester:
    def __init__(self, cfg: BacktestConfig, data: MarketData, limits: RiskLimits | None = None):
        self.cfg = cfg
        self.data = data
        self.limits = limits or RiskLimits()
        data.load_prices(cfg.universe)
        self.days = data.trading_days(cfg.start, cfg.end)
        closes = pd.DataFrame({t: data.prices(t, cfg.end)["Close"] for t in [*cfg.universe, "SPY"]})
        self.rets = closes.pct_change().loc[self.days]
        self.sectors = {t: data.profile(t).get("sector", "Unknown") for t in cfg.universe}

    def n_decisions(self) -> int:
        return len(rebalance_dates(self.days)) * len(self.cfg.universe)

    def run_profile(self, profile: Profile, client: OtariClient | None, log=print) -> dict:
        cfg = self.cfg
        committee = Committee(profile, self.data, client, self.limits, anonymize=cfg.anonymize)
        rdates = set(rebalance_dates(self.days))
        w = pd.Series(0.0, index=cfg.universe)  # current (drifted) weights
        pending: pd.Series | None = None
        equity, peak = 1.0, 1.0
        daily, decisions, gross = [], [], []
        turnover = 0.0
        for day in self.days:
            r = self.rets.loc[day, cfg.universe].fillna(0.0)
            port_r = float((w * r).sum())
            # drift
            w = w * (1 + r) / (1 + port_r) if (1 + port_r) != 0 else w
            cost = 0.0
            if pending is not None:  # execute yesterday's decision at today's close
                trade = (pending - w).abs().sum()
                cost = trade * cfg.tc_bps / 1e4
                turnover += trade
                w, pending = pending, None
            net = port_r - cost
            equity *= 1 + net
            peak = max(peak, equity)
            daily.append((day, net))
            gross.append(float(w.sum()))
            if day in rdates:
                dd = 1 - equity / peak

                rank = cfg.construction == "rank"

                def one(t, _w=w.copy(), _dd=dd):
                    return committee.decide(t, day.date(), current_weight=float(_w[t]), portfolio_drawdown=_dd,
                                            size=not rank)

                with ThreadPoolExecutor(max_workers=cfg.workers) as pool:
                    ds = list(pool.map(one, cfg.universe))
                    if rank:  # score first, then size: long the best outlooks, out of the rest
                        ranked = sorted(ds, key=lambda d: d.judge.get("outlook", -2), reverse=True)
                        picks = {d.ticker for d in ranked[:cfg.top_k] if d.judge.get("outlook", -2) > 0}

                        def sized(d, _w=w.copy(), _dd=dd):
                            buy = d.ticker in picks
                            return committee.size(d, "BUY" if buy else "OUT", self.limits.max_weight if buy else 0.0,
                                                  float(_w[d.ticker]), _dd)

                        ds = list(pool.map(sized, ds))
                decisions += ds
                target, notes = apply_portfolio_limits({d.ticker: d.final_weight for d in ds}, self.sectors,
                                                       self.limits)
                pending = pd.Series(target).reindex(cfg.universe).fillna(0.0)
                acts = " ".join(f"{d.ticker}:{d.final_action[0]}{d.final_weight:.0%}" for d in ds)
                log(f"[{profile.name}] {day.date()} equity {equity:.4f} | {acts}" + (f" | {'; '.join(notes)}" if notes else ""))
        ret = pd.Series(dict(daily))
        return {"profile": profile.name, "returns": ret, "decisions": decisions,
                "avg_gross": float(np.mean(gross)), "turnover": turnover}

    def benchmarks(self) -> dict[str, pd.Series]:
        spy = self.rets["SPY"].fillna(0.0)
        ew = self.rets[self.cfg.universe].fillna(0.0).mean(axis=1)
        return {"SPY buy&hold": spy, "Equal-weight universe": ew}

    def signal_quality(self, decisions: list[Decision]) -> dict:
        """Forward excess return over SPY from t+1 close to the next rebalance's t+1 close."""
        rd = rebalance_dates(self.days)
        idx = list(self.days)
        cum = (1 + self.rets.fillna(0)).cumprod()
        rows = []
        for d in decisions:
            t = pd.Timestamp(d.as_of)
            i = rd.index(t)
            if i + 1 >= len(rd):
                continue
            a, b = idx.index(t) + 1, idx.index(rd[i + 1]) + 1
            if b >= len(idx):
                continue
            fwd = cum[d.ticker].iloc[b] / cum[d.ticker].iloc[a] - 1
            spy = cum["SPY"].iloc[b] / cum["SPY"].iloc[a] - 1
            rows.append({"date": d.as_of, "ticker": d.ticker, "signal": signed_signal(d),
                         "action": d.final_action, "fwd_excess": fwd - spy})
        df = pd.DataFrame(rows)
        if df.empty:
            return {}
        ics = [g["signal"].rank().corr(g["fwd_excess"].rank()) for _, g in df.groupby("date")  # Spearman
               if g["signal"].nunique() > 1]
        ics = [x for x in ics if not np.isnan(x)]
        buys = df[df["action"] == "BUY"]
        return {"mean_ic": float(np.mean(ics)) if ics else np.nan,
                "ic_tstat": float(np.mean(ics) / (np.std(ics, ddof=1) / np.sqrt(len(ics)))) if len(ics) > 2 else np.nan,
                "ic_dates": len(ics),
                "buy_hit_rate": float((buys["fwd_excess"] > 0).mean()) if len(buys) else np.nan,
                "buy_avg_fwd_excess": float(buys["fwd_excess"].mean()) if len(buys) else np.nan,
                "n_buy": int(len(buys)), "n_sell": int(df["action"].isin(["SELL", "OUT"]).sum()),
                "n_veto": int((df["action"] == "VETO").sum())}


def write_run(out: Path, bt: Backtester, results: list[dict], ledger_df: pd.DataFrame, meta: dict):
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for name, ret in bt.benchmarks().items():
        rows.append({"strategy": name, **perf_metrics(ret)})
        curves[name] = (1 + ret).cumprod()
    for res in results:
        p = res["profile"]
        m = {"strategy": f"committee:{p}", **perf_metrics(res["returns"]), **bt.signal_quality(res["decisions"]),
             "avg_gross": res["avg_gross"], "turnover": res["turnover"]}
        led = ledger_df[ledger_df["profile"] == p] if not ledger_df.empty else ledger_df
        n_dec = len(res["decisions"])
        cost = float(led["cost_usd"].fillna(0).sum()) if not led.empty else 0.0
        m.update({"decisions": n_dec, "llm_calls": int(len(led)), "llm_cost_usd": cost,
                  "cost_per_decision_usd": cost / n_dec if n_dec else 0.0,
                  "failed_calls": int((~led["ok"]).sum()) if not led.empty else 0,
                  "otari_fallbacks": int(led["fallback"].sum()) if not led.empty else 0,
                  "cited_unknown_fields": sum(d.grounding.get("cited_unknown", 0) for d in res["decisions"]),
                  "decisions_with_errors": sum(bool(d.errors) for d in res["decisions"])})
        rows.append(m)
        curves[f"committee:{p}"] = (1 + res["returns"]).cumprod()
        with open(out / f"decisions_{p}.jsonl", "w") as f:
            for d in res["decisions"]:
                f.write(json.dumps(d.as_dict(), ensure_ascii=False, default=str) + "\n")
    table = pd.DataFrame(rows).set_index("strategy")
    table.to_csv(out / "metrics.csv")
    pd.DataFrame(curves).to_csv(out / "equity.csv")
    if not ledger_df.empty:
        ledger_df.to_csv(out / "ledger.csv", index=False)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    _plot(pd.DataFrame(curves), out / "equity.png")
    return table


def _plot(curves: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)
    for c in curves.columns:
        ax.plot(curves.index, curves[c], label=c, lw=1.6 if c.startswith("committee") else 1.1,
                ls="-" if c.startswith("committee") else "--")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Walk-forward backtest (1-day execution lag, costs included)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
