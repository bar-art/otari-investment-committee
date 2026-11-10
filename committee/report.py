"""Markdown report for a backtest run."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PCT = ["total_return", "ann_return", "ann_vol", "max_drawdown", "es975_1d", "buy_hit_rate",
       "buy_avg_fwd_excess", "avg_gross"]


COUNTS = {"n_buy", "n_sell", "n_veto", "decisions", "llm_calls", "failed_calls", "otari_fallbacks",
          "cited_unknown_fields", "decisions_with_errors", "ic_dates", "days"}


def _fmt(col, v):
    if pd.isna(v):
        return "–"
    if col in PCT:
        return f"{v:.2%}"
    if col in ("llm_cost_usd", "cost_per_decision_usd"):
        return f"${v:.4f}"
    if col in COUNTS:
        return str(int(v))
    return f"{v:.2f}" if isinstance(v, float) else str(v)


def write_report(run_dir: Path) -> Path:
    meta = json.loads((run_dir / "meta.json").read_text())
    t = pd.read_csv(run_dir / "metrics.csv", index_col=0)
    perf = [c for c in ["total_return", "ann_vol", "sharpe", "max_drawdown", "es975_1d", "avg_gross", "turnover"] if c in t]
    sig = [c for c in ["mean_ic", "ic_tstat", "buy_hit_rate", "buy_avg_fwd_excess", "n_buy", "n_sell", "n_veto"] if c in t]
    ops = [c for c in ["decisions", "llm_calls", "llm_cost_usd", "cost_per_decision_usd", "failed_calls",
                       "otari_fallbacks", "cited_unknown_fields", "decisions_with_errors"] if c in t]

    def table(cols, rows=None):
        sub = t.loc[rows] if rows is not None else t
        head = "| strategy | " + " | ".join(cols) + " |\n|---|" + "---|" * len(cols) + "\n"
        return head + "\n".join(f"| {i} | " + " | ".join(_fmt(c, r[c]) for c in cols) + " |" for i, r in sub.iterrows())

    committees = [i for i in t.index if i.startswith("committee:")]
    md = [f"# Backtest {meta['run_id']}",
          f"Universe: {', '.join(meta['universe'])}  \nWindow: {meta['start']} to {meta['end']}, weekly decisions, "
          f"one-day execution lag, {meta['tc_bps']} bps per unit of turnover, cash earns 0.  \n"
          f"Prompts anonymized: {meta['anonymize']}.",
          "## Performance", table(perf), ""]
    if committees and sig:
        md += ["## Signal quality (forward excess return over SPY to next rebalance)", table(sig, committees), ""]
    if committees and ops:
        md += ["## Cost and reliability (from the Otari-tagged ledger)", table(ops, committees), ""]
    md += ["![equity](equity.png)", "",
           "*Small sample: a few months of weekly decisions on a dozen stocks. Treat differences in "
           "returns as noise unless the IC t-stat is large. Research only, not investment advice.*"]
    path = run_dir / "report.md"
    path.write_text("\n\n".join(md))
    return path
