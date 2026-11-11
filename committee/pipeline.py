"""One committee decision for one stock on one date.

  [fundamental | technical | events]  (parallel)  ->  bull <-> bear  ->  judge
      -> deterministic risk gate (+ LLM risk officer that can only cut)  ->  final weight

Any LLM failure fails closed: the role is dropped (analysts) or the decision becomes HOLD.
"""
from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .agents import (AnalystView, DebateTurn, JudgeDecision, RiskOfficerView, analyst_messages,
                     debate_messages, judge_messages, risk_messages, rules_analyst, rules_judge)
from .config import Profile, RiskLimits
from .data import DataSource
from .features import event_features, fundamental_features, technical_features
from .llm import LLMError, OtariClient
from .risk import gate, hist_var_es, risk_metrics

SANDBOX_CODE = """import json, numpy as np
r = np.array(json.loads('{returns}'))
var = -np.quantile(r, 0.05)
q = np.quantile(r, 0.025)
es = -r[r <= q].mean()
print(json.dumps({{"var95_1d": float(var), "es975_1d": float(es), "n": int(len(r))}}))
"""

ANALYSTS = ("fundamental", "technical", "events")


@dataclass
class Decision:
    ticker: str
    as_of: str
    profile: str
    session_id: str
    features: dict
    views: dict = field(default_factory=dict)
    debate: dict = field(default_factory=dict)
    judge: dict = field(default_factory=dict)
    risk_officer: dict = field(default_factory=dict)
    risk: dict = field(default_factory=dict)
    final_action: str = "HOLD"
    final_weight: float = 0.0
    errors: list[str] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    otari: dict = field(default_factory=dict)   # which gateway features this decision used

    def as_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def grounding_check(cited: list[str], inputs: dict) -> tuple[int, int]:
    """Count citations that name a real input field. Ungrounded citations are a hallucination signal."""
    ok = sum(1 for c in cited if c in inputs)
    return ok, len(cited) - ok


class Committee:
    def __init__(self, profile: Profile, data: DataSource, client: OtariClient | None = None,
                 limits: RiskLimits | None = None, anonymize: bool = False, use_risk_officer: bool = True,
                 end_user: str | None = None):
        if profile.uses_llm and client is None:
            raise ValueError(f"profile '{profile.name}' needs an OtariClient")
        self.profile = profile
        self.data = data
        self.client = client
        self.limits = limits or RiskLimits()
        self.anonymize = anonymize
        self.use_risk_officer = use_risk_officer
        self.end_user = end_user

    # ----- inputs -----
    def build_features(self, ticker: str, as_of: date) -> dict:
        px = self.data.prices(ticker, as_of)
        bench = self.data.prices("SPY", as_of)
        price = float(px["Close"].iloc[-1])
        try:
            fund = fundamental_features(self.data.facts(ticker), price, as_of)
        except Exception as exc:
            fund = {"error": f"fundamentals unavailable: {type(exc).__name__}"}
        try:
            events = event_features(self.data.filings(ticker), as_of)
        except Exception as exc:
            events = {"error": f"filings unavailable: {type(exc).__name__}"}
        if not self.anonymize:
            heads = self.data.headlines(ticker, as_of)
            if heads:
                events["recent_headlines"] = heads
        return {"fundamental": fund, "technical": technical_features(px, bench), "events": events,
                "_risk": risk_metrics(px, bench), "_sector": self.data.profile(ticker).get("sector", "Unknown"),
                "_returns": px["Close"].pct_change().dropna().iloc[-252:].round(8).tolist()}

    def _latest_8k_url(self, ticker: str, as_of: date) -> str:
        try:
            f = self.data.filings(ticker)
        except Exception:
            return ""
        f = f[(f["form"] == "8-K") & (f["filed"] <= pd.Timestamp(as_of))
              & (f["filed"] > pd.Timestamp(as_of) - pd.Timedelta(days=30))]
        return f.sort_values("filed").iloc[-1]["url"] if not f.empty and "url" in f else ""

    def _label(self, ticker: str, sector: str, as_of: date) -> str:
        # Anonymized prompts hide the ticker and date so the model cannot recall what happened next.
        return f"Company X (sector: {sector})" if self.anonymize else f"{ticker} (sector: {sector}, as of {as_of})"

    def _ctx(self, role: str, d: Decision) -> dict:
        return {"role": role, "profile": self.profile.name, "ticker": d.ticker, "as_of": d.as_of,
                "user": self.end_user}

    def _ask(self, role, system, user, schema, d: Decision, chain: list[str] | None = None,
             guardrails: list[dict] | None = None):
        return self.client.chat_json(role=role, chain=chain or self.profile.chain_for(role), system=system,
                                     user=user, schema=schema, temperature=self.profile.temperature_for(role),
                                     profile=self.profile.name, ticker=d.ticker, as_of=d.as_of,
                                     conversation_id=d.session_id, end_user=self.end_user, guardrails=guardrails)

    # ----- decision -----
    def decide(self, ticker: str, as_of: date, current_weight: float = 0.0,
               portfolio_drawdown: float = 0.0, size: bool = True) -> Decision:
        feats = self.build_features(ticker, as_of)
        metrics, sector, returns = feats.pop("_risk"), feats.pop("_sector"), feats.pop("_returns")
        d = Decision(ticker=ticker, as_of=as_of.isoformat(), profile=self.profile.name,
                     session_id=f"{self.profile.name}-{ticker}-{as_of}-{uuid.uuid4().hex[:6]}", features=feats)
        label = self._label(ticker, sector, as_of)
        llm = self.profile.uses_llm
        cfg = self.client.settings if self.client else None
        if self.end_user:
            d.otari["end_user"] = self.end_user

        # 0) web fetch: the latest 8-K's text, read through Otari (live mode only: it names the company)
        if llm and cfg.web_fetch and not self.anonymize:
            url = self._latest_8k_url(ticker, as_of)
            if url:
                try:
                    text = self.client.fetch(url, max_chars=12000, **self._ctx("events", d))
                    body = text.find("Item ")  # skip the cover page
                    feats["events"]["latest_8k_excerpt"] = text[body if body > 0 else 0:][:5000]
                    d.otari["web_fetch"] = url
                except LLMError as exc:
                    d.errors.append(f"web fetch: {exc}")
        # untrusted outside text (8-K body, headlines) goes through an Otari guardrail
        events_guard = None
        if llm and cfg.guardrail_profile:
            events_guard = [{"profile": cfg.guardrail_profile, "mode": cfg.guardrail_mode, "on_unavailable": "monitor"}]

        # 1) analysts in parallel
        def run_analyst(role):
            if not llm:
                return role, rules_analyst(role, feats[role])
            system, user = analyst_messages(role, label, feats[role])
            return role, self._ask(role, system, user, AnalystView, d,
                                   guardrails=events_guard if role == "events" else None)

        views: dict[str, AnalystView] = {}
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_analyst, r) for r in ANALYSTS]
            for f in futures:
                try:
                    role, view = f.result()
                    views[role] = view
                except LLMError as exc:
                    d.errors.append(str(exc))
        d.views = {k: v.model_dump() for k, v in views.items()}
        g_ok = g_bad = 0
        for role, v in views.items():
            ok, bad = grounding_check(v.cited_fields, feats[role])
            g_ok, g_bad = g_ok + ok, g_bad + bad

        # 2) debate + 3) judge
        max_w = self.limits.max_weight
        judge: JudgeDecision | None = None
        if not views:
            d.errors.append("no analyst views: fail closed")
        elif not llm:
            judge = rules_judge(views, max_w, current_weight)
        else:
            packed = {k: {"view": d.views[k], "inputs": feats[k]} for k in views}
            bull_args: list[str] = []
            bear_args: list[str] = []
            try:
                for _ in range(self.profile.debate_rounds):
                    s, u = debate_messages("bull", label, packed, bear_args)
                    bull = self._ask("bull", s, u, DebateTurn, d)
                    bull_args += bull.arguments
                    s, u = debate_messages("bear", label, packed, bull_args)
                    bear = self._ask("bear", s, u, DebateTurn, d)
                    bear_args += bear.arguments
                d.debate = {"bull": bull_args, "bear": bear_args}
                s, u = judge_messages(label, d.views, bull_args, bear_args, max_w, current_weight)
                chain = None
                left = self.client.remaining_budget("judge", self.end_user)
                if left is not None:
                    d.otari["judge_budget_left_usd"] = round(left, 4)
                    if left < cfg.judge_budget_floor_usd and self.profile.economy_judge:
                        chain = [self.profile.economy_judge]  # tier down before the cap is hit
                        d.otari["judge_tiered_down_to"] = self.profile.economy_judge
                judge = self._ask("judge", s, u, JudgeDecision, d, chain=chain)
            except LLMError as exc:
                d.errors.append(str(exc))
        if judge is None:
            # outlook -2 so a failed committee is never ranked into the portfolio
            judge = JudgeDecision(action="HOLD", outlook=-2, conviction=0, target_weight=current_weight,
                                  thesis="committee failed: fail closed", what_would_change_my_mind="")
        d.judge = judge.model_dump()

        # 4) risk sizing (the backtest can defer it to rank stocks across the universe first)
        d._sizing = {"label": label, "metrics": metrics, "returns": returns}
        if size:
            self.size(d, judge.action, judge.target_weight, current_weight, portfolio_drawdown)
        d.grounding = {"cited_ok": g_ok, "cited_unknown": g_bad}
        d.features["sector"] = sector
        return d

    def size(self, d: Decision, action: str, proposed_weight: float, current_weight: float = 0.0,
             portfolio_drawdown: float = 0.0) -> Decision:
        """LLM risk officer (can only cut), deterministic gate, then the Otari sandbox check."""
        ctx = d._sizing
        metrics, returns, label = ctx["metrics"], ctx["returns"], ctx["label"]
        llm = self.profile.uses_llm
        cfg = self.client.settings if self.client else None
        feats = d.features
        llm_cap = None
        if llm and self.use_risk_officer and action == "BUY":
            try:
                proposal = {**d.judge, "action": action, "target_weight": proposed_weight}
                s, u = risk_messages(label, proposal, metrics.as_dict(), feats["events"])
                ro = self._ask("risk", s, u, RiskOfficerView, d)
                llm_cap = ro.max_weight
                d.risk_officer = ro.model_dump()
            except LLMError as exc:
                d.errors.append(str(exc))
                llm_cap = 0.0  # no risk sign-off, no new risk
        rd = gate(proposed_weight, action, metrics, self.limits, current_weight=current_weight,
                  portfolio_drawdown=portfolio_drawdown, event_red_flags=feats["events"].get("red_flags"),
                  fundamentals_age_days=feats["fundamental"].get("days_since_last_filing"), llm_cap=llm_cap)
        # independent recompute of VaR/ES in Otari's code sandbox; a mismatch vetoes the trade
        if llm and cfg.code_check and action == "BUY" and rd.final_weight > 0:
            try:
                out = self.client.execute(SANDBOX_CODE.format(returns=json.dumps(returns)), **self._ctx("risk", d))
                remote = json.loads(out.get("stdout", "").strip().splitlines()[-1])
                lv, le = hist_var_es(returns)
                ok = abs(remote["var95_1d"] - lv) < 1e-9 and abs(remote["es975_1d"] - le) < 1e-9
                rd.checks.append({"rule": "otari_sandbox_recompute", "status": "PASS" if ok else "VETO",
                                  "detail": f"sandbox VaR {remote['var95_1d']:.4%} ES {remote['es975_1d']:.4%} "
                                            f"vs local VaR {lv:.4%} ES {le:.4%}"})
                d.otari["code_execution"] = "match" if ok else "mismatch"
                if not ok:
                    rd.final_weight, rd.approved = 0.0, False
            except Exception as exc:
                rd.checks.append({"rule": "otari_sandbox_recompute", "status": "SKIP", "detail": str(exc)[:160]})
        d.risk = rd.as_dict()
        d.final_weight = rd.final_weight
        d.final_action = action if rd.approved else "VETO"
        return d
