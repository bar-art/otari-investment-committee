"""Agent roles: output schemas, prompts, and a deterministic rules version of each role.

The rules versions are the no-LLM baseline (profile `rules`) and the offline mode. Comparing
an LLM profile against them shows what the models add beyond the same features.
"""
from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, Field, field_validator

Action = Literal["BUY", "HOLD", "SELL"]
# Models often overshoot length limits; trim instead of failing the whole answer.
Points = Annotated[list[str], BeforeValidator(lambda v: [str(x) for x in (v or [])][:4])]
Text = Annotated[str, BeforeValidator(lambda v: str(v or "")[:600])]


class AnalystView(BaseModel):
    score: float = Field(ge=-2, le=2, description="-2 very bearish ... +2 very bullish")
    confidence: float = Field(ge=0, le=1)
    key_points: Points
    cited_fields: list[str] = Field(description="names of input fields the view relies on")


class DebateTurn(BaseModel):
    arguments: Points
    rebuttal: Text = ""
    cited_fields: list[str] = []


class JudgeDecision(BaseModel):
    action: Action
    outlook: float = Field(ge=-2, le=2, description="expected performance vs a typical US large cap over "
                                                    "the next month, -2 much worse ... +2 much better")
    conviction: float = Field(ge=0, le=1)
    target_weight: float = Field(ge=0, le=1, description="fraction of portfolio for BUY, else 0")
    thesis: Text
    what_would_change_my_mind: Text = ""

    @field_validator("action", mode="before")
    @classmethod
    def upper(cls, v):
        return str(v).strip().upper()


class RiskOfficerView(BaseModel):
    max_weight: float = Field(ge=0, le=1)
    concerns: Points


GUARDRAIL = ("Use only the data provided. Do not use outside knowledge about any company, "
             "date or market event. If the data is thin, say so and lower confidence. "
             "Be concise: at most 3 points of at most 40 words each. "
             "This is research, not investment advice.")

ANALYST_PROMPTS = {
    "fundamental": "You are a fundamental equity analyst for US stocks. Judge valuation (earnings, "
                   "sales and FCF yields, book-to-market), profitability, growth and balance-sheet "
                   "leverage from the point-in-time SEC data given.",
    "technical": "You are a technical and quantitative analyst. Judge momentum (12-1 month excess "
                 "return), trend, mean-reversion signals, volatility and volume from the data given.",
    "events": "You are an event-driven analyst. Judge recent SEC filings (8-K items, insider Form 4 "
              "activity) and, when present, recent headlines. Treat red-flag items as serious.",
}


def analyst_messages(role: str, label: str, features: dict) -> tuple[str, str]:
    system = f"{ANALYST_PROMPTS[role]} {GUARDRAIL} Score from -2 to +2. Cite the field names you used."
    user = f"Stock: {label}\nInput fields:\n{json.dumps(features, ensure_ascii=False, indent=1)}"
    return system, user


def debate_messages(side: str, label: str, views: dict[str, dict], opponent: list[str]) -> tuple[str, str]:
    stance = ("Make the strongest honest case FOR buying" if side == "bull"
              else "Make the strongest honest case AGAINST buying (risks, weak evidence, crowding)")
    system = f"You are the {side} on an investment committee. {stance}. {GUARDRAIL}"
    user = f"Stock: {label}\nAnalyst views and their inputs:\n{json.dumps(views, ensure_ascii=False, indent=1)}"
    if opponent:
        user += f"\n\nThe other side argued:\n- " + "\n- ".join(opponent) + "\nRebut the strongest point."
    return system, user


def judge_messages(label: str, views: dict, bull: list[str], bear: list[str], max_weight: float,
                   current_weight: float) -> tuple[str, str]:
    system = ("You are the portfolio manager chairing the committee. Weigh the analysts and the debate, "
              "then decide BUY (hold or add up to target_weight), HOLD (keep current weight) or SELL "
              f"(exit). Long-only. target_weight must be <= {max_weight:.2f}. Prefer HOLD when evidence "
              "conflicts. Separately, always give `outlook`: your best estimate of how this stock will do "
              "versus a typical US large cap over the next month, even when you HOLD. The portfolio is built "
              f"by ranking outlooks across stocks, so use the full scale. {GUARDRAIL}")
    user = (f"Stock: {label}\nCurrent portfolio weight: {current_weight:.3f}\n"
            f"Analyst views:\n{json.dumps(views, ensure_ascii=False, indent=1)}\n"
            f"Bull: {json.dumps(bull, ensure_ascii=False)}\nBear: {json.dumps(bear, ensure_ascii=False)}")
    return system, user


def risk_messages(label: str, decision: dict, metrics: dict, events: dict) -> tuple[str, str]:
    system = ("You are the risk officer. Hard limits are already enforced in code. Set the maximum "
              "weight you are comfortable with for unquantified risks (event risk, crowding, thin data). "
              f"You can only reduce the proposed weight. {GUARDRAIL}")
    user = (f"Stock: {label}\nProposed decision: {json.dumps(decision)}\nRisk metrics: {json.dumps(metrics)}\n"
            f"Events: {json.dumps(events, ensure_ascii=False)}")
    return system, user


# ---------------- deterministic rules (baseline / offline) ----------------
def _clip(x, lo=-2.0, hi=2.0):
    return max(lo, min(hi, x))


def _g(d, k, default=0.0):
    v = d.get(k)
    return default if v is None else v


def rules_analyst(role: str, f: dict) -> AnalystView:
    if role == "fundamental":
        s = (8 * _g(f, "earnings_yield") - 0.3 + 4 * _g(f, "fcf_yield") + 2 * _g(f, "revenue_growth_yoy")
             + 1.5 * _g(f, "operating_margin_ttm") - 0.1 * min(_g(f, "liabilities_to_equity"), 5))
        cited = ["earnings_yield", "fcf_yield", "revenue_growth_yoy", "operating_margin_ttm", "liabilities_to_equity"]
        conf = 0.6 if f.get("earnings_yield") is not None else 0.2
    elif role == "technical":
        mom = _g(f, "excess_ret_12m_ex1m_vs_spy")
        trend = 0.5 if _g(f, "price_vs_sma200") > 0 else -0.5
        rev = -0.5 if _g(f, "rsi_14", 50) > 75 else (0.3 if _g(f, "rsi_14", 50) < 30 else 0)
        s = 3 * mom + trend + rev
        cited = ["excess_ret_12m_ex1m_vs_spy", "price_vs_sma200", "rsi_14"]
        conf = 0.5
    else:
        s = -2.0 if f.get("red_flags") else 0.0
        s += 0.2 if any(i.startswith("2.02") for i in f.get("8k_items", [])) else 0
        cited = ["red_flags", "8k_items"]
        conf = 0.3
    return AnalystView(score=round(_clip(s), 3), confidence=conf, key_points=[f"rules score {s:+.2f}"],
                       cited_fields=cited)


def rules_judge(views: dict[str, AnalystView], max_weight: float, current_weight: float) -> JudgeDecision:
    w = {"fundamental": 0.4, "technical": 0.4, "events": 0.2}
    s = sum(w[k] * v.score for k, v in views.items())
    o = round(_clip(s), 3)
    if s > 0.25:
        return JudgeDecision(action="BUY", outlook=o, conviction=min(1, s / 2), target_weight=max_weight * min(1, s),
                             thesis=f"weighted score {s:+.2f}", what_would_change_my_mind="score below 0")
    if s < -0.25:
        return JudgeDecision(action="SELL", outlook=o, conviction=min(1, -s / 2), target_weight=0,
                             thesis=f"weighted score {s:+.2f}", what_would_change_my_mind="score above 0")
    return JudgeDecision(action="HOLD", outlook=o, conviction=0.2, target_weight=current_weight,
                         thesis=f"weighted score {s:+.2f}", what_would_change_my_mind="|score| > 0.25")
