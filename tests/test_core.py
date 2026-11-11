"""Offline tests: no network, no keys."""
from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from committee.agents import AnalystView, JudgeDecision
from committee.config import Profile, RiskLimits, OtariSettings
from committee.features import event_features, fundamental_features, ttm, _records
from committee.llm import LLMError, Ledger, OtariClient, extract_json
from committee.pipeline import Committee
from committee.risk import apply_portfolio_limits, gate, hist_var_es, RiskMetrics


# ---------------- synthetic data ----------------
class FakeData:
    def __init__(self, n=400, seed=0, red_flag=False):
        rng = np.random.default_rng(seed)
        idx = pd.bdate_range("2025-01-01", periods=n)
        self.px = {}
        for t, mu in [("SPY", 0.0003), ("AAA", 0.0006)]:
            c = 100 * np.exp(np.cumsum(rng.normal(mu, 0.012, n)))
            self.px[t] = pd.DataFrame({"Open": c, "High": c, "Low": c, "Close": c,
                                       "Volume": rng.integers(1e6, 2e6, n)}, index=idx)
        self.red_flag = red_flag

    def prices(self, t, as_of):
        df = self.px[t]
        return df[df.index <= pd.Timestamp(as_of)]

    def facts(self, t):
        return {"facts": {"us-gaap": {}, "dei": {}}}

    def filings(self, t):
        return pd.DataFrame({"form": ["8-K", "10-Q"], "filed": pd.to_datetime(["2026-05-20", "2026-05-01"]),
                             "items": ["4.02,9.01" if self.red_flag else "2.02,9.01", ""]})

    def profile(self, t):
        return {"name": t, "sector": "Test"}

    def headlines(self, t, as_of):
        return []


AS_OF = date(2026, 5, 29)


# ---------------- JSON extraction ----------------
def test_extract_json_handles_fences_and_prose():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Sure! Here it is: {"a": {"b": 2}} hope this helps') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_schema_trims_long_lists():
    v = AnalystView.model_validate({"score": 1, "confidence": 0.5, "key_points": list("abcdef"), "cited_fields": []})
    assert len(v.key_points) == 4
    assert JudgeDecision.model_validate({"action": "buy", "outlook": 1, "conviction": 0.5, "target_weight": 0.05,
                                         "thesis": "x"}).action == "BUY"


# ---------------- Otari client with a fake OpenAI SDK ----------------
class FakeRaw:
    def __init__(self, text, model="m", fallback="false", cost="0.001200"):
        self.headers = {"Otari-Model": model, "Otari-Provider": "fake", "Otari-Fallback": fallback,
                        "Otari-Request-ID": "req-1"}
        usage = SimpleNamespace(prompt_tokens=100, completion_tokens=20, model_extra={"cost_usd": cost})
        self._resp = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
                                     usage=usage, model=model)

    def parse(self):
        return self._resp


def fake_factory(script):
    """script: model -> list of answers (str or Exception), consumed in order."""
    calls = []

    def create(**kw):
        calls.append(kw)
        ans = script[kw["model"]].pop(0)
        if isinstance(ans, Exception):
            raise ans
        return FakeRaw(ans, model=kw["model"])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        with_raw_response=SimpleNamespace(create=create))))
    return (lambda key: client), calls


def make_client(script, **kw):
    factory, calls = fake_factory(script)
    settings = OtariSettings(base_url="http://fake", default_key="k")
    return OtariClient(settings, Ledger(), openai_factory=factory, run_id="t", **kw), calls


GOOD = json.dumps({"score": 1, "confidence": 0.7, "key_points": ["p"], "cited_fields": ["ret_1m"]})


def _ask(client, chain):
    return client.chat_json(role="technical", chain=chain, system="s", user="u", schema=AnalystView,
                            temperature=0.2, profile="p", ticker="AAA", as_of="2026-05-29", conversation_id="c1")


def test_client_records_cost_tags_and_session():
    client, calls = make_client({"m1": [GOOD]})
    v = _ask(client, ["m1"])
    assert v.score == 1
    rec = client.ledger.records[0]
    assert rec.cost_usd == pytest.approx(0.0012) and rec.request_id == "req-1" and rec.ok
    assert calls[0]["extra_body"]["metadata"]["role"] == "technical"
    assert calls[0]["extra_headers"]["Otari-Conversation-Id"] == "c1"


def test_client_repairs_invalid_json_once():
    client, calls = make_client({"m1": ["not json", GOOD]})
    assert _ask(client, ["m1"]).score == 1
    assert [r.ok for r in client.ledger.records] == [False, True]
    assert client.ledger.records[1].repaired
    assert calls[1]["messages"][-1]["role"] == "user"  # repair instruction appended


def test_client_falls_back_down_the_chain():
    client, _ = make_client({"m1": [RuntimeError("403 budget_exceeded")], "m2": [GOOD]})
    assert _ask(client, ["m1", "m2"]).score == 1
    assert client.ledger.records[-1].client_fallback
    assert client.ledger.summary()["failed_calls"] == 1


def test_client_raises_when_everything_fails():
    client, _ = make_client({"m1": ["bad", "still bad"], "m2": [RuntimeError("down")]})
    with pytest.raises(LLMError):
        _ask(client, ["m1", "m2"])


def test_cache_makes_reruns_free(tmp_path):
    client, calls = make_client({"m1": [GOOD]}, cache_dir=tmp_path)
    _ask(client, ["m1"])
    _ask(client, ["m1"])  # served from cache, the fake has no second answer
    assert len(calls) == 1 and client.ledger.records[1].cached and client.ledger.records[1].cost_usd == 0


# ---------------- risk ----------------
def test_var_es_on_known_distribution():
    r = np.linspace(-0.05, 0.05, 1001)
    var, es = hist_var_es(r)
    assert var == pytest.approx(0.045, abs=1e-3)
    assert es == pytest.approx(0.04875, abs=1e-3)
    assert es >= var


def test_gate_caps_and_vetoes():
    m = RiskMetrics(var95_1d=0.02, es975_1d=0.04, vol_ann=0.40, beta_1y=1.0, history_days=252)
    lim = RiskLimits()
    d = gate(0.5, "BUY", m, lim, fundamentals_age_days=30)
    assert d.final_weight == pytest.approx(min(lim.max_weight, lim.vol_budget / 0.40, lim.es_budget / 0.04))
    assert gate(0.05, "BUY", m, lim, fundamentals_age_days=30, llm_cap=0.01).final_weight == 0.01
    # the risk officer cannot raise size
    assert gate(0.02, "BUY", m, lim, fundamentals_age_days=30, llm_cap=0.9).final_weight == 0.02
    v = gate(0.05, "BUY", m, lim, event_red_flags=["4.02"], fundamentals_age_days=30)
    assert not v.approved and v.final_weight == 0
    assert gate(0.05, "BUY", m, lim, fundamentals_age_days=None).final_weight <= lim.max_weight / 2


def test_drawdown_stop_blocks_adds():
    m = RiskMetrics(0.01, 0.02, 0.2, 1.0, 252)
    d = gate(0.08, "BUY", m, RiskLimits(), current_weight=0.03, portfolio_drawdown=0.2, fundamentals_age_days=10)
    assert d.final_weight == 0.03


def test_portfolio_limits_sector_and_gross():
    lim = RiskLimits()
    w, notes = apply_portfolio_limits({"A": 0.1, "B": 0.1, "C": 0.1, "D": 0.1}, {"A": "x", "B": "x", "C": "x", "D": "y"}, lim)
    assert sum(v for k, v in w.items() if k in "ABC") == pytest.approx(lim.max_sector_weight)
    assert notes


# ---------------- point-in-time fundamentals ----------------
def _facts(rows):
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": rows}}}, "dei": {}}}


def _q(start, end, val, filed, fp="Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": "10-Q", "fp": fp}


QROWS = [
    _q("2024-01-01", "2024-03-31", 100, "2024-05-01"), _q("2024-04-01", "2024-06-30", 100, "2024-08-01"),
    _q("2024-07-01", "2024-09-30", 100, "2024-11-01"), _q("2024-01-01", "2024-12-31", 400, "2025-02-15", "FY"),
    _q("2025-01-01", "2025-03-31", 120, "2025-05-01"), _q("2025-04-01", "2025-06-30", 120, "2025-08-01"),
]


def test_ttm_from_quarters_and_derived_q4():
    df = _records(_facts(QROWS), "Revenues", date(2025, 9, 1))
    value, growth, end = ttm(df)
    assert value == 100 + 100 + 120 + 120  # Q3'24 + Q4'24 (=400-300) + Q1'25 + Q2'25
    assert end == pd.Timestamp("2025-06-30")


def test_point_in_time_hides_later_filings():
    df = _records(_facts(QROWS), "Revenues", date(2025, 7, 1))  # Q2'25 not yet filed
    assert df["end"].max() == pd.Timestamp("2025-03-31")


def test_restatement_filed_later_is_invisible():
    rows = QROWS + [_q("2025-04-01", "2025-06-30", 999, "2025-12-01")]
    early = _records(_facts(rows), "Revenues", date(2025, 9, 1))
    assert early[early["end"] == "2025-06-30"]["val"].iloc[0] == 120


def test_event_red_flags():
    f = FakeData(red_flag=True).filings("AAA")
    ev = event_features(f, AS_OF)
    assert ev["red_flags"] == ["4.02"] and ev["n_8k"] == 1


# ---------------- end-to-end committee ----------------
RULES = Profile(name="rules", description="", models={}, fallbacks={})


def test_rules_committee_end_to_end():
    d = Committee(RULES, FakeData()).decide("AAA", AS_OF)
    assert set(d.views) == {"fundamental", "technical", "events"}
    assert d.final_action in {"BUY", "HOLD", "SELL", "VETO"}
    assert 0 <= d.final_weight <= RiskLimits().max_weight / 2  # fundamentals missing -> half cap


def test_red_flag_vetoes_any_buy():
    d = Committee(RULES, FakeData(red_flag=True)).decide("AAA", AS_OF, current_weight=0.05)
    assert d.final_weight == 0


LLM = Profile(name="llm", description="", models={"analyst": "a", "debater": "b", "judge": "j", "risk": "r"},
              fallbacks={})


def _answers(judge_action="BUY", risk_cap=0.02):
    view = GOOD
    turn = json.dumps({"arguments": ["x"], "rebuttal": "", "cited_fields": []})
    judge = json.dumps({"action": judge_action, "outlook": 1.5, "conviction": 0.8, "target_weight": 0.08, "thesis": "t",
                        "what_would_change_my_mind": "w"})
    risk = json.dumps({"max_weight": risk_cap, "concerns": ["c"]})
    return {"a": [view] * 3, "b": [turn, turn], "j": [judge], "r": [risk]}


def test_llm_committee_respects_risk_officer_and_tags_session():
    client, calls = make_client(_answers(risk_cap=0.02))
    d = Committee(LLM, FakeData(), client, anonymize=True).decide("AAA", AS_OF)
    assert d.final_weight <= 0.02
    assert len({c["extra_headers"]["Otari-Conversation-Id"] for c in calls}) == 1  # one Otari session
    assert "AAA" not in json.dumps([c["messages"] for c in calls])  # anonymized prompts
    assert {c["extra_body"]["metadata"]["role"] for c in calls} == {"fundamental", "technical", "events",
                                                                     "bull", "bear", "judge", "risk"}


def test_llm_committee_fails_closed_when_judge_fails():
    ans = _answers()
    ans["j"] = [RuntimeError("down")]
    client, _ = make_client(ans)
    d = Committee(LLM, FakeData(), client).decide("AAA", AS_OF, current_weight=0.0)
    assert d.judge["action"] == "HOLD" and d.final_weight == 0 and d.errors


def test_json_mode_is_dropped_for_models_that_refuse_it():
    factory, calls = fake_factory({"m1": [GOOD, GOOD]})
    inner = factory("k")
    create = inner.chat.completions.with_raw_response.create

    def picky(**kw):
        if "response_format" in kw:
            calls.append(kw)
            raise RuntimeError("400 unsupported_parameter: response_format")
        return create(**kw)

    inner.chat.completions.with_raw_response.create = picky
    client = OtariClient(OtariSettings(base_url="http://fake", default_key="k"), Ledger(),
                         openai_factory=lambda key: inner)
    assert _ask(client, ["m1"]).score == 1
    _ask(client, ["m1"])  # second call goes straight to plain mode
    assert sum("response_format" in c for c in calls) == 1


def test_scoring_and_sizing_can_be_split():
    c = Committee(RULES, FakeData())
    d = c.decide("AAA", AS_OF, size=False)
    assert d.risk == {} and "outlook" in d.judge
    c.size(d, "BUY", 0.10)
    assert d.risk["checks"] and 0 < d.final_weight <= RiskLimits().max_weight
    c.size(d, "OUT", 0.0, current_weight=0.05)
    assert d.final_weight == 0 and d.final_action == "OUT"
    assert "_sizing" not in d.as_dict()


def test_failed_judge_is_never_ranked_in():
    ans = _answers()
    ans["j"] = [RuntimeError("down")]
    client, _ = make_client(ans)
    d = Committee(LLM, FakeData(), client).decide("AAA", AS_OF, size=False)
    assert d.judge["outlook"] == -2
