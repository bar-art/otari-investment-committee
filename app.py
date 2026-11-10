"""Streamlit demo: streamlit run app.py"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import streamlit as st

from committee.config import ROOT, DataSettings, OtariSettings, load_profiles
from committee.data import MarketData
from committee.llm import Ledger, OtariClient
from committee.pipeline import Committee

st.set_page_config(page_title="Investment Committee on Otari", layout="wide")
st.title("AI Investment Committee for US equities")
st.caption("Seven agents, one Otari gateway: per-agent keys and budgets, model routing, fallback and "
           "cost per decision. Research only, not investment advice.")

profiles = load_profiles()
live, past = st.tabs(["Run a decision", "Backtest results"])

with live:
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    ticker = c1.text_input("Ticker", "NVDA").upper().strip()
    as_of = c2.date_input("As of", date.today())
    pname = c3.selectbox("Model profile", list(profiles), index=list(profiles).index("open") if "open" in profiles else 0)
    anonymize = c4.checkbox("Anonymize prompts", value=False, help="Hide ticker and date from the models")
    end_user = c5.selectbox("Bill to (Otari end user)", ["(agent keys)", "desk:alice", "desk:bob"],
                            help="With OTARI_SERVICE_KEY set, each desk has its own Otari budget")
    end_user = None if end_user.startswith("(") else end_user
    st.caption(profiles[pname].description)
    if st.button("Convene the committee", type="primary"):
        profile = profiles[pname]
        data = MarketData(start=(as_of - timedelta(days=500)).isoformat(), end=(as_of + timedelta(days=1)).isoformat())
        client = (OtariClient(OtariSettings(), Ledger(), cache_dir=DataSettings().cache_dir / "llm",
                              run_id=f"ui-{datetime.now():%Y%m%d%H%M%S}") if profile.uses_llm else None)
        with st.spinner("Analysts, debate, judge, risk..."):
            d = Committee(profile, data, client, anonymize=anonymize, end_user=end_user).decide(ticker, as_of)

        badge = {"BUY": "🟢", "SELL": "🔴", "HOLD": "⚪", "VETO": "⛔"}[d.final_action]
        st.subheader(f"{badge} {d.final_action}  ·  final weight {d.final_weight:.2%}")
        st.write(f"**Judge:** {d.judge['action']} (conviction {d.judge['conviction']:.2f}, proposed "
                 f"{d.judge['target_weight']:.1%}). {d.judge['thesis']}")
        if d.judge.get("what_would_change_my_mind"):
            st.caption(f"What would change my mind: {d.judge['what_would_change_my_mind']}")

        cols = st.columns(3)
        for col, role in zip(cols, ["fundamental", "technical", "events"]):
            with col:
                v = d.views.get(role)
                st.markdown(f"#### {role.title()}")
                if v:
                    st.metric("score (-2..+2)", f"{v['score']:+.2f}", f"confidence {v['confidence']:.2f}")
                    for p in v["key_points"]:
                        st.write(f"- {p}")
                else:
                    st.error("no view (call failed)")
                with st.expander("Point-in-time inputs"):
                    st.json(d.features[role])

        if d.debate:
            b1, b2 = st.columns(2)
            b1.markdown("#### 🐂 Bull")
            for a in d.debate["bull"]:
                b1.write(f"- {a}")
            b2.markdown("#### 🐻 Bear")
            for a in d.debate["bear"]:
                b2.write(f"- {a}")

        st.markdown("#### Risk gate (deterministic; the LLM risk officer can only cut)")
        st.dataframe(pd.DataFrame(d.risk["checks"]), width="stretch", hide_index=True)
        st.caption("Metrics: " + ", ".join(f"{k} {v}" for k, v in d.risk["metrics"].items()))
        if d.risk_officer:
            st.write(f"Risk officer cap {d.risk_officer['max_weight']:.1%}: " + "; ".join(d.risk_officer["concerns"]))
        if d.errors:
            st.warning("Failed closed on: " + " | ".join(d.errors))

        if d.otari:
            st.markdown("#### Otari features used in this decision")
            st.json(d.otari)
        if client:
            led = client.ledger.to_frame()
            st.markdown("#### What this decision cost (Otari inline `usage.cost_usd`)")
            s = client.ledger.summary()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("LLM calls", s["calls"])
            m2.metric("Cost", f"${s['cost_usd']:.4f}")
            m3.metric("Tokens", f"{s['prompt_tokens'] + s['completion_tokens']:,}")
            m4.metric("Fallbacks", s["otari_fallbacks"] + s["client_fallbacks"])
            by_role = led.groupby("role").agg(cost_usd=("cost_usd", "sum"), calls=("role", "size"),
                                              model=("served_model", "last")).reset_index()
            st.bar_chart(by_role, x="role", y="cost_usd")
            st.dataframe(led[["role", "kind", "requested_model", "served_model", "provider", "fallback", "attempts",
                              "end_user", "guardrails", "prompt_tokens",
                              "completion_tokens", "cost_usd", "latency_ms", "cached", "ok", "error"]],
                         width="stretch", hide_index=True)
            st.caption(f"Otari session id: `{d.session_id}`. Open it in the Otari dashboard to replay the decision.")

with past:
    runs = sorted((ROOT / "runs").glob("*/metrics.csv"), reverse=True)
    if not runs:
        st.info("No runs yet: python -m committee backtest --profiles rules,routed")
    else:
        run = st.selectbox("Run", [r.parent.name for r in runs])
        rd = ROOT / "runs" / run
        st.markdown((rd / "report.md").read_text().split("![equity]")[0])
        st.line_chart(pd.read_csv(rd / "equity.csv", index_col=0, parse_dates=True))
        files = sorted(rd.glob("decisions_*.jsonl"))
        if files:
            f = st.selectbox("Decision log", [x.name for x in files])
            rows = [json.loads(l) for l in (rd / f).read_text().splitlines()]
            df = pd.DataFrame([{"date": r["as_of"], "ticker": r["ticker"], "action": r["final_action"],
                                "weight": r["final_weight"], "judge": r["judge"].get("thesis", "")[:120],
                                "errors": len(r["errors"])} for r in rows])
            st.dataframe(df, width="stretch", hide_index=True)
