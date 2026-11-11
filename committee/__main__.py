"""CLI: python -m committee {doctor,analyze,backtest,usage,profiles}"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta

from .config import DEFAULT_UNIVERSE, ROOT, DataSettings, OtariSettings, load_profiles
from .data import MarketData
from .llm import Ledger, OtariClient


def _client(run_id: str, cache: bool = True) -> OtariClient:
    return OtariClient(OtariSettings(), Ledger(), cache_dir=DataSettings().cache_dir / "llm" if cache else None,
                       run_id=run_id)


def cmd_profiles(_):
    for p in load_profiles().values():
        models = ", ".join(f"{k}={v}" for k, v in p.models.items()) or "no LLM"
        print(f"{p.name:10s} {p.description}\n{'':10s} {models}")


def cmd_doctor(_):
    s = OtariSettings()
    print(f"Otari base URL: {s.base_url}")
    print(f"default key set: {bool(s.default_key)}")
    from .config import ROLES
    for r in ROLES:
        own = bool(__import__('os').getenv(f"OTARI_API_KEY_{r.upper()}"))
        print(f"  {r:12s} {'own key' if own else 'default key'}")
    try:
        models = _client("doctor").list_models()
        print(f"gateway OK, {len(models)} models visible. Profile models:")
        for p in load_profiles().values():
            for slot, m in p.models.items():
                print(f"  [{p.name}] {slot:8s} {m:45s} {'listed' if m in models else 'not listed (catalog ids may still route)'}")
    except Exception as exc:
        print(f"gateway check failed: {exc}")
        sys.exit(1)


def cmd_analyze(a):
    from .pipeline import Committee
    profile = load_profiles()[a.profile]
    as_of = date.fromisoformat(a.as_of) if a.as_of else date.today()
    data = MarketData(start=(as_of - timedelta(days=500)).isoformat(), end=(as_of + timedelta(days=1)).isoformat())
    client = _client(f"analyze-{datetime.now():%Y%m%d%H%M%S}") if profile.uses_llm else None
    d = Committee(profile, data, client, anonymize=a.anonymize, end_user=a.user).decide(a.ticker.upper(), as_of,
                                                                         current_weight=a.current_weight)
    if a.json:
        print(json.dumps(d.as_dict(), indent=2, ensure_ascii=False, default=str))
    else:
        print(f"\n{d.ticker} as of {d.as_of} | profile {d.profile} | session {d.session_id}")
        for role, v in d.views.items():
            print(f"  {role:12s} score {v['score']:+.2f} conf {v['confidence']:.2f} | {'; '.join(v['key_points'])[:160]}")
        if d.debate:
            print(f"  bull: {' | '.join(d.debate['bull'])[:300]}")
            print(f"  bear: {' | '.join(d.debate['bear'])[:300]}")
        j = d.judge
        print(f"  judge: {j['action']} conviction {j['conviction']:.2f} target {j['target_weight']:.1%} | {j['thesis'][:240]}")
        for c in d.risk["checks"]:
            if c["status"] != "PASS":
                print(f"  risk {c['status']:4s} {c['rule']}: {c['detail']}")
        print(f"  FINAL: {d.final_action} weight {d.final_weight:.2%}")
        if d.otari:
            print(f"  Otari: {json.dumps(d.otari)}")
        if d.errors:
            print(f"  errors: {d.errors}")
    if client:
        print("\nLLM usage:", json.dumps(client.ledger.summary()))


def cmd_backtest(a):
    from .backtest import BacktestConfig, Backtester, write_run
    import pandas as pd
    profiles = load_profiles()
    names = a.profiles.split(",")
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    universe = a.universe.split(",") if a.universe else DEFAULT_UNIVERSE
    for n in names:
        p = profiles[n]
        if p.uses_llm and p.knowledge_cutoff and start.isoformat() <= p.knowledge_cutoff:
            print(f"WARNING: backtest starts {start} before profile '{n}' knowledge cutoff "
                  f"{p.knowledge_cutoff}; results may reflect memorized outcomes. Anonymization "
                  f"{'is on' if not a.no_anonymize else 'is OFF'}.")
    data = MarketData(start=(start - timedelta(days=420)).isoformat(), end=(end + timedelta(days=10)).isoformat())
    bt = Backtester(BacktestConfig(universe, start, end, tc_bps=a.tc_bps, workers=a.workers,
                                   anonymize=not a.no_anonymize, construction=a.construction, top_k=a.top_k), data)
    n = bt.n_decisions()
    llm_names = [x for x in names if profiles[x].uses_llm]
    calls = n * 7 * len(llm_names)
    print(f"{n} decisions per profile; about {calls} LLM calls for {llm_names or 'no LLM profiles'}.")
    if calls > 300 and not a.yes:
        if input("Continue? [y/N] ").strip().lower() != "y":
            return
    run_id = a.run_id or f"bt-{datetime.now():%Y%m%d-%H%M%S}"
    client = _client(run_id) if llm_names else None
    results = [bt.run_profile(profiles[x], client if profiles[x].uses_llm else None) for x in names]
    ledger = client.ledger.to_frame() if client else pd.DataFrame()
    out = ROOT / "runs" / run_id
    table = write_run(out, bt, results, ledger, {"run_id": run_id, "universe": universe, "start": start, "end": end,
                                                 "profiles": names, "anonymize": not a.no_anonymize,
                                                 "tc_bps": a.tc_bps, "construction": a.construction,
                                                 "top_k": a.top_k})
    from .report import write_report
    write_report(out)
    cols = [c for c in ["total_return", "sharpe", "max_drawdown", "mean_ic", "buy_hit_rate", "llm_cost_usd",
                        "cost_per_decision_usd"] if c in table]
    print(table[cols].to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nwritten to {out}")


def cmd_usage(a):
    print(json.dumps(_client("usage", cache=False).usage_by_role(), indent=2))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="committee", description="Multi-agent US equity committee on Otari")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("profiles").set_defaults(fn=cmd_profiles)
    sub.add_parser("doctor").set_defaults(fn=cmd_doctor)
    s = sub.add_parser("analyze")
    s.add_argument("ticker")
    s.add_argument("--profile", default="open")
    s.add_argument("--user", help="bill this decision to an Otari end user (needs OTARI_SERVICE_KEY)")
    s.add_argument("--as-of")
    s.add_argument("--current-weight", type=float, default=0.0)
    s.add_argument("--anonymize", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_analyze)
    s = sub.add_parser("backtest")
    s.add_argument("--profiles", default="rules")
    s.add_argument("--start", default="2026-07-01")
    s.add_argument("--end", default="2026-11-06")
    s.add_argument("--universe")
    s.add_argument("--tc-bps", type=float, default=5.0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--no-anonymize", action="store_true")
    s.add_argument("--construction", choices=["rank", "judge"], default="rank",
                   help="rank: long the top-k judge outlooks each week; judge: follow each BUY/HOLD/SELL")
    s.add_argument("--top-k", type=int, default=3)
    s.add_argument("--run-id")
    s.add_argument("--yes", action="store_true")
    s.set_defaults(fn=cmd_backtest)
    sub.add_parser("usage").set_defaults(fn=cmd_usage)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
