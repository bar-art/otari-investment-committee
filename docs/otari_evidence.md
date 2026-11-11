# Otari features: what was run and what proves it

Everything below ran against hosted otari.ai (EU region) on 2026-11-10 and 2026-11-11.
The request ids are Otari's own `Otari-Request-ID` values. Keys are never shown.

| Rubric feature | Command | Evidence |
|---|---|---|
| Open-weight models, distinct roles | `python -m committee analyze JPM --profile otari` | The ledger shows `served_model`: DeepSeek-V4.1-Flash for the 3 analysts, GLM-5.3-Flash for bull and bear, DeepSeek-V4-Pro for the judge, Qwen3.8-Flash for the risk officer. All are open-weight, from four labs |
| Model routing and fallbacks | same | Each call names a routing policy (`committee-analyst`, `committee-debater`, `committee-judge`, `committee-risk`). `Otari-Model` gives the serving model and `Otari-Attempt-Count` and `Otari-Fallback` are logged per call. A client-side chain behind each policy covers a gateway-level `invalid_model` or outage (unit-tested: `test_client_falls_back_down_the_chain`) |
| Budgets and spending limits | `JUDGE_BUDGET_FLOOR_USD=5 python -m committee analyze HD --profile open` | The judge has its own key with a $2/day budget. `GET /key/budget` returned `remaining_usd: 2.0`. With remaining budget below the floor, the decision recorded `judge_tiered_down_to: alibaba/qwen3.8-flash` and completed without errors. The judge's remaining budget fell to $1.9995 and then $1.9975 as decisions ran, so judge spend lands on the judge's key |
| Per-user budgets | `python -m committee analyze PG --profile otari --user desk:alice` and `... GOOGL ... --user desk:bob` | `GET /end-users` on the service key lists `desk:alice` (spend $0.013962) and `desk:bob` (spend $0.014065), both on `desk-budget` ($0.50/day), not blocked |
| Code execution | before any BUY, and as a probe on NVDA | `POST /code/execute` (request `1d5014ee-0852-4528-a90c-56802ecadc71`) recomputed 1-day historical VaR95 4.1642% and ES97.5 5.7979% from 252 returns. Both match the local values to within 1e-9, so the gate check `otari_sandbox_recompute` is PASS |
| Web fetch | `OTARI_WEB_FETCH=1 python -m committee analyze AAPL --profile open` | `POST /fetch` (`builtin_fetch`, request `61ac91e4-6a53-42b5-a5a7-79b9014e7ef4`) read Apple's 8-K of 2026-11-05. The events analyst cited the Item 8.01 note issuance. `GET /usage/requests/{id}` shows the fetch was billed $0.010000, and the client writes that into its ledger |
| Spend attribution | every call | `metadata` tags (role, profile, run, ticker, end_user), one `Otari-Conversation-Id` per decision, and inline `usage.cost_usd` per call |

Not claimed: guardrails. The code sends a `prompt-injection` guardrail on the events analyst
when `OTARI_GUARDRAIL_PROFILE` is set, but this organization has no guardrails service
configured (`Otari-Guardrails: status unavailable, reason not_configured`). MCP is not used either.
