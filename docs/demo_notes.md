# Demo shot list (about 4 minutes)

Keywords only. Speak in your own words; the rules require your own screen recording and voice.

Before recording:
- In the terminal, run `streamlit run app.py`.
- Open otari.ai in a second tab.
- Run one live decision beforehand, so the cache is warm and you have a fallback if the network is slow.

## 1. The problem (0:00–0:30)
- Multi-agent finance demos are expensive black boxes.
- My questions: what does each agent cost, which model answered, can a runaway agent overspend?
- Answer: every call goes through Otari.

## 2. Live decision (0:30–2:00), Streamlit tab "Run a decision"
- Ticker NVDA, profile `otari`, bill to `desk:alice`. Click **Convene the committee**.
- **Three analyst cards**: the same open-weight model (DeepSeek V4.1 Flash) running three jobs in parallel. Open "Point-in-time inputs" to show the SEC data.
- **Events card**: it read the latest **8-K through Otari web fetch** (`web_fetch` in the Otari features box).
- **Bull and bear**: a different lab (GLM-5.3 Flash), so the debate is not one model arguing with itself.
- **Judge**: DeepSeek V4 Pro, through the routing policy `committee-judge`.
- **Risk gate table**: hard limits; the LLM risk officer can only cut. Show the `otari_sandbox_recompute` row: **VaR/ES recomputed in Otari's code sandbox**, and a mismatch vetoes the trade. If this decision is HOLD, show it on a BUY from the backtest log instead.
- **Cost table**: `served_model`, `attempts`, `fallback`, `end_user`, `cost_usd` per call. Total about 1.4 cents.

## 3. Otari dashboard (2:00–3:00)
- **Routing → Policies**: the four committee policies and their backups. They are **routing and fallbacks**.
- **API keys**: `committee-judge` with $2/day. These are **per-agent budgets**.
  - Mention: below the floor, the judge **tiers down** to Qwen 3.8 Flash.
  - Mention: at the cap, the decision fails closed.
- **End users**: `desk:alice` and `desk:bob`, each on `desk-budget`. These are **per-user budgets**.
- **Activity / Sessions**: one session per decision, spend grouped by the `role` tag.

## 4. Does it work? (3:00–4:00), Streamlit tab "Backtest results"
- Run `submission-v2`: weekly decisions, Jul–Nov 2026, 12 large caps, long the top 3 judge outlooks.
- v1 lesson: the judge held 227 of 228 times because the debate always conflicts. The fix was to split scoring from sizing.
- 56 sandbox checks, all PASS. The risk officer cut every pick and vetoed CVX on suspicious data.
- **Prompts anonymized**: no ticker, no date, so the model cannot recall the future.
- Fundamentals come from **SEC filings cut at their filing date**.
- Compare `open` with the **no-LLM rules baseline** on IC, hit rate and cost per decision.
- Be honest: a small sample; read the IC t-stat, not the returns.

## 5. Close (4:00–4:30)
- Open-weight models from four labs.
- Six Otari features, each used in the app, not just enabled.
- Fail-closed everywhere.
- Repo URL.
