"""LLM access through the Otari gateway.

Every agent call goes through `OtariClient.chat_json`, which
- sends the request to Otari's OpenAI-compatible endpoint with the agent's own virtual key,
- tags it (role, profile, run, ticker) so spend can be grouped by agent in Otari,
- groups one committee decision into one Otari session (Otari-Conversation-Id),
- validates the answer against a pydantic schema, repairs once, then falls back
  to the next model in the profile's chain,
- records tokens, Otari's inline `usage.cost_usd`, the serving model and fallback flag.

Failures raise `LLMError`; callers fail closed (HOLD, zero weight).
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from .config import OtariSettings

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class CallRecord:
    run_id: str
    profile: str
    role: str
    ticker: str
    as_of: str
    requested_model: str
    served_model: str = ""
    provider: str = ""
    fallback: bool = False            # Otari served a different candidate than the first
    client_fallback: bool = False     # we moved down the profile's chain
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float | None = None
    latency_ms: int = 0
    cached: bool = False
    repaired: bool = False
    ok: bool = True
    error: str = ""
    request_id: str = ""
    attempts: int = 1                 # Otari-Attempt-Count (candidates tried inside a routing policy)
    end_user: str = ""                # billed end user (per-user budgets)
    guardrails: str = ""              # Otari-Guardrails verdicts
    kind: str = "chat"                # chat | fetch | code


class Ledger:
    """Thread-safe record of every model call."""

    def __init__(self):
        self._lock = threading.Lock()
        self.records: list[CallRecord] = []

    def add(self, rec: CallRecord):
        with self._lock:
            self.records.append(rec)

    def to_frame(self):
        import pandas as pd
        return pd.DataFrame([asdict(r) for r in self.records])

    def summary(self) -> dict:
        recs = self.records
        billed = [r for r in recs if not r.cached and r.cost_usd is not None]
        return {
            "calls": len(recs),
            "cached_calls": sum(r.cached for r in recs),
            "failed_calls": sum(not r.ok for r in recs),
            "otari_fallbacks": sum(r.fallback for r in recs),
            "client_fallbacks": sum(r.client_fallback for r in recs),
            "repairs": sum(r.repaired for r in recs),
            "prompt_tokens": sum(r.prompt_tokens for r in recs),
            "completion_tokens": sum(r.completion_tokens for r in recs),
            "cost_usd": round(sum(r.cost_usd for r in billed), 6),
            "unpriced_calls": sum(1 for r in recs if not r.cached and r.ok and r.cost_usd is None),
        }


_JSON_BLOCK = re.compile(r"\{.*\}", re.S)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model answer (handles ```json fences and prose)."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.M)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = _JSON_BLOCK.search(text)
    if not m:
        raise ValueError("no JSON object in answer")
    return json.loads(m.group(0))


def schema_hint(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema(), ensure_ascii=False)


class ResponseCache:
    """Disk cache so re-running a backtest costs nothing. Cached calls are logged with cost 0."""

    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(model: str, messages: list[dict], temperature: float) -> str:
        blob = json.dumps([model, messages, temperature], sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        if not self.path:
            return None
        f = self.path / f"{key}.json"
        return json.loads(f.read_text()) if f.exists() else None

    def put(self, key: str, value: dict):
        if self.path:
            (self.path / f"{key}.json").write_text(json.dumps(value, ensure_ascii=False))


class OtariClient:
    def __init__(self, settings: OtariSettings | None = None, ledger: Ledger | None = None,
                 cache_dir: Path | None = None, run_id: str = "adhoc", max_tokens: int = 1600,
                 openai_factory=None):
        self.settings = settings or OtariSettings()
        self.ledger = ledger or Ledger()
        self.cache = ResponseCache(cache_dir)
        self.run_id = run_id
        self.max_tokens = max_tokens
        self._clients: dict[str, object] = {}
        self._factory = openai_factory  # injectable for tests
        self._no_json_mode: set[str] = set()  # models that refused response_format

    def _client(self, role: str, user: str | None = None):
        key = self.settings.key_for(role, user)
        if not key:
            raise LLMError(f"no Otari key for role '{role}': set OTARI_API_KEY or OTARI_API_KEY_{role.upper()}")
        if key not in self._clients:
            if self._factory:
                self._clients[key] = self._factory(key)
            else:
                from openai import OpenAI
                self._clients[key] = OpenAI(base_url=self.settings.base_url, api_key=key,
                                            timeout=self.settings.timeout_s, max_retries=2)
        return self._clients[key]

    def _call(self, role: str, model: str, messages: list[dict], temperature: float,
              tags: dict[str, str], conversation_id: str, user: str | None = None,
              guardrails: list[dict] | None = None):
        client = self._client(role, user)
        t0 = time.perf_counter()
        body: dict = {"metadata": tags}
        if guardrails:
            body["guardrails"] = guardrails
        kwargs = dict(model=model, messages=messages, temperature=temperature, max_tokens=self.max_tokens,
                      extra_body=body, extra_headers={"Otari-Conversation-Id": conversation_id})
        if user:
            kwargs["user"] = user
        if model not in self._no_json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            raw = client.chat.completions.with_raw_response.create(**kwargs)
        except Exception as exc:  # Otari answers 400 unsupported_parameter for models without JSON mode
            if "response_format" not in kwargs or not ("unsupported_parameter" in str(exc) or "response_format" in str(exc)):
                raise
            self._no_json_mode.add(model)
            kwargs.pop("response_format")
            raw = client.chat.completions.with_raw_response.create(**kwargs)
        latency = int((time.perf_counter() - t0) * 1000)
        resp = raw.parse()
        headers = raw.headers
        usage = resp.usage
        extra = (getattr(usage, "model_extra", None) or {}) if usage else {}
        cost = extra.get("cost_usd", getattr(usage, "cost_usd", None)) if usage else None
        return {
            "text": resp.choices[0].message.content or "",
            "finish_reason": getattr(resp.choices[0], "finish_reason", "") or "",
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "cost_usd": float(cost) if cost is not None else None,
            "served_model": headers.get("Otari-Model", resp.model or model),
            "provider": headers.get("Otari-Provider", ""),
            "fallback": headers.get("Otari-Fallback", "false").lower() == "true",
            "request_id": headers.get("Otari-Request-ID", ""),
            "attempts": int(headers.get("Otari-Attempt-Count", "1") or 1),
            "guardrails": headers.get("Otari-Guardrails", ""),
            "latency_ms": latency,
        }

    def chat_json(self, *, role: str, chain: list[str], system: str, user: str,
                  schema: type[T], temperature: float, profile: str, ticker: str,
                  as_of: str, conversation_id: str, end_user: str | None = None,
                  guardrails: list[dict] | None = None) -> T:
        system_full = (f"{system}\n\nAnswer with one JSON object only, no prose, matching this JSON schema:\n"
                       f"{schema_hint(schema)}")
        tags = {"role": role, "profile": profile, "run": self.run_id, "ticker": ticker, "app": "investment-committee"}
        if end_user:
            tags["end_user"] = end_user
        last_error = ""
        for i, model in enumerate(chain):
            messages = [{"role": "system", "content": system_full}, {"role": "user", "content": user}]
            for attempt in range(2):  # second attempt = repair
                rec = CallRecord(run_id=self.run_id, profile=profile, role=role, ticker=ticker, as_of=as_of,
                                 requested_model=model, client_fallback=i > 0, repaired=attempt > 0,
                                 end_user=end_user or "")
                ckey = self.cache.key(model, messages, temperature)
                try:
                    hit = self.cache.get(ckey)
                    if hit:
                        out, rec.cached = hit, True
                    else:
                        out = self._call(role, model, messages, temperature, tags, conversation_id,
                                         end_user, guardrails)
                    rec.served_model = out["served_model"]
                    rec.provider = out["provider"]
                    rec.fallback = out["fallback"]
                    rec.prompt_tokens = out["prompt_tokens"]
                    rec.completion_tokens = out["completion_tokens"]
                    rec.cost_usd = 0.0 if rec.cached else out["cost_usd"]
                    rec.latency_ms = 0 if rec.cached else out["latency_ms"]
                    rec.request_id = out["request_id"]
                    rec.attempts = out.get("attempts", 1)
                    rec.guardrails = out.get("guardrails", "")
                except Exception as exc:  # network, budget_exceeded (403), provider errors
                    rec.ok, rec.error = False, f"{type(exc).__name__}: {exc}"[:300]
                    self.ledger.add(rec)
                    last_error = rec.error
                    break  # next model in chain
                try:
                    parsed = schema.model_validate(extract_json(out["text"]))
                except (ValueError, ValidationError) as exc:
                    rec.ok, rec.error = False, f"invalid output (finish={out.get('finish_reason', '')}): {exc}"[:300]
                    self.ledger.add(rec)
                    last_error = rec.error
                    messages = messages + [
                        {"role": "assistant", "content": out["text"][:4000]},
                        {"role": "user", "content": f"That answer was invalid ({str(exc)[:300]}). "
                                                    "Reply again with one valid JSON object only."},
                    ]
                    continue
                if not hit:
                    self.cache.put(ckey, out)
                self.ledger.add(rec)
                return parsed
        raise LLMError(f"{role}: all models failed ({last_error})")

    # ---------- non-chat Otari endpoints ----------
    def _post(self, path: str, payload: dict, *, role: str, kind: str, profile: str, ticker: str,
              as_of: str, user: str | None = None, timeout: float = 90) -> dict:
        import requests
        rec = CallRecord(run_id=self.run_id, profile=profile, role=role, ticker=ticker, as_of=as_of,
                         requested_model=path, kind=kind, end_user=user or "")
        if user:
            payload = {**payload, "user": user}
        t0 = time.perf_counter()
        try:
            r = requests.post(f"{self.settings.base_url}/{path}", json=payload, timeout=timeout,
                              headers={"Authorization": f"Bearer {self.settings.key_for(role, user)}"})
            rec.latency_ms = int((time.perf_counter() - t0) * 1000)
            rec.request_id = r.headers.get("Otari-Request-ID", "")
            if r.status_code >= 400:
                raise LLMError(f"{path} {r.status_code} {r.headers.get('Otari-Error-Code', '')}: {r.text[:200]}")
            data = r.json()
            cost = (data.get("usage") or {}).get("cost_usd")
            rec.cost_usd = float(cost) if cost is not None else None
            return data
        except Exception as exc:
            rec.ok, rec.error = False, f"{type(exc).__name__}: {exc}"[:300]
            raise LLMError(rec.error) from exc
        finally:
            self.ledger.add(rec)

    def fetch(self, url: str, max_chars: int = 6000, **ctx) -> str:
        """Read a public page through Otari's web fetch (POST /fetch)."""
        data = self._post("fetch", {"url": url, "max_chars": max_chars}, kind="fetch", **ctx)
        for k in ("content", "text", "markdown"):
            if data.get(k):
                return str(data[k])
        results = data.get("results") or []
        return str(results[0].get("content", "")) if results else ""

    def execute(self, code: str, timeout_seconds: int = 30, **ctx) -> dict:
        """Run Python in Otari's sandbox (POST /code/execute), no model involved."""
        return self._post("code/execute", {"code": code, "timeout_seconds": timeout_seconds}, kind="code",
                          timeout=timeout_seconds + 60, **ctx)

    def remaining_budget(self, role: str, user: str | None = None) -> float | None:
        """Dollars the key can still spend (GET /key/budget), None when no dollar cap applies."""
        import requests
        try:
            r = requests.get(f"{self.settings.base_url}/key/budget", timeout=20,
                             headers={"Authorization": f"Bearer {self.settings.key_for(role, user)}"})
            r.raise_for_status()
            v = r.json().get("remaining_usd")
            return float(v) if v is not None else None
        except Exception:
            return None

    def usage_by_role(self, role: str = "judge") -> dict:
        """Ask Otari for this key's spend grouped by the `role` tag (reconciles with the ledger)."""
        import requests
        r = requests.get(f"{self.settings.base_url}/key/usage/summary",
                         params={"group_by_tag": "role", "dimensions": "none"},
                         headers={"Authorization": f"Bearer {self.settings.key_for(role)}"},
                         timeout=30)
        r.raise_for_status()
        return r.json()

    def list_models(self) -> list[str]:
        client = self._client("judge")
        return [m.id for m in client.models.list().data]
