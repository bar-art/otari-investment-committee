"""Configuration: Otari connection, model profiles and risk limits."""
from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

ROLES = ("fundamental", "technical", "events", "bull", "bear", "judge", "risk")

# Roles that share a model slot in profiles.toml
ROLE_SLOT = {
    "fundamental": "analyst",
    "technical": "analyst",
    "events": "analyst",
    "bull": "debater",
    "bear": "debater",
    "judge": "judge",
    "risk": "risk",
}

DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META",
    "JPM", "CVX", "JNJ", "PG", "HD", "CAT",
]


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class OtariSettings:
    base_url: str = os.getenv("OTARI_BASE_URL", "https://api.otari.ai/api/v1")
    default_key: str = os.getenv("OTARI_API_KEY", "")
    # A service key bills each request to an end user (`user` field) with its own budget.
    service_key: str = os.getenv("OTARI_SERVICE_KEY", "")
    end_user_budget: str = os.getenv("OTARI_END_USER_BUDGET", "")  # budget id new end users start on
    timeout_s: float = float(os.getenv("OTARI_TIMEOUT_S", "90"))
    web_fetch: bool = _flag("OTARI_WEB_FETCH")             # read 8-K text through POST /fetch
    code_check: bool = _flag("OTARI_CODE_CHECK", "1")      # recompute VaR/ES in Otari's sandbox
    guardrail_profile: str = os.getenv("OTARI_GUARDRAIL_PROFILE", "")   # e.g. prompt-injection
    guardrail_mode: str = os.getenv("OTARI_GUARDRAIL_MODE", "block")
    judge_budget_floor_usd: float = float(os.getenv("JUDGE_BUDGET_FLOOR_USD", "0.50"))

    def key_for(self, role: str, user: str | None = None) -> str:
        """End-user requests go on the service key; otherwise the agent's own virtual key
        (OTARI_API_KEY_JUDGE, ...) so each agent has its own budget."""
        if user and self.service_key:
            return self.service_key
        return os.getenv(f"OTARI_API_KEY_{role.upper()}", "") or self.default_key


@dataclass(frozen=True)
class Profile:
    name: str
    description: str
    models: dict[str, str]                 # slot -> model selector
    fallbacks: dict[str, list[str]]        # slot -> client-side fallback chain
    knowledge_cutoff: str | None = None    # latest training cutoff among the models
    debate_rounds: int = 1
    temperature: dict[str, float] = field(default_factory=dict)
    economy_judge: str | None = None       # used when the judge key's budget runs low

    @property
    def uses_llm(self) -> bool:
        return bool(self.models)

    def model_for(self, role: str) -> str:
        return self.models[ROLE_SLOT[role]]

    def chain_for(self, role: str) -> list[str]:
        slot = ROLE_SLOT[role]
        return [self.models[slot], *self.fallbacks.get(slot, [])]

    def temperature_for(self, role: str) -> float:
        default = {"analyst": 0.2, "debater": 0.5, "judge": 0.2, "risk": 0.0}
        slot = ROLE_SLOT[role]
        return self.temperature.get(slot, default[slot])


def load_profiles(path: Path | None = None) -> dict[str, Profile]:
    path = path or Path(os.getenv("COMMITTEE_PROFILES", ROOT / "profiles.toml"))
    raw = tomllib.loads(Path(path).read_text())
    profiles = {}
    for name, cfg in raw.items():
        profiles[name] = Profile(
            name=name,
            description=cfg.get("description", ""),
            models=cfg.get("models", {}),
            fallbacks=cfg.get("fallbacks", {}),
            knowledge_cutoff=cfg.get("knowledge_cutoff"),
            debate_rounds=int(cfg.get("debate_rounds", 1)),
            temperature=cfg.get("temperature", {}),
            economy_judge=cfg.get("economy_judge"),
        )
    return profiles


@dataclass(frozen=True)
class RiskLimits:
    """Deterministic limits. No LLM output can loosen these."""
    max_weight: float = float(os.getenv("MAX_WEIGHT", "0.10"))
    max_sector_weight: float = float(os.getenv("MAX_SECTOR_WEIGHT", "0.30"))
    max_gross: float = float(os.getenv("MAX_GROSS", "1.0"))
    vol_budget: float = float(os.getenv("VOL_BUDGET", "0.03"))       # weight * annual vol
    es_budget: float = float(os.getenv("ES_BUDGET", "0.004"))        # weight * 1d ES97.5
    drawdown_stop: float = float(os.getenv("DRAWDOWN_STOP", "0.15"))  # block new buys
    stale_fundamentals_days: int = int(os.getenv("STALE_FUNDAMENTALS_DAYS", "200"))
    min_history_days: int = 126


@dataclass(frozen=True)
class DataSettings:
    cache_dir: Path = Path(os.getenv("COMMITTEE_CACHE", ROOT / ".cache"))
    # SEC asks automated clients to identify themselves: "Name contact@domain"
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "otari-investment-committee research@example.org")
