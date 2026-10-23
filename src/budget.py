"""Budget estimates. Spending is recorded and never blocks a run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from src.config_loader import ExperimentConfig


@dataclass
class BudgetTracker:
    cfg: ExperimentConfig
    spent_usd: float = 0.0
    requests: int = 0
    breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def max_usd(self) -> float:
        return float(self.cfg.section("budget").get("max_usd", 20.0))

    def add(self, category: str, amount: float) -> None:
        self.spent_usd += amount
        self.breakdown[category] = self.breakdown.get(category, 0.0) + amount
        self.requests += 1

    def check(self) -> None:
        """Kept so callers can record spend. Does not stop the run."""
        return None

    def estimate_tts(self, char_count: int) -> float:
        rate = float(
            self.cfg.section("budget")
            .get("pricing_estimates", {})
            .get("aura_per_1k_chars", 0.03)
        )
        return (char_count / 1000.0) * rate

    def estimate_stt_minutes(self, minutes: float, keyterm: bool = False) -> float:
        est = self.cfg.section("budget").get("pricing_estimates", {})
        rate = float(est.get("nova3_per_minute", 0.0043))
        if keyterm:
            rate += float(est.get("keyterm_per_minute", 0.0013))
        return minutes * rate

    def estimate_read(self, n_requests: int) -> float:
        rate = float(
            self.cfg.section("budget").get("pricing_estimates", {}).get("read_per_request", 0.0)
        )
        return n_requests * rate

    def to_dict(self) -> dict:
        return {
            "spent_usd": self.spent_usd,
            "max_usd": self.max_usd,
            "requests": self.requests,
            "breakdown": self.breakdown,
        }

    def load(self, path: Path) -> None:
        if not path.is_file():
            return
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        self.spent_usd = float(data.get("spent_usd", 0.0))
        self.requests = int(data.get("requests", 0))
        breakdown = data.get("breakdown") or {}
        self.breakdown = dict(breakdown) if isinstance(breakdown, dict) else {}

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self.to_dict()
        if path.is_file():
            try:
                with path.open(encoding="utf-8") as f:
                    existing = json.load(f)
            except json.JSONDecodeError:
                existing = {}
            if isinstance(existing, dict) and "preflight" in existing:
                payload["preflight"] = existing["preflight"]
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
