"""Load experiment configuration and resolve paths relative to repo root."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclass
class ExperimentConfig:
    raw: dict[str, Any]
    root: Path = field(default_factory=repo_root)

    @property
    def dry_run(self) -> bool:
        return bool(self.raw.get("run", {}).get("dry_run", True))

    @property
    def resume(self) -> bool:
        return bool(self.raw.get("run", {}).get("resume", True))

    @property
    def random_seed(self) -> int:
        return int(self.raw.get("project", {}).get("random_seed", 42))

    @property
    def pilot_utterances(self) -> int:
        return int(self.raw.get("run", {}).get("pilot_utterances", 25))

    @property
    def full_utterances(self) -> int:
        return int(self.raw.get("run", {}).get("full_utterances", 450))

    def path(self, key: str) -> Path:
        rel = self.raw.get("paths", {}).get(key, key)
        return (self.root / rel).resolve()

    @property
    def results_dir(self) -> Path:
        return self.path("results_dir")

    @property
    def references_jsonl(self) -> Path:
        return self.path("references_jsonl")

    @property
    def manifest_csv(self) -> Path:
        return self.path("manifest_csv")

    @property
    def audio_dir(self) -> Path:
        return self.path("audio_dir")

    @property
    def figures_dir(self) -> Path:
        return self.path("figures_dir")

    @property
    def tables_dir(self) -> Path:
        return self.path("tables_dir")

    def section(self, name: str) -> dict[str, Any]:
        val = self.raw.get(name, {})
        return val if isinstance(val, dict) else {}


def load_config(config_path: str | Path | None = None) -> ExperimentConfig:
    load_dotenv(repo_root() / ".env")
    root = repo_root()
    path = Path(config_path) if config_path else root / "config" / "experiment.yaml"
    if not path.is_absolute():
        path = root / path
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return ExperimentConfig(raw=raw, root=root)


def require_deepgram_key() -> str:
    key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    if not key:
        raise RuntimeError("DEEPGRAM_API_KEY is not set. Copy .env.example to .env and add your key.")
    return key
