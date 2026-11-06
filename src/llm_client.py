"""Disk-cached chat completions through OpenRouter, used by the semantic judge and the LLM baselines."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.config_loader import ExperimentConfig
from src.logging_utils import setup_logging

logger = setup_logging()

JUDGE_MODEL = "openai/gpt-4o-mini"
LLM_WORKERS = 8


def _cache_file(cfg: ExperimentConfig, model: str, temperature: float, prompt: str, tag: str) -> Path:
    digest = hashlib.sha256(json.dumps([model, temperature, prompt, tag]).encode()).hexdigest()
    return cfg.results_dir / "llm_cache" / f"{digest}.json"


@retry(retry=retry_if_exception_type(httpx.TransportError), wait=wait_exponential(min=1, max=20), stop=stop_after_attempt(4), reraise=True)
def _post(base: str, key: str, body: dict[str, Any]) -> httpx.Response:
    return httpx.post(f"{base}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body, timeout=90.0)


def _strip_fence(text: str) -> str:
    """Some models wrap the JSON in a markdown code fence even when asked for a JSON object."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def chat_json(cfg: ExperimentConfig, prompt: str, model: str = JUDGE_MODEL, temperature: float = 0.0, tag: str = "") -> dict[str, Any]:
    """Return the parsed JSON object the model answers with. `tag` separates cache entries for deliberate repeats."""
    path = _cache_file(cfg, model, temperature, prompt, tag)
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is not set.")
    base = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    resp = _post(
        base,
        key,
        {
            "model": model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        },
    )
    resp.raise_for_status()
    parsed = json.loads(_strip_fence(resp.json()["choices"][0]["message"]["content"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(parsed), encoding="utf-8")
    return parsed


def map_batches(cfg: ExperimentConfig, prompts: list[str], model: str = JUDGE_MODEL, temperature: float = 0.0, tags: list[str] | None = None) -> list[dict[str, Any]]:
    tags = tags or [""] * len(prompts)

    def one(args: tuple[str, str]) -> dict[str, Any]:
        try:
            return chat_json(cfg, args[0], model, temperature, args[1])
        except (httpx.HTTPError, ValueError, KeyError) as exc:
            logger.warning("LLM batch failed: %s", exc)
            return {}

    with ThreadPoolExecutor(LLM_WORKERS) as pool:
        return list(pool.map(one, zip(prompts, tags)))
