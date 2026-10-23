# Speech-to-Insight Gap

Reproducible evaluation harness for measuring how acoustic degradation and transcription errors relate to Deepgram downstream sentiment, intent, and topic outputs.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Add DEEPGRAM_API_KEY and OPENROUTER_API_KEY (gpt-4o-mini paraphrases).
```

## Configuration

Edit [`config/experiment.yaml`](config/experiment.yaml) for pilot/full sizes, acoustic SNR levels, Deepgram models, and budget caps (`max_usd: 20`).

## Run (dry-run first — no API charges)

Default config has `run.dry_run: true`. Explicit dry-run:

```bash
python -m src.cli run-pilot --dry-run
python -m src.cli paper-assets
pytest
```

## Live pilot (after `.env` is set)

```bash
python -m src.cli run-pilot --live
python -m src.cli preflight-cost --target 450
python -m src.cli validate-results
```

## Full experiment

```bash
python -m src.cli run-full --live --validate
python -m src.cli paper-assets
```

## Artifacts (single results tree)

| Path | Contents |
|------|----------|
| `results/raw_responses/` | Cached TTS/listen/read JSON |
| `results/transcripts.csv` | ASR transcripts |
| `results/intelligence.csv` | Normalized sentiment/intent/topics |
| `results/metrics.csv` | WER + disagreement per utterance/condition |
| `results/summary.json` | Validation + statistics |
| `results/cost_report.json` | Spend estimates and preflight |
| `results/failures.csv` | Failed API calls |

Audio files live under `data/audio/` (gitignored).

## Stage commands

`generate-dataset`, `validate-dataset`, `tts`, `perturb`, `validate-audio`, `listen`, `read`, `metrics`, `stats`, `paper-assets`, `preflight-cost`, `run-pilot`, `run-full`, `validate-results`

All support `--config`, `--no-resume`, and `--max-requests`. Use `--live` to call Deepgram.

## Paper

LaTeX sources in [`paper/`](paper/). Compile after generating `tables/` and `figures/` via `paper-assets`.

## Planning docs

- [`idea.md`](idea.md) — research framing  
- [`detiael.md`](detiael.md) — design and budget notes  
- [`prompt.md`](prompt.md) — full project specification  
