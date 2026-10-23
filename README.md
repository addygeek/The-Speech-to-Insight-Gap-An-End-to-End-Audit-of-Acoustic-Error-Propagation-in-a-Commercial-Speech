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
