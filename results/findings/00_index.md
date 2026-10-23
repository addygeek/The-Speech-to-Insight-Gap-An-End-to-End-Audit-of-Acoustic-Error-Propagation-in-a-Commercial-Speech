# Full findings index

Every number in this folder was recomputed from `results/metrics.csv` (1,800 rows), `results/intelligence.csv` (4,200 rows), `results/summary.json`, and `data/references.jsonl` on 23 October 2026. `results/findings.md` is the short machine-written summary. These files are the in-depth reading of the same run.

How the files were produced is `results/run.md`.

| File | What it covers |
|---|---|
| [01_transcription.md](01_transcription.md) | Word and character error, paired clean vs mild vs strong, the 44 transcripts that changed |
| [02_critical_spans.md](02_critical_spans.md) | The 0.166 critical-span rate, split by span type, and why times and amounts dominate |
| [03_sentiment.md](03_sentiment.md) | Label mix, reference vs transcript, audio vs text, match to the author tags, the Spearman result |
| [04_intent_and_topics.md](04_intent_and_topics.md) | Open intent strings, topic sets, Jaccard, and the audio-native vs text-native gap |
| [05_scenarios.md](05_scenarios.md) | All eight scenarios, all three conditions, every main rate |
| [06_claims.md](06_claims.md) | Sentences the paper can use, and the limits of this run |

## The result in one page

600 sentences, one Aura voice, three conditions (clean, 15 dB pink noise, 8 dB pink noise). Nova-3 transcribes and scores the audio. Text intelligence scores the original sentence and each transcript.

Mean word error rate is 0.032 in every condition. The 95% bootstrap intervals overlap (about 0.029 to 0.035). Half the utterances are exact. Clean and strong transcripts are identical for 556 of 600 sentences.

Sentiment almost never changes with the transcript: 1 of 600 clean, 1 of 600 mild, 3 of 600 strong. Audio-native sentiment and text-native sentiment agree on all 1,800 recordings.

Intent strings disagree 60.5% clean, 63.7% mild, 66.0% strong. Topic sets change about 80% of the time, with mean Jaccard about 0.41. Most of that disagreement is present even when the strong transcript is identical to the clean one (intent 64.6% on those 556 sentences).

The gap in this run is between stable words and unstable open-ended intent and topic strings. It is not a gradient of word error caused by these two noise levels.
