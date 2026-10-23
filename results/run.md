# What the run is

This file describes the live study that produced the tables in this folder. Counts were taken from the files on disk on 23 October 2026. The short numeric summary of the outcomes is `findings.md`. This document is the account of how those outcomes were produced.

## The material

The run starts from `data/references.jsonl`: 600 English customer-service sentences, every one marked `dataset_split = full`. Each sentence is 25–40 words (mean 30.39, median 30). Four hundred were filled from templates. Two hundred were paraphrases (`generation_template_id = llm_paraphrase`). Each record carries a scenario, an author `expected_sentiment`, an author `expected_intent`, and a list of critical spans (place, name, date, amount, time, negation, action, number).

Those author labels are the study’s own tags. They are not the Deepgram labels. `metrics.csv` never scores the API against `expected_intent`. It scores API output against API output.

One voice speaks every sentence. There is no second speaker, no telephone codec, and no second vendor.

## The path each sentence takes

For each of the 600 sentences the pipeline does five things, in this order.

1. **Speech.** Deepgram Aura, model `aura-2-asteria-en`, returns 16-bit mono WAV at 24,000 Hz. The call is saved in `raw_responses/tts/{utterance_id}_clean.json` with the sentence, the request id, the HTTP status, and the byte length. There are 600 of these files. A checked file is status 200. The playable file is `data/audio/{utterance_id}/clean.wav`.

2. **Noise.** The clean WAV is copied twice and mixed with generated pink noise. Mild is 15 dB signal-to-noise. Strong is 8 dB. The noise is not a recorded room. It is synthesized, scaled so the noise power matches that SNR, and added to the speech. Each mix writes `data/audio/{utterance_id}/mild.wav` or `strong.wav`, plus a `.meta.json` that records `noise_source = generated_pink`, the SNR, and the noise seed actually used. The config base seed is 42. The per-file seed stored in the meta file is `42 + hash(utterance_id + condition) % 10000`. Python’s `hash` changes across processes unless `PYTHONHASHSEED` is fixed, so a later remix can differ. The WAV files in `data/audio` are the exact files that were sent to transcription. Cite those files, and the seeds in their meta JSON, as the acoustic condition.

3. **Transcription and audio intelligence.** Each of the 1,800 WAV files is posted to `POST /v1/listen`. The saved query parameters are `model=nova-3`, `language=en`, `smart_format=true`, `sentiment=true`, `intents=true`, `topics=true`. The body is the WAV. The response JSON is `raw_responses/listen/{utterance_id}_{condition}.json`. Each file has `request_id`, `audio_minutes`, `timestamp_utc`, `meta` (`utterance_id`, `condition`, `text_source = audio`), and `response.results` with `channels` (the transcript), `sentiments`, `intents`, and `topics`. The folder holds 600 clean, 600 mild, and 600 strong files. Summed `audio_minutes` is 296.94, which is 9.9 seconds per file and about 99 minutes of unique speech.

4. **Text intelligence.** `POST /v1/read` is called with `language=en` and sentiment, intents, and topics turned on. The body is `{"text": "..."}`. Four texts are sent for every sentence: the original reference, the clean transcript, the mild transcript, and the strong transcript. That is 2,400 calls, and `raw_responses/read/` contains 600 of each (`{id}_reference.json`, `{id}_transcript_clean.json`, `{id}_transcript_mild.json`, `{id}_transcript_strong.json`). A read file stores `meta.text` (the exact string sent), `meta.text_source`, `request_id`, and `response.results` with `sentiments`, `intents`, and `topics`. Read responses have no transcript channel. The words they scored are the string in `meta.text`.

5. **Tables.** Parsing flattens the 4,200 responses into `intelligence.csv` (4,200 data rows: 1,800 audio-native and 2,400 text-native) and `transcripts.csv` (1,800 rows, one transcript per recording). `metrics.csv` then has 1,800 rows, one per utterance per condition. `summary.json` adds bootstrap intervals (2,000 resamples, seed 42), scenario means, and the Spearman correlation. `findings.md` and `results_section.tex` are written from those two files only.

Nothing else was sent. Keyterm prompting on the strong recordings was off. The planned second pass over 40 clean recordings was off. No human relabeled the API output.

## What each product is allowed to mean

`/v1/listen` hears the WAV. Its transcript is the hypothesis for word error rate. Its sentiment, intent, and topics are the audio-native scores. Those scores are what the columns `*_audio_vs_text` compare.

`/v1/read` never hears audio. On the reference sentence it is the text baseline. On a transcript it is the same text model looking at whatever the recognizer wrote. Those scores are what the columns `*_ref_vs_transcript` compare.

Word error rate compares the reference sentence with the listen transcript after lowercasing, stripping punctuation, and collapsing whitespace. Numbers are not rewritten (`$250.00` and `250` stay different tokens once the symbols are gone). Critical-span error rate is a substring test under that same normalizer. It is not a judgment that a name or a time was absent to a listener. `findings.md` reports the rate 0.166. That rate needs the type breakdown before it is interpreted: under this normalizer, time and currency formatting account for almost all of the flags, while names, places, dates, and negations are rarely flagged.

Sentiment disagreement is a mismatch of the three labels `positive`, `negative`, and `neutral`. Intent disagreement is an exact match of the API’s own phrase after lowercasing and a three-line synonym map (`cancel order` and `cancel my order` become `cancellation`; `book appointment` and `schedule appointment` become `scheduling`). Topic disagreement is whether the set of topic strings changed. Jaccard is the overlap of those sets. None of these is accuracy against a human.

## The three comparisons sitting in metrics.csv

| Question | Held fixed | Where it is |
|---|---|---|
| Did the noise change the words? | Same sentence, clean vs mild vs strong | `wer`, `cer`, and the three transcripts |
| Did the transcript change the text scores? | `/v1/read` on the reference vs `/v1/read` on the transcript | columns ending `_ref_vs_transcript` |
| Do the audio scores match the text scores? | `/v1/listen` vs `/v1/read` on that recording’s transcript | columns ending `_audio_vs_text` |

The measured answer, already in `findings.md`, is that mean WER stays 0.032 in every condition, sentiment almost never flips, and intent and topic strings disagree on most items. That pattern is why the run was built as three comparisons rather than as a single accuracy number.

## Cost record

`cost_report.json` is an estimate, not an invoice.

| Line | Amount | How it was priced |
|---|---|---|
| Aura TTS | $2.90 | $0.030 per 1,000 characters |
| Nova-3 | $0.13 | $0.0043 per minute on the 296.94 minutes |
| `/v1/read` | $0.00 | `read_per_request` is 0 in the config because no public per-request price was entered |
| Total | $3.03 | 3,177 counted requests |

The paper can say speech was estimated at about $3.03. It should not say text intelligence was free. The 2,400 read calls happened. Their price is unknown to this file.

The tracker cap was $40. The run did not stop on cost.

## What the failure log is

`failures.csv` has 1,553 rows: 16 listen and 1,537 read.

The 16 listen rows are recordings that timed out or lost the network during the parallel pass. A later pass rewrote them. `raw_responses/listen` now has 1,800 complete JSON files, 600 per condition.

The 1,537 read rows are the first text-analysis attempt. Deepgram returned HTTP 400 because `language` was missing, and that attempt saved nothing. The successful attempt sent `language=en` and wrote all 2,400 read files. The failure log is a history of rejected calls. It is not a list of missing results.

## Files to cite

| File | Rows or files | Role in the paper |
|---|---|---|
| `data/references.jsonl` | 600 | The sentences, scenarios, author labels, critical spans |
| `data/audio/{id}/{clean,mild,strong}.wav` | 1,800 | The audio that was transcribed |
| `raw_responses/tts/` | 600 | Aura request ids |
| `raw_responses/listen/` | 1,800 | Transcripts and audio-native scores |
| `raw_responses/read/` | 2,400 | Text-native scores, with the exact text in `meta.text` |
| `transcripts.csv` | 1,800 | Hypothesis strings used for WER |
| `intelligence.csv` | 4,200 | Labels, scores, topics, request ids |
| `metrics.csv` | 1,800 | One comparison row per utterance per condition |
| `summary.json` | — | Intervals, scenario means, Spearman |
| `findings.md` | — | The short measured paragraph |
| `cost_report.json` | — | The $3.03 speech estimate |
