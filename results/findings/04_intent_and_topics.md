# Intent and topics

Source: `metrics.csv` intent and topic columns, and `intelligence.csv` columns `intent_normalized` and `topics`. Intent disagreement is exact string match after lowercasing and this synonym map: `cancel order` and `cancel my order` become `cancellation`; `book appointment` and `schedule appointment` become `scheduling`. Topics are split on `|`. A set change means the two sets differ. Jaccard is their overlap. Empty sets on both sides count as Jaccard 1.

## Intent, reference text versus transcript

| Condition | Disagreement |
|---|---|
| Clean | 0.605 |
| Mild | 0.637 |
| Strong | 0.660 |

## Intent, audio-native versus text-native

| Condition | Disagreement |
|---|---|
| Clean | 0.627 |
| Mild | 0.643 |
| Strong | 0.645 |

## Why the rate is high while WER is 0.032

The API does not emit the closed label stored in `expected_intent`. It writes a fresh phrase.

- Reference `/v1/read` produced 233 distinct intent strings for 600 sentences.
- Clean `/v1/listen` produced 245 distinct intent strings.
- Exact match between `expected_intent` and reference `/v1/read` is 131 of 600 (21.8%).

The open phrasing shows up in the intelligence file as pairs such as `change flight` and `change flight to seattle`, or `check package status` and a longer variant of the same request. Exact match treats those as disagreements. The words of the sentence can be right and the intent string can still change.

On the 556 utterances whose strong transcript equals the clean transcript, intent disagreement against the reference text is already 0.646. On the 44 whose transcript changed, it is 0.841. The strings are unstable before the noise changes any words. A changed transcript raises the rate further.

The synonym map only rescues a few cancellation and scheduling phrases. That is why requests and cancellations show intent disagreement of about 0.05 to 0.08 in `05_scenarios.md`, while complaints and banking stay above 0.74.

## Topics, reference text versus transcript

| Condition | Set changed | Mean Jaccard | Share with Jaccard 0 |
|---|---|---|---|
| Clean | 0.795 | 0.421 | 24.8% |
| Mild | 0.818 | 0.406 | 26.5% |
| Strong | 0.803 | 0.415 | 27.2% |

Median Jaccard is 0.333. The 90th percentile reaches 1, so some pairs match exactly. About a quarter share no topic string at all.

On the 44 changed strong transcripts the topic set changes 88.6% of the time. On the 556 unchanged transcripts it still changes 79.7% of the time. Topic lists move whether or not the recognizer changed the sentence.

## Topics, audio-native versus text-native

| Condition | Set changed | Mean Jaccard |
|---|---|---|
| Clean | 0.790 | 0.427 |
| Mild | 0.822 | 0.411 |
| Strong | 0.822 | 0.416 |

`/v1/listen` and `/v1/read` agree on sentiment for every recording. They do not agree on the intent phrase or the topic set. The pipeline gap and the transcription gap are different results. Sentiment transfers from audio to text. Intent and topics do not, even when both calls see the same words.

## How to write it

Call intent and topic outputs unstable open strings. Give the 60.5 / 63.7 / 66.0 intent rates and the roughly 80% topic-set rates. Then give the 556 unchanged transcripts, where intent still disagrees 64.6% of the time, so the reader does not attribute the rate to the 8 dB noise. Mention the 233 distinct reference intents and the 21.8% match to the author label so “disagreement” is not read as “the recognizer missed the request.”
