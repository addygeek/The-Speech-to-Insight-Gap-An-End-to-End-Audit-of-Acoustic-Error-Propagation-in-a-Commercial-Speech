# Sentiment

Source: `intelligence.csv` columns `sentiment_label` and `sentiment_score`, and `metrics.csv` columns `sentiment_disagreement_ref_vs_transcript` and `sentiment_disagreement_audio_vs_text`. A disagreement is any mismatch among `positive`, `negative`, and `neutral` after lowercasing. N = 600 per text source, 1,800 audio recordings.

## What the text model says about the original sentences

`/v1/read` on the reference text is 50.7% neutral, 31.7% negative, and 17.7% positive. The mean sentiment score on those 600 texts is −0.091.

Transcript `/read` stays in that band.

| Source | Neutral | Negative | Positive | Mean score |
|---|---|---|---|---|
| Clean transcript | 50.8% | 31.7% | 17.5% | −0.102 |
| Mild transcript | 50.8% | 31.7% | 17.5% | −0.101 |
| Strong transcript | 50.5% | 31.8% | 17.7% | −0.101 |

## Reference text versus transcript

| Condition | Disagreement rate | Count |
|---|---|---|
| Clean | 0.0017 | 1 of 600 |
| Mild | 0.0017 | 1 of 600 |
| Strong | 0.0050 | 3 of 600 |

`tables/table3_disagreement.tex` is this rate only.

The 44 utterances whose strong transcript differs from the clean transcript account for most of the strong-condition flips: disagreement 0.045 on those 44 (2 sentences) and 0.0018 on the other 556 (1 sentence). A changed transcript is where a sentiment flip can occur. The count in the whole study is still three strong-condition flips.

## Audio versus text

`/v1/listen` sentiment and `/v1/read` sentiment on that recording’s transcript disagree on 0 of 1,800 rows.

Audio sentiment itself matches between clean and mild for all 600 utterances, and between clean and strong for 598 of 600.

## Author tag versus the API

`expected_sentiment` on the reference record matches reference `/v1/read` for 425 of 600 sentences (70.8%). The other 175 are the API and the template label disagreeing before any noise is applied. That 70.8% is agreement with the label written when the sentence was generated. It is separate from the acoustic condition, and it is separate from human ground truth.

## Association with word error

Spearman correlation of WER with reference-versus-transcript sentiment disagreement: rho = 0.027, p = 0.253, n = 1,800. `figures/fig3_wer_vs_disagreement.pdf` is this relationship. Sentiment disagreement is nearly constant, so there is little for WER to explain.

## How to write it

Sentiment is the stable output. It survives the transcript, it matches between the audio API and the text API, and it barely moves between clean and strong audio. Keep it separate from intent and topics, which do not share this stability.
