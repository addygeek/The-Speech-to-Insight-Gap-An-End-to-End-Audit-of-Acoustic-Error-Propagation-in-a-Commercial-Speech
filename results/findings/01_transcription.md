# Transcription

Source: `metrics.csv` columns `wer`, `cer`, `substitutions_rate`, `deletions_rate`, `insertions_rate`, and the transcripts in `intelligence.csv` where `text_source` is `audio`. N = 600 utterances in each condition. Bootstrap intervals are from `summary.json` (2,000 resamples, seed 42).

Word error rate compares the reference sentence with the Nova-3 transcript after lowercasing, stripping punctuation, and collapsing whitespace. Numbers are not normalized.

## Means

| Condition | Mean WER | 95% interval | Median | WER = 0 | WER > 0.10 | Mean CER |
|---|---|---|---|---|---|---|
| Clean | 0.032 | 0.029–0.035 | 0.012 | 50.0% | 2.3% | 0.012 |
| Mild (15 dB) | 0.032 | 0.029–0.035 | 0.000 | 50.7% | 2.2% | 0.011 |
| Strong (8 dB) | 0.032 | 0.029–0.035 | 0.000 | 51.0% | 2.3% | 0.011 |

The intervals overlap completely. These two noise levels did not move average transcription accuracy on this voice.

More precise means, for the appendix: clean 0.0322, mild 0.0319, strong 0.0318. Character error: 0.0117, 0.0114, 0.0113.

## What kind of error

On the clean recordings the mean rates are substitutions 0.023, deletions 0.009, insertions 0.0001. Mild and strong match that pattern (substitutions 0.023 and 0.022, deletions 0.009, insertions 0.0001). The recognizer sometimes replaces or drops a word. It almost never adds one.

## Paired result

Each sentence has three recordings, so the comparison that matters is within the sentence.

- For 96.0% of utterances, clean, mild, and strong WER are equal.
- Mean absolute change from clean to mild is 0.0008 WER. From clean to strong it is 0.0014.
- Strong WER is higher than clean for 1.3% of utterances and lower for 2.3%. The strong mix is not a consistent step down.

The transcripts themselves:

| Pair | Identical transcripts |
|---|---|
| Clean and mild | 97.2% |
| Mild and strong | 93.7% |
| Clean and strong | 92.7% |

Clean and strong differ for 44 utterances. On those 44, mean strong WER is 0.045. On the other 556, mean strong WER is 0.031. The noise design produced a small subset of changed transcripts, not a shift of the whole set.

`figures/fig2_wer_by_condition.pdf` and `tables/table2_wer.tex` show the flat distributions. The table’s standard deviation is 0.038 in every condition.

## How to write it

Report the three means with the intervals, then the paired facts: 96% equal WER, 44 of 600 strong transcripts differ from clean. Follow with the scenario table in `05_scenarios.md`, because the WER that does exist sits in appointment scheduling (about 0.097) and technical support (about 0.067), not in the noise factor.

This bounds the noise used here: generated pink noise at 15 dB and 8 dB on one synthetic voice.
