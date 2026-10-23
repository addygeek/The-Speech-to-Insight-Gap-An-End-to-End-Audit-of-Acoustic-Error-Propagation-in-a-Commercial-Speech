# What the paper can say

Tied to the other files in this folder. The design limits are in `results/run.md`.

## Supported

- Across 600 paired utterances, mean word error rate is 0.032 under clean speech, 15 dB pink noise, and 8 dB pink noise. The 95% bootstrap intervals overlap, about 0.029 to 0.035.
- For 96% of utterances the three word-error rates are equal. The strong transcript differs from the clean transcript for 44 of 600 sentences.
- Remaining word errors are mostly substitutions. The mean insertion rate is 0.0001.
- The critical-span rate is 0.166 in every condition. On the strong transcript, 169 of 169 time spans and 97 of 187 amount spans are flagged by the punctuation-stripping matcher, while names (3/259), places (1/317), dates (0/256), and negations (0/152) are rarely flagged.
- Reference-versus-transcript sentiment disagreement is 1, 1, and 3 sentences out of 600.
- Audio-native and text-native sentiment labels agree on all 1,800 recordings. Clean and strong audio sentiment differ for 2 of 600 utterances.
- Intent strings disagree 60.5%, 63.7%, and 66.0% of the time. On the 556 unchanged strong transcripts the rate is still 64.6%.
- Topic sets change for about 80% of items in every condition. Mean Jaccard is about 0.41.
- Reference text intelligence uses 233 distinct intent phrases. Exact match to the author intent tag is 21.8% (131/600). Exact match to the author sentiment tag is 70.8% (425/600).
- Word error and sentiment disagreement are uncorrelated here: Spearman rho 0.027, p 0.253, n = 1,800.
- Appointment scheduling carries the high word error (about 0.097). Requests and cancellations carry the low intent disagreement (under 0.08).

## Outside this run

- Intent and topic changes show up when the transcript is unchanged, so the 8 dB mix is a poor explanation of those rates.
- The 0.166 critical-span rate is carried by time and currency formatting. Names, dates, and negations are a different result, given in the type table.
- API sentiment matches the author tag on 70.8% of reference texts. That is agreement with the generated label.
- Match to the author intent tag (21.8%) and match between two API calls are different quantities. The disagreement rates describe the second one.
- The speech is synthetic, the voice is Aura-2 Asteria, the recognizer is Nova-3, and the noise is generated pink noise at two SNRs.
- `cost_report.json` estimates speech at $3.03. It has no public unit price for `/v1/read`, so the $0 read line is a missing price.
- Shapira et al. (ACL 2025) is prior work on transcript noise and downstream NLP. This run is a paired commercial path with controlled SNR.

## Figures and tables already on disk

- `tables/table1_dataset.tex` — 600 utterances, 1,800 metric rows, three conditions.
- `tables/table2_wer.tex` — mean WER 0.032, standard deviation 0.038, N = 600.
- `tables/table3_disagreement.tex` — sentiment disagreement only (0.002, 0.002, 0.005).
- `figures/fig1_pipeline.pdf` — the path from sentence to scores.
- `figures/fig2_wer_by_condition.pdf` — WER by condition.
- `figures/fig3_wer_vs_disagreement.pdf` — WER against sentiment disagreement.

`results/results_section.tex` restates the three WER means and the Spearman correlation. The prose of the paper should follow these findings files. That generated paragraph can stay as a check that the means have not drifted.
