# Measured findings

Every number below is computed from results/metrics.csv and results/summary.json.
Missing files are reported as missing. No value is filled in by hand.

Rows: 1800. Unique utterances: 600.
Exploratory flag: False.

## By acoustic condition

- clean: n=600, mean WER=0.032 (95% bootstrap interval 0.029 to 0.035), mean CER=0.012, critical-span error rate=0.166, sentiment disagreement=0.002, intent disagreement=0.605, topic-set change=0.795, topic Jaccard=0.421.
- mild: n=600, mean WER=0.032 (95% bootstrap interval 0.029 to 0.035), mean CER=0.011, critical-span error rate=0.166, sentiment disagreement=0.002, intent disagreement=0.637, topic-set change=0.818, topic Jaccard=0.406.
- strong: n=600, mean WER=0.032 (95% bootstrap interval 0.029 to 0.035), mean CER=0.011, critical-span error rate=0.166, sentiment disagreement=0.005, intent disagreement=0.660, topic-set change=0.803, topic Jaccard=0.415.

## Association

Spearman correlation of utterance WER with sentiment disagreement: rho=0.027, p=0.2534, n=1800.

## Cost record

Recorded estimate total USD: 3.03.

## What this does not show

These figures describe this synthetic English set and this vendor configuration. Disagreement with the reference-text analysis is not a human ground-truth error. A higher word error rate is not, by itself, a measure of semantic damage.
