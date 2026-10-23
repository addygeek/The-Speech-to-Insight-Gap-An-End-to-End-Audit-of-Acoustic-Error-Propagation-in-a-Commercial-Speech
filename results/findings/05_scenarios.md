# Results by scenario

Source: `metrics.csv`, grouped by `scenario` and `acoustic_condition`. Utterance counts are from `data/references.jsonl`. A row count of 288 for appointment scheduling is 96 utterances times three conditions. Rates below are within one condition, so the N in the prose is the number of utterances.

Pooled mean WER from `summary.json`, across the three conditions: appointment scheduling 0.097 (288 rows), technical support 0.067 (219), customer support 0.024 (237), banking and payments 0.021 (129), requests and cancellations 0.019 (195), delivery tracking 0.006 (231), travel booking 0.003 (249), complaints and compliments 0.003 (252).

## Appointment scheduling (96 utterances)

Highest word error and a critical-span rate of 0.333, in line with the time-span matcher in `02_critical_spans.md`. Intent disagreement is the lowest among the high-WER scenarios.

| Condition | WER | Sentiment | Intent | Topic change | Topic Jaccard | Audio–text intent | Critical span |
|---|---|---|---|---|---|---|---|
| Clean | 0.097 | 0.010 | 0.396 | 0.781 | 0.518 | 0.427 | 0.333 |
| Mild | 0.096 | 0.010 | 0.479 | 0.760 | 0.524 | 0.500 | 0.333 |
| Strong | 0.097 | 0.010 | 0.510 | 0.688 | 0.563 | 0.542 | 0.333 |

## Technical support (73)

Second-highest WER. Intent and topic disagreement stay high. Critical-span rate is 0.333 in every condition.

| Condition | WER | Sentiment | Intent | Topic change | Topic Jaccard | Audio–text intent | Critical span |
|---|---|---|---|---|---|---|---|
| Clean | 0.067 | 0.000 | 0.822 | 0.877 | 0.300 | 0.849 | 0.333 |
| Mild | 0.068 | 0.000 | 0.863 | 0.849 | 0.279 | 0.836 | 0.333 |
| Strong | 0.067 | 0.000 | 0.877 | 0.945 | 0.242 | 0.904 | 0.333 |

## Customer support (79)

| Condition | WER | Sentiment | Intent | Topic change | Topic Jaccard | Audio–text intent | Critical span |
|---|---|---|---|---|---|---|---|
| Clean | 0.024 | 0.000 | 0.709 | 0.709 | 0.508 | 0.772 | 0.181 |
| Mild | 0.023 | 0.000 | 0.848 | 0.785 | 0.502 | 0.835 | 0.181 |
| Strong | 0.024 | 0.000 | 0.772 | 0.747 | 0.457 | 0.747 | 0.186 |

## Banking and payments (43)

Smallest scenario. WER is low. Topic overlap with the reference is the weakest in the study (Jaccard about 0.17–0.23). Intent disagreement is high.

| Condition | WER | Sentiment | Intent | Topic change | Topic Jaccard | Audio–text intent | Critical span |
|---|---|---|---|---|---|---|---|
| Clean | 0.021 | 0.000 | 0.744 | 0.977 | 0.169 | 0.744 | 0.244 |
| Mild | 0.021 | 0.000 | 0.814 | 0.977 | 0.171 | 0.791 | 0.244 |
| Strong | 0.021 | 0.000 | 0.884 | 0.860 | 0.233 | 0.814 | 0.244 |

## Requests and cancellations (65)

The intent exception. Disagreement stays under 0.08 because those sentences often receive the mapped label `cancellation`. Word error is low. The critical-span rate is still 0.262, consistent with amounts in cancellation sentences.

| Condition | WER | Sentiment | Intent | Topic change | Topic Jaccard | Audio–text intent | Critical span |
|---|---|---|---|---|---|---|---|
| Clean | 0.019 | 0.000 | 0.046 | 0.877 | 0.348 | 0.062 | 0.262 |
| Mild | 0.019 | 0.000 | 0.077 | 0.908 | 0.300 | 0.092 | 0.262 |
| Strong | 0.019 | 0.000 | 0.077 | 0.908 | 0.346 | 0.077 | 0.262 |

## Delivery tracking (77)

Very low WER and a critical-span rate of 0. Intent disagreement stays above 0.80. Strong is one of the few cells with any reference-versus-transcript sentiment disagreement (0.013).

| Condition | WER | Sentiment | Intent | Topic change | Topic Jaccard | Audio–text intent | Critical span |
|---|---|---|---|---|---|---|---|
| Clean | 0.008 | 0.000 | 0.831 | 0.818 | 0.386 | 0.805 | 0.000 |
| Mild | 0.006 | 0.000 | 0.805 | 0.870 | 0.369 | 0.831 | 0.000 |
| Strong | 0.004 | 0.013 | 0.870 | 0.857 | 0.382 | 0.818 | 0.000 |

## Travel booking (83)

WER about 0.003. Topic Jaccard is the highest against the reference (about 0.55–0.59). Intent disagreement is about 0.42–0.43, lower than banking or complaints. Strong sentiment disagreement is 0.012.

| Condition | WER | Sentiment | Intent | Topic change | Topic Jaccard | Audio–text intent | Critical span |
|---|---|---|---|---|---|---|---|
| Clean | 0.003 | 0.000 | 0.434 | 0.711 | 0.545 | 0.506 | 0.006 |
| Mild | 0.003 | 0.000 | 0.422 | 0.759 | 0.555 | 0.410 | 0.006 |
| Strong | 0.003 | 0.012 | 0.422 | 0.711 | 0.592 | 0.470 | 0.006 |

## Complaints and compliments (84)

Low critical-span rate (0.012, 0.012, 0.006) and WER of 0.003. Intent disagreement is among the highest (0.82–0.92). The words are easy and the intent phrase is not stable.

| Condition | WER | Sentiment | Intent | Topic change | Topic Jaccard | Audio–text intent | Critical span |
|---|---|---|---|---|---|---|---|
| Clean | 0.003 | 0.000 | 0.881 | 0.726 | 0.426 | 0.857 | 0.012 |
| Mild | 0.003 | 0.000 | 0.821 | 0.750 | 0.382 | 0.869 | 0.012 |
| Strong | 0.003 | 0.000 | 0.917 | 0.798 | 0.358 | 0.810 | 0.006 |

## How to write it

One pooled intent rate hides the cancellation result, and one pooled WER hides appointment scheduling. Use at least those two contrasts. Within each scenario, the three noise rows sit close together. The scenario identity moves the numbers more than the SNR does.
