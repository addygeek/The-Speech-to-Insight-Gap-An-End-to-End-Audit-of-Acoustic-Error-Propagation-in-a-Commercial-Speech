# Critical spans

Source: `metrics.csv` column `critical_span_error_rate`, plus a type-level recount of the strong transcript against `data/references.jsonl`. The normalizer lowercases, strips punctuation, collapses whitespace, and does not rewrite numbers. A span counts as an error when that normalized string is present on one side and absent on the other.

## The headline rate

| Condition | Mean rate | Utterances with at least one flagged span |
|---|---|---|
| Clean | 0.166 | 44.8% |
| Mild | 0.166 | 44.7% |
| Strong | 0.166 | 44.7% |

The median rate is 0 in every condition. Noise does not change this number. The rate is a property of the sentences and the matcher.

Annotations on the 600 references: place 317, name 259, date 256, amount 187, time 169, negation 152, action 65, number 43.

## By type, strong transcript

| Type | Flagged / annotated | Rate | Span text absent from the reference itself |
|---|---|---|---|
| Time | 169 / 169 | 1.000 | 0 |
| Amount | 97 / 187 | 0.519 | 0 |
| Name | 3 / 259 | 0.012 | 2 |
| Place | 1 / 317 | 0.003 | 3 |
| Date | 0 / 256 | 0.000 | 1 |
| Negation | 0 / 152 | 0.000 | 39 |
| Number | 0 / 43 | 0.000 | 0 |
| Action | 0 / 65 | 0.000 | 0 |

Names, places, dates, negations, plain numbers, and action phrases are almost never flagged. Times are always flagged. About half of currency spans are flagged.

## Why times and amounts look failed

Punctuation removal glues digits together, so the substring test misses strings a reader would count as present.

- Span `$250.00` becomes `25000`. The transcript contains `250`. The dollars survived. The metric still flags the span because `25000` is not a substring of the transcript.
- Span `9:30 AM` becomes `930 am`. The transcript contains `0930am`. The clock time survived. The metric flags it because of the leading zero and the missing space.

Thirty-nine negation spans are not literal substrings of the reference sentence, so those annotations cannot be scored as kept or lost. Both sides fail the substring test, and the comparison records no error.

## How to write it

Put 0.166 in the table, then the type table in the same paragraph. The supported reading is that, under this normalizer, formatting of times and currency dominates the critical-span flags, and names, places, dates, and negations stay stable across the strong mix. The rate does not move from clean to strong, so it is not an effect of the 8 dB mix.
