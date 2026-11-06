# Human check of intent pairs

File: `results/human_label_sample.csv` (100 string-disagreeing intent pairs, seeded sample, fixed by `src/semantic_match.py`).

Each row holds two short phrases that the text-intelligence API produced for the same customer call (`a`, `b`).
Fill `human_same` with:

- `1`: both phrases name the same caller goal. A paraphrase, or a different level of detail about the same goal, counts as the same.
- `0`: the goals differ, or one phrase names a goal the other does not.

Rules:

1. Judge the pair from the two phrases only. Do not look at the `judge_same` column until every row is labeled.
2. Do not use the reference sentence or any other context.
3. If both readings are plausible, choose `0`.

Then run `python -m src followup-analysis`. The `human_agreement` block of `results/semantic_match.json` reports the human paraphrase fraction, judge-human accuracy, and Cohen's kappa.
Report the judge-based paraphrase rate as the headline only if kappa against the human labels is at least 0.6; otherwise report the human rate.
