# Which "Task 6 text-only" number is which

The repo quotes several Task 6 text-only accuracies. They are the same bracket rule measured on
different row sets and splits, not different findings. All are on Task 6 rows with CT unless noted.

| value | rows | split | source |
| --: | :-- | :-- | :-- |
| 0.7539 | all 16,031 (943 volumes) | none, rule fitted on the same rows | `information_ceilings.py` (pooled) |
| 0.7894 | evaluable 2,873 (169 volumes) | none, rule fitted on the same rows | `information_ceilings.py` (evaluable) |
| 0.8062 | 2,873 rows, test 578 | 80/20 patient split inside `text_only_leak_audit.py` | `text_only_leak_audit.json` |
| 0.7958 | 2,873 rows, test 578 | 80/20 patient split inside `verify_deleak.py` (different patient shuffle) | `verify_deleak.json` |
| 0.7852 / 0.8067 / **0.8059** | manifest train 2,295 / val 238 / **test 340** | the 80/10/10 patient manifests used for training | recomputed from `split_manifests/*_qa.csv`; equals the floors in `inference_audit/RESULTS.md` |

Why they differ: the bracket rule is trivially the same everywhere (two answers are reachable per
bracket group), so the spread, 0.785 to 0.807, is sampling noise from which patients land in which
split. At 238 to 578 items a 95% interval is about ±3 to 6 points, and these rows come from only 169
volumes, so the true interval is wider still.

What to quote:
- **0.789** for the label-structure statement ("78.9% of the evaluable rows are solved before any
  pixel is read"). No split involved.
- **0.8059** (manifest test, n = 340) whenever a model is compared against the floor, because the
  model was evaluated on exactly those rows.
- Do not mix columns within one sentence, and do not call the differences between them findings.
