# Does the baseline receive the prior scan? Yes, for both Task 5 and Task 6

Settled by reading the v7 code and counting (2026-11-04). Reproduce: `python tests/check_pairing_rate.py`.

**Code.** In `retrain/kaggle_retrain_v7.py`, `VQADataset.__getitem__` sets `is_long = True` when the
task name contains "longitudinal" or "temporal". Both Task 5 (`Static_Temporal`) and Task 6
(`Longitudinal_Temporal`) match, so both go through `_pair`, which stacks the nearest earlier
lettered scan of the same patient (T0) with the referenced scan (T1). The fallback (T1 duplicated)
only triggers if no earlier scan exists or the filename does not parse.

**Count.** Task 5 and Task 6 rows with a genuine earlier scan: train 2,295/2,295, val 238/238,
test 340/340, for each task. No row uses the fallback.

**What this changes.**
- The statement "the baseline never receives the prior scan" in `research_notes/` is wrong for this
  code. The training audit's reading was right.
- The benchmark as released asks about one referenced scan, so the single-volume ceiling in
  `information_ceilings.py` is a statement about the benchmark's information, not about this
  model's input. Keep the two apart in the write-up.
- The blank-CT result is stronger than "the model had no prior to use": the model was given the
  earlier scan as well and its Task 5 and Task 6 answers still did not change when the volume was zeroed.
- A prior-scan ablation (`zero_prior`, `swap_prior` controls) is still needed to separate the two
  volumes, and it is a GPU run.
