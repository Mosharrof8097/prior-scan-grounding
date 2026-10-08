# Text-only leakage audit of 3D-RAD

CPU-only analyses of what 3D-RAD's questions give away before any CT scan is
read. Every script writes a `.json` next to itself. No GPU, no model weights.

Run order (each is independent):

| script | what it establishes |
| :-- | :-- |
| `text_only_leak_audit.py` | per-task accuracy reachable from question text alone |
| `compare_to_baseline.py` | that floor vs the trained checkpoint's reported accuracy |
| `task5_task6_twins.py` | Task 5 and Task 6 are the same items, differently phrased |
| `information_ceilings.py` | hard floors/ceilings from the label structure |
| `build_3drad_t.py` | 3D-RAD-T: synthetic de-leaked Task 6 (`T-mask`, `T-strip`) |
| `verify_deleak.py` | that de-leaking removes the shortcut |
| `task6_task4_linkage.py` | Task 4 covers first timepoints only, disjoint from Task 5/6 |
| `md5_volume_audit.py` | duplicate-audit algorithm 4, re-run to completion |
| `save_split_manifests.py` | regenerates the 80/10/10 manifests offline |

## Findings

**1. Task 6 is handed its label.** Every Task 6 question embeds the past
presence/absence sequence as a bracket, e.g. `lung opacity: [1]`. That bracket
equals the ground-truth past bit on all 16,031 rows — a label, not a hint. It
eliminates 2 of the 4 answer options with no exceptions, leaving **75.4%** of
Task 6 solvable with no pixels read.

**2. Task 4 is 82.6% solvable by always answering the majority class** (two
answers, heavy imbalance). Task 4 is 72% of the held-out set, so it dominates
any aggregate "single-scan" number.

**3. Tasks 5 and 6 are the same 16,031 items.** Same 943 volumes, same 18
findings, **100.00% identical answers** on every matched (volume, finding) pair.
The only difference is that Task 6 reveals the past bit in text and Task 5 does
not. 3D-RAD therefore ships its own matched leaked / un-leaked control pair.

**4. The information structure fixes what is achievable.** The four answers are
a product of two bits — was the finding present before, is it present now:

| input available | best possible accuracy |
| :-- | --: |
| text only (Task 5 phrasing) | 0.5552 |
| text only (Task 6 bracket) | 0.7539 |
| Task 5 + a perfect read of the one provided volume | 0.7392 |
| Task 5 + the prior scan as well | 1.0000 |

Task 5 as the benchmark specifies it supplies a single volume, so the past bit
has no source: **a flawless reader cannot exceed 73.9%**. Supplying the prior
scan lifts the ceiling to 1.0 and makes both bits image-derived.

**5. A trained 3D VLM adds little over the floor.** On the val split, the
checkpoint (epoch 3, val_loss 0.2853) scores 0.7667 single-scan and 0.7733
longitudinal, against text-only floors of 0.6969 and 0.7122 — **+7.0 and
+6.1 points**. Its longitudinal bucket also pools Tasks 5 and 6, i.e. the same
238 clinical questions counted twice, once leaked and once not, which averages
away the effect of interest.

**6. Duplicate-audit algorithm 4 passes.** 3,039 volumes hashed, 3,039 unique
MD5s, zero bitwise-identical clusters, zero cross-split hash leakage. This
supersedes the all-zero block in `duplicate_audit_report.json`, which came from
an interrupted run rather than a measurement.

## Why no radiologist is required

The open risk was whether a de-leaked temporal question is answerable at all.
It does not need expert annotation to settle:

- Task 5 **is** the de-leaked question, published by the benchmark's own
  authors, so answerability is their claim rather than a new assumption.
- Floors and ceilings above are derived from label structure alone.
- Any model scoring above the text-only floor using only images is itself an
  empirical lower bound on answerability.

An expert read would still sharpen the Task 5 ceiling, but it is no longer a
blocker.

## Caveats

- Only 3D-RAD's released image subset is usable: HuggingFace publishes
  `Images/test.zip` alone, so the 136,195 train-split QA rows have no volumes.
  Everything here is the official test set, re-split at patient level.
- 3,039 volumes are on disk but only 2,696 are referenced by any QA record.
- The checkpoint's 0.7667 / 0.7733 are strict exact match over 150 generated
  samples per bucket; the floors here are accuracy over every item in the
  bucket. Per-task model accuracy is needed to close the comparison.
- Bitwise-unique volumes are not necessarily visually distinct: different
  reconstructions of one study differ in bytes. Patient-level grouping, not the
  hash audit, is what controls that.
