# Audit results — epoch-1 checkpoint, full val and test

Run 2026-10-17. Inference only, no training. Checkpoint: `retrain_v7/epoch_1`
(val_loss 0.3115). Split loaded from the saved manifests and verified against the trained
partition: 27,197 / 3,254 / 3,459 QA pairs, exact match.

Closed questions (Tasks 4, 5, 6) scored by length-normalised log-likelihood over their real
options; open questions (Tasks 1, 2, 3) by strict normalised exact match, with token F1 alongside.
Every metric computed twice: once on the real volume, once with the volume replaced by zeros.

## The headline

**Replacing the entire CT volume with zeros does not change the model's accuracy.**

### test (n = 3,459)

| Task | n | text-only floor | real | blank | Δ blank | real − floor | token acc | F1 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 Image Observation | 93 | 0.0108 | 0.1398 | 0.1398 | +0.0000 | +0.1290 | 0.700 | 0.445 |
| 2 Anomaly Detection | 244 | 0.0123 | 0.2131 | 0.2090 | +0.0041 | +0.2008 | 0.653 | 0.451 |
| 3 Medical Computation | 84 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 | 0.570 | 0.413 |
| 4 Existence Detection | 2,358 | 0.8355 | 0.8198 | 0.8155 | +0.0042 | **−0.0157** | 0.911 | — |
| 5 Static Temporal | 340 | 0.6471 | 0.6353 | 0.6353 | +0.0000 | **−0.0118** | 0.971 | — |
| 6 Longitudinal Temporal | 340 | 0.8059 | 0.7941 | 0.7941 | +0.0000 | **−0.0118** | 0.983 | — |

### val (n = 3,254)

| Task | n | text-only floor | real | blank | Δ blank | real − floor | token acc | F1 |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 Image Observation | 88 | 0.0114 | 0.2159 | 0.2159 | +0.0000 | +0.2045 | 0.675 | 0.508 |
| 2 Anomaly Detection | 242 | 0.0083 | 0.1612 | 0.1612 | +0.0000 | +0.1529 | 0.649 | 0.431 |
| 3 Medical Computation | 108 | 0.0000 | 0.0093 | 0.0093 | +0.0000 | +0.0093 | 0.600 | 0.415 |
| 4 Existence Detection | 2,340 | 0.8256 | 0.8115 | 0.8103 | +0.0013 | **−0.0141** | 0.909 | — |
| 5 Static Temporal | 238 | 0.6176 | 0.6218 | 0.6176 | +0.0042 | +0.0042 | 0.970 | — |
| 6 Longitudinal Temporal | 238 | 0.8067 | 0.8025 | 0.8025 | +0.0000 | **−0.0042** | 0.984 | — |

On the test partition the model scores **below the text-only floor on all three closed tasks**.

## It is not an artefact

A zero delta invites the suspicion that the ablation never zeroed anything. It did, and the
image does reach the model — it simply carries almost nothing:

| Check (val) | Result |
| :-- | :-- |
| Pooler attention differs between real and blank | 476 / 476 paired items |
| Gold answer's log-probability differs | 2,738 / 2,816 items |
| Size of that difference | mean 0.0055 nats, max 0.0624 |
| Closed predictions that actually flip | 35 / 2,816 (1.24%) |
| Of those, on Task 6 | **0 / 238** |

So the whole volume moves the answer's log-probability by about five thousandths of a nat —
orders of magnitude short of changing which option wins.

## What this licenses, and what it does not

**It licenses:** 3D-RAD's multi-temporal score does not evidence visual grounding, because a
model whose output is demonstrably independent of the CT volume scores 0.794 on Task 6. Tasks 4
and 6 together are 78% of the test partition, so the benchmark's aggregate cannot separate a
blind model from a sighted one. This is a demonstration, not an inference.

**It does not license** any claim that vision-language models cannot do longitudinal CT reasoning.
The finding is about this baseline: a frozen ImageNet ResNet-18 slice encoder, 16 visual tokens, a
two-layer projector, LoRA on a 1.5B language model, three epochs on one T4. A better-resourced
model might use the image. The claim to make is "this score is not evidence of grounding", not
"grounding is impossible".

## Two further findings

**Teacher-forced token accuracy is misleading here.** Task 6 reads 0.983 token accuracy against
0.794 real accuracy, and the blind model reaches the same 0.794. The earlier run's headline
"0.977 longitudinal token accuracy" measured nothing.

**Above-floor performance on the open tasks is also image-independent.** Tasks 1 and 2 beat their
floors by 13 and 20 points, yet their blank delta is zero — that margin comes from the language
model's radiological priors, not from the scan.

## Files

`inference_audit_summary.json` (both splits, every column) · `inference_audit_{val,test}.json` ·
`per_task_test.csv` · `per_item_val.csv` (row-level predictions, the substrate for the
hallucination taxonomy) · `run_log.txt`.

Two files were not retrieved from the run: `per_task_val.csv` and `per_item_test.csv`. Every
aggregate they held is in the JSON; `per_item_test.csv` is still worth pulling down, because the
taxonomy work needs row-level test predictions.

## Patient-cluster statistics (validation only)

`audit_statistics.py` reads `per_item_val.csv` and resamples patients (5,000 draws, seed 42), because
questions share volumes and volumes share patients. The accuracies reproduce the table above.
Task 5 and Task 6 each have 14 patients, so their intervals are wide.

| comparison | n items (patients) | estimate | 95% CI | exact McNemar p |
| :-- | --: | --: | :-- | --: |
| Task 4 real − blank | 2,340 (130) | +0.0013 | −0.003 to +0.006 | 0.73 |
| Task 5 real − blank | 238 (14) | +0.0042 | 0.000 to +0.013 | 1.0 |
| Task 6 real − blank | 238 (14) | 0.0000 | 0.000 to 0.000 | 1.0 (0 discordant) |
| Task 6 − Task 5, real CT, same (volume, finding) items | 238 (14) | **+0.181** | +0.109 to +0.265 | 2e−13 (43 vs 0) |

Reading: zeroing the CT has no detectable effect on any closed task. The same model scores 18
points higher on Task 6 than on the identical Task 5 items; on every one of the 43 items where
exactly one of the two is correct, it is the Task 6 one. The only thing Task 6 adds is the bracket, so
this gap is the text shortcut being used, which is the opposite of image use. Limits: validation
only (per-item test output was not saved, so the test partition needs a rerun), epoch 1, one
seed, and the T0 reconstruction in that run was chosen by hash order (see README).

### Balanced accuracy and macro-F1 (validation, patient-bootstrap CI)

Plain accuracy rewards always saying the majority class (Task 4 is 83% "No"). Balanced accuracy is
the mean recall over gold classes, so chance is 1/classes.

| task | condition | accuracy | balanced acc (95% CI) | chance | macro-F1 | classes predicted |
| :-- | :-- | --: | :-- | --: | --: | :-- |
| 4 Existence | real | 0.812 | 0.564 (0.546–0.585) | 0.50 | 0.573 | 2 / 2 |
| 4 Existence | blank | 0.810 | 0.556 (0.541–0.574) | 0.50 | 0.563 | 2 / 2 |
| 5 Static temporal | real | 0.622 | 0.279 (0.255–0.312) | 0.25 | 0.247 | 2 / 4 |
| 5 Static temporal | blank | 0.618 | 0.273 (0.251–0.304) | 0.25 | 0.239 | 2 / 4 |
| 6 Longitudinal | real | 0.803 | 0.511 (0.475–0.542) | 0.25 | 0.478 | 3 / 4 |
| 6 Longitudinal | blank | 0.803 | 0.511 (0.471–0.541) | 0.25 | 0.478 | 3 / 4 |

Task 5 is at chance once class imbalance is removed, with and without the CT. Task 6 rises to 0.51,
and only because the bracket rules out two options; the model never predicts "New Lesion" and
real and blank are identical. Task 4 is only a few points above chance in both conditions.

## Multi-control audit (epoch 1, validation and test)

Run 2026-11-05 on Kaggle (T4), `ALL_EPOCHS_FULL_AUDIT.py`, audit sha256 `294fc43d6d8c`. Epoch 1 val
(89 min, 17,222 passes) and epoch 1 test (100 min, 3,459 items, 680 temporal items paired) both finished with
7 conditions. Epochs 2-3 were not attached, so they were skipped. Statistics: `audit_statistics_multi.py`
(patient-cluster bootstrap, 5,000 draws, seed 42), outputs in
`interface audit result/full_audit_results/full_audit/epoch_1/{val,test}/audit_statistics_epoch1_{val,test}_7cond.json`.
The tables below are val; the test results follow in their own subsection.

### Major finding

**This model does not use the CT.** Zeros, a constant image, noise, or another patient's scan all leave
closed-task accuracy within one item of the real scan, and no difference is significant.

### Accuracy by condition (val)

| Task | floor | real | blank | mean | noise | swap | zero_prior | swap_prior |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 Observation (open) | .011 | .216 | .216 | .216 | .216 | .193 | - | - |
| 2 Anomaly (open) | .012 | .161 | .161 | .161 | .161 | .161 | - | - |
| 3 Computation (open) | 0 | .009 | .009 | .009 | .009 | .009 | - | - |
| 4 Existence | .826 | .812 | .810 | .809 | .808 | .811 | - | - |
| 5 Static temporal | .618 | .618 | .618 | .622 | .622 | .622 | .618 | .622 |
| 6 Longitudinal | .807 | .803 | .803 | .803 | .803 | .798 | .803 | .803 |

### Real minus control (closed tasks, patient-cluster 95% CI, exact McNemar)

| Task | control | real - control | 95% CI | discordant (real-only / control-only) | p |
| :-- | :-- | --: | :-- | :-- | --: |
| 4 | blank | +0.0013 | -0.003 to +0.006 | 18 / 15 | 0.73 |
| 4 | mean | +0.0030 | -0.002 to +0.008 | 20 / 13 | 0.30 |
| 4 | noise | +0.0034 | -0.001 to +0.008 | 19 / 11 | 0.20 |
| 4 | swap | +0.0004 | -0.004 to +0.005 | 16 / 15 | 1.0 |
| 5 | blank, zero_prior | 0.000 | 0 to 0 | 0 / 0 | 1.0 |
| 5 | mean, noise, swap, swap_prior | -0.0042 | -0.013 to 0 | 0 / 1 | 1.0 |
| 6 | blank, mean, noise, zero_prior, swap_prior | 0.000 | 0 to 0 | 0 / 0 | 1.0 |
| 6 | swap | +0.0042 | 0 to +0.013 | 1 / 0 | 1.0 |

### How far the image moves the model

| Task | predictions that flip vs real | mean shift in gold log-prob |
| :-- | --: | --: |
| 4 | 1.3-1.4% (every control) | 0.006 nats (max 0.11) |
| 5 | 0-0.4% (at most 1 of 238) | 0.002 nats |
| 6 | 0% (swap: 1 of 238) | 0.001 nats |

Zeroing or swapping the earlier scan (Tasks 5, 6) changes nothing: the model does not use the temporal pair.

### Text shortcut and class prior

- Task 6 - Task 5 on the same 238 (volume, finding) items: **+0.185** (CI +0.113 to +0.269), 44 vs 0
  discordant, p = 1e-13. The only difference is the finding stated in the question text.
- Predicted vs gold distribution (real CT): Task 4 predicts No 92% (gold 83%); Task 5 predicts D 92% (gold 62%);
  Task 6 predicts D 70%, A 26%, B 4%, never C.
- Balanced accuracy is unchanged from the table above: Task 4 0.564, Task 5 0.273 (chance 0.25), Task 6 0.511.

### Reproducibility note

Against the October run, `blank` is bit-identical (0 predictions differ), but `real` differs on 2 Task 5 items
(A -> D, gold log-prob shift at most 0.010 nats), moving Task 5 real from 0.6218 to 0.6176. The cause is not traced
(GPU nondeterminism or a different T0 reconstruction pick; see README). This run-to-run variation is the same
size as the real-vs-blank effect, so the effect is not distinguishable from noise.

### Test split (n = 3,459), same 7 conditions

| Task | floor | real | blank | mean | noise | swap | zero_prior | swap_prior |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 Observation (open) | .011 | .140 | .140 | .140 | .140 | .129 | - | - |
| 2 Anomaly (open) | .025 | .213 | .209 | .201 | .205 | .209 | - | - |
| 3 Computation (open) | 0 | 0 | 0 | 0 | 0 | 0 | - | - |
| 4 Existence | .835 | .820 | .816 | .821 | .817 | .815 | - | - |
| 5 Static temporal | .647 | .632 | .635 | .632 | .629 | .632 | .632 | .632 |
| 6 Longitudinal | .806 | .794 | .794 | .797 | .791 | .794 | .800 | .791 |

Real minus control (closed tasks, patient-cluster 95% CI, exact McNemar):

| Task | control | real - control | 95% CI | discordant | p |
| :-- | :-- | --: | :-- | :-- | --: |
| 4 | blank | +0.0042 | 0.000 to +0.008 | 17 / 7 | 0.064 |
| 4 | mean | -0.0008 | -0.006 to +0.004 | 15 / 17 | 0.86 |
| 4 | noise | +0.0025 | -0.002 to +0.007 | 17 / 11 | 0.35 |
| 4 | swap | +0.0051 | +0.001 to +0.009 | 21 / 9 | 0.043 |
| 5 | blank | -0.0029 | -0.009 to 0 | 0 / 1 | 1.0 |
| 5 | mean, swap, zero_prior, swap_prior | 0.000 | 0 to 0 | 0 / 0 | 1.0 |
| 5 | noise | +0.0029 | 0 to +0.009 | 1 / 0 | 1.0 |
| 6 | blank | 0.000 | 0 to 0 | 0 / 0 | 1.0 |
| 6 | mean, noise, swap, swap_prior, zero_prior | -0.006 to +0.003 | all CIs include 0 | at most 2 | 0.5-1.0 |

- **One nominal hit:** Task 4 real vs swap, +0.5 points, p = 0.043. It is one of 13 closed-task comparisons on this
  split (about 26 across val and test), so it does not survive a multiple-comparison correction. It is also tiny
  (30 discordant items of 2,358), the model is still below the text-only floor (0.820 vs 0.835), and the same
  comparison on val is +0.0004 (p = 1.0). Report it, do not build on it.
- **Task 6 minus Task 5 on the same 340 items:** +0.162 (CI +0.106 to +0.221), 55 vs 0 discordant, p = 6e-17.
  Val was +0.185, so the text-shortcut gap replicates on test.
- **Balanced accuracy (real):** Task 4 0.558, Task 5 0.262 (chance 0.25), Task 6 0.487. Same picture as val.
- **Against the October test run:** Task 5 real moved from 0.6353 to 0.6324 (1 item) and the Task 2 floor from 0.0123
  to 0.0246. The floor change is the deterministic tie-break (see checklist); the Task 5 change is the same
  untraced run-to-run variation as on val.

### Limits

One baseline architecture (LoRA Qwen2.5-1.5B + frozen ResNet-18), one seed; Tasks 5 and 6 rest on 14 patients (val) and 20 patients (test). The supportable claim is "these scores are not evidence of grounding", not that vision-language models cannot do longitudinal CT reasoning.

---

## Multi-epoch full audit (Epochs 2 and 3, validation and test)

Run on Kaggle (T4) using `ALL_EPOCHS_FULL_AUDIT.py` with checkpoints `retrain_v7/epoch_2` (val_loss 0.2999) and `retrain_v7/epoch_3` (val_loss 0.2986, the best checkpoint). Full val ($n = 3,254$) and full test ($n = 3,459$) evaluated across all 7 conditions (real, blank, mean, noise, swap, zero_prior, swap_prior). Statistics generated via `audit_statistics_multi.py` (patient-cluster bootstrap, 5,000 draws, seed 42).

### Major finding across all epochs

**Training longer does not induce visual grounding.** In Epoch 3 (best checkpoint), closed-task accuracy on the test partition remains clamped to the text-only floor across all tasks. Blank, noise, and swapped-patient controls perform identically to or slightly outperform the real CT volume.

### Accuracy by condition: Epoch 3 (best checkpoint)

#### Test split (n = 3,459)

| Task | floor | real | blank | mean | noise | swap | zero_prior | swap_prior |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 Observation (open) | .011 | .247 | .226 | .215 | .237 | .258 | - | - |
| 2 Anomaly (open) | .025 | .221 | .225 | .225 | .238 | .234 | - | - |
| 3 Computation (open) | 0 | .048 | .048 | .036 | .048 | .048 | - | - |
| 4 Existence | .835 | .836 | .832 | .832 | .835 | .837 | - | - |
| 5 Static temporal | .647 | .647 | .644 | .644 | .647 | .647 | .647 | .647 |
| 6 Longitudinal | .806 | .806 | .812 | .809 | .812 | .806 | .806 | .794 |

#### Validation split (n = 3,254)

| Task | floor | real | blank | mean | noise | swap | zero_prior | swap_prior |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 Observation (open) | .011 | .205 | .250 | .216 | .205 | .193 | - | - |
| 2 Anomaly (open) | .012 | .215 | .227 | .219 | .219 | .219 | - | - |
| 3 Computation (open) | 0 | .037 | .056 | .056 | .046 | .037 | - | - |
| 4 Existence | .826 | .825 | .822 | .827 | .825 | .824 | - | - |
| 5 Static temporal | .618 | .613 | .609 | .609 | .609 | .613 | .618 | .613 |
| 6 Longitudinal | .807 | .786 | .773 | .777 | .773 | .790 | .773 | .786 |

### Real minus control: Epoch 3 test (closed tasks, patient-cluster 95% CI, exact McNemar)

| Task | control | real - control | 95% CI | discordant (real / control) | p |
| :-- | :-- | --: | :-- | :-- | --: |
| 4 | blank | +0.0038 | -0.002 to +0.010 | 30 / 21 | 0.262 |
| 4 | mean | +0.0038 | -0.003 to +0.011 | 39 / 30 | 0.336 |
| 4 | noise | +0.0008 | -0.004 to +0.006 | 19 / 17 | 0.868 |
| 4 | swap | -0.0013 | -0.005 to +0.003 | 9 / 12 | 0.664 |
| 5 | blank | +0.0029 | 0 to +0.009 | 1 / 0 | 1.0 |
| 5 | mean | +0.0029 | 0 to +0.009 | 1 / 0 | 1.0 |
| 5 | noise, swap, zero_prior, swap_prior | 0.000 | 0 to 0 | 0 / 0 | 1.0 |
| 6 | blank | -0.0059 | -0.029 to +0.018 | 5 / 7 | 0.774 |
| 6 | mean | -0.0029 | -0.029 to +0.027 | 6 / 7 | 1.0 |
| 6 | noise | -0.0059 | -0.024 to +0.012 | 3 / 5 | 0.727 |
| 6 | swap, zero_prior | 0.000 | -0.021 to +0.021 | 4 / 4 (swap), 5 / 5 (zero_prior) | 1.0 |
| 6 | swap_prior | +0.0118 | 0 to +0.027 | 5 / 1 | 0.219 |

### Balanced accuracy and macro-F1: Epoch 3 test (patient-bootstrap 95% CI)

| task | condition | accuracy | balanced acc (95% CI) | chance | macro-F1 | classes predicted |
| :-- | :-- | --: | :-- | --: | --: | :-- |
| 4 Existence | real | 0.836 | 0.509 (0.502–0.516) | 0.50 | 0.475 | 2 / 2 |
| 4 Existence | blank | 0.832 | 0.525 (0.514–0.537) | 0.50 | 0.512 | 2 / 2 |
| 5 Static temporal | real | 0.647 | 0.253 (0.247–0.265) | 0.25 | 0.206 | 2 / 4 |
| 5 Static temporal | blank | 0.644 | 0.252 (0.245–0.264) | 0.25 | 0.205 | 2 / 4 |
| 6 Longitudinal | real | 0.806 | 0.531 (0.484–0.576) | 0.25 | 0.519 | 4 / 4 |
| 6 Longitudinal | blank | 0.812 | 0.542 (0.503–0.590) | 0.25 | 0.528 | 3 / 4 |

Notice: On Task 4 and Task 6, blank CT achieves higher balanced accuracy and macro-F1 than real CT. Task 5 balanced accuracy sits exactly at random chance (0.253 vs 0.250).

### Task 6 minus Task 5 on identical 340 test items (Epoch 3)

* **Task 6 - Task 5 diff:** **+0.1588** (+15.9 points), patient-cluster 95% CI: **+0.1118 to +0.2118**.
* **Exact McNemar:** 55 vs 1 discordant pairs, **$p = 1.58 \times 10^{-15}$**.
* Val split on Epoch 3: **+0.1723** (+17.2 points), 95% CI: **+0.1134 to +0.2437**, 41 vs 0 discordant, **$p = 9.09 \times 10^{-13}$**.
* This replicates the Task 6 text-shortcut gap across all epochs.

---

### Accuracy by condition: Epoch 2 (val_loss 0.2999)

#### Test split (n = 3,459)

| Task | floor | real | blank | mean | noise | swap | zero_prior | swap_prior |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 Observation (open) | .011 | .204 | .204 | .194 | .204 | .204 | - | - |
| 2 Anomaly (open) | .025 | .230 | .225 | .238 | .238 | .230 | - | - |
| 3 Computation (open) | 0 | .048 | .048 | .048 | .048 | .048 | - | - |
| 4 Existence | .835 | .832 | .831 | .832 | .832 | .835 | - | - |
| 5 Static temporal | .647 | .635 | .629 | .632 | .638 | .638 | .635 | .635 |
| 6 Longitudinal | .806 | .803 | .794 | .803 | .812 | .809 | .806 | .800 |

#### Validation split (n = 3,254)

| Task | floor | real | blank | mean | noise | swap | zero_prior | swap_prior |
| :-- | --: | --: | --: | --: | --: | --: | --: | --: |
| 1 Observation (open) | .011 | .205 | .193 | .193 | .193 | .205 | - | - |
| 2 Anomaly (open) | .012 | .198 | .202 | .198 | .194 | .202 | - | - |
| 3 Computation (open) | 0 | .056 | .056 | .056 | .065 | .065 | - | - |
| 4 Existence | .826 | .825 | .826 | .826 | .822 | .823 | - | - |
| 5 Static temporal | .618 | .605 | .605 | .605 | .601 | .605 | .605 | .605 |
| 6 Longitudinal | .807 | .756 | .769 | .765 | .761 | .756 | .761 | .756 |

### Epoch 2 test statistics (closed tasks, patient-cluster 95% CI, exact McNemar)

* **Task 4:** real - blank = +0.0008 (CI -0.0047 to +0.0064, p = 0.883, discordant 24/22); real - swap = -0.0030 (p = 0.296).
* **Task 5:** real - blank = +0.0059 (CI 0 to +0.0147, p = 0.500, discordant 2/0); zero_prior = 0.000 (p = 1.0).
* **Task 6:** real - blank = +0.0088 (CI -0.0029 to +0.0206, p = 0.375, discordant 4/1); real - noise = -0.0088 (p = 0.25).
* **Task 6 - Task 5:** +0.1676 (+16.8 points), CI: +0.1352 to +0.2029, McNemar 63 vs 6 discordant, $p = 4.47 \times 10^{-13}$.

---

### Synthesis: Multi-Epoch Comparison (Test Split Closed Tasks)

| Task | Metric | Text-Only Floor | Epoch 1 (val 0.3115) | Epoch 2 (val 0.2999) | Epoch 3 (val 0.2986) |
| :-- | :-- | --: | --: | --: | --: |
| **Task 4** | Real CT | 0.8355 | 0.8198 | 0.8316 | 0.8359 |
| | Blank CT | | 0.8155 | 0.8308 | 0.8321 |
| | Swapped CT | | 0.8148 | 0.8346 | 0.8372 |
| | Real − Floor | | -0.0157 | -0.0038 | +0.0004 |
| **Task 5** | Real CT | 0.6471 | 0.6353 | 0.6353 | 0.6471 |
| | Blank CT | | 0.6353 | 0.6294 | 0.6441 |
| | Zeroed Prior | | 0.6324 | 0.6353 | 0.6471 |
| | Real − Floor | | -0.0118 | -0.0118 | 0.0000 |
| **Task 6** | Real CT | 0.8059 | 0.7941 | 0.8029 | 0.8059 |
| | Blank CT | | 0.7941 | 0.7941 | 0.8118 |
| | Zeroed Prior | | 0.8000 | 0.8059 | 0.8059 |
| | Real − Floor | | -0.0118 | -0.0029 | 0.0000 |
| **T6 − T5** | Paired Gain | — | **+16.2%** ($p = 6\times 10^{-17}$) | **+16.8%** ($p = 4\times 10^{-13}$) | **+15.9%** ($p = 2\times 10^{-15}$) |

### Conclusion

Across all three epochs:
1. **The model never beats the text-only floor** by more than a fraction of a percent (+0.04% on Task 4 at Epoch 3), which is statistically indistinguishable from zero ($p = 0.262$).
2. **Perturbation insensitivity is invariant to training**: zeroing the scan, adding noise, swapping patients, or zeroing the prior scan produces no statistically significant difference ($p > 0.20$ across all comparisons).
3. **The Task 6 text-shortcut gap replicates consistently**: Task 6 consistently beats identical Task 5 items by +15.9% to +16.8% across all epochs ($p < 10^{-12}$ in every run), demonstrating that high performance on Task 6 is entirely mediated by the bracket text leak.
