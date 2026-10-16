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
