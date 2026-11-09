# Reverse-complement augmentation for DNABERT-2 on GUE histone tasks

Small, honest study of whether reverse-complement (RC) data augmentation improves fine-tuned DNABERT-2 (117M) on GUE epigenetic-mark tasks.

## Method
- Model: DNABERT-2-117M, fine-tuned with the official finetune/train.py (patched for newer library versions).
- Baseline: original training set. RC arm: every training sequence plus its reverse complement (2x data).
- Both arms use the same 4491 optimizer steps, lr 3e-5, batch size 8, fp16, max length 128.
- Validation and test sets are unchanged. Metric: Matthews correlation (MCC) on the test set.
- Welch t-test across seeds.

## Results (test MCC, mean ± std across seeds)
| Task | n baseline | Baseline MCC | n RC | RC MCC | Welch p |
|---|---|---|---|---|---|
| H3 | 5 | 0.7408 ± 0.0198 | 5 | 0.7763 ± 0.0083 | 0.012 |
| H3K4me3 | 3 | 0.2137 ± 0.1857 | 3 | 0.2734 ± 0.0442 | 0.637 |
| H4 | 3 | 0.8136 ± 0.0160 | 3 | 0.8062 ± 0.0095 | 0.533 |

## Findings
- H3: RC augmentation improved MCC, consistently across seeds.
- H4: no measurable difference.
- H3K4me3: inconclusive. Training was unstable under the fixed step budget (one baseline run collapsed to MCC 0).
- The effect appears task-dependent. A possible explanation is that RC helps more when the baseline is weaker, but this was not tested.

## Limitations
- Few seeds (3 to 5); the first H3 runs were not seed-matched.
- Only 3 of 10 GUE tasks tested; no multiple-comparison correction.
- The fixed step budget under-trains larger tasks.
- Newer library versions, fp16 and PyTorch attention (no Triton) were used, so baselines are my own reproduction and may differ from the paper.

Per-run results are in results.csv.

## Reference
Zhou et al., DNABERT-2 (ICLR 2024). Code: https://github.com/MAGICS-LAB/DNABERT_2