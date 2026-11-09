# Reverse-complement augmentation for DNABERT-2 on GUE histone tasks

**Summary.** I tested whether adding the reverse complement of every training sequence improves fine-tuned DNABERT-2 (117M parameters) on GUE epigenetic-mark tasks. It helped clearly on H3 (test MCC 0.741 to 0.776, Welch p = 0.012, 5 seeds), made no measurable difference on H4, and was inconclusive on H3K4me3. The effect appears task-dependent.

*Status: personal research project, not peer reviewed.*

## Results (test MCC, mean ± std across seeds)

| Task | n baseline | Baseline MCC | n RC | RC MCC | Welch p |
|---|---|---|---|---|---|
| H3 | 5 | 0.7408 ± 0.0198 | 5 | 0.7763 ± 0.0083 | 0.012 |
| H3K4me3 | 3 | 0.2137 ± 0.1857 | 3 | 0.2734 ± 0.0442 | 0.637 |
| H4 | 3 | 0.8136 ± 0.0160 | 3 | 0.8062 ± 0.0095 | 0.533 |

Per-run numbers are in `results.csv`.

## Findings
- **H3:** RC augmentation improved MCC; all 5 RC runs scored above 0.76, and the baseline varied more between seeds. For three tasks, a Bonferroni threshold would be about 0.017, so p = 0.012 passes it, but with 5 seeds I treat it as suggestive.
- **H4:** no measurable difference (difference of -0.007 MCC, p = 0.53).
- **H3K4me3:** inconclusive. Training was unstable under my fixed step budget, and one baseline run appears to have collapsed (MCC 0.000), so this task does not support or contradict the other results.
- A possible explanation is that RC helps more when the baseline is weaker (H3 baseline 0.74 vs H4 baseline 0.81). I did not test this.

## Method
- Model: DNABERT-2-117M, fine-tuned with the authors' `finetune/train.py`, patched to load the model config correctly with newer library versions.
- Baseline arm: the original training set. RC arm: every training sequence plus its reverse complement (2x data).
- Both arms use the same 4491 optimizer steps (so the RC arm sees about half as many passes over its larger set), learning rate 3e-5, batch size 8, fp16, max length 128.
- Validation and test sets are unchanged. Metric: Matthews correlation coefficient (MCC) on the test set. The script restores the best checkpoint according to the Trainer's validation criterion; I did not change this.
- Statistics: Welch t-test across seeds, per task.

## Limitations
- Few seeds (3 to 5). The two initial H3 runs used the script's default seed and evaluated every 200 steps; the other runs used seeds 1 to 4 and evaluated every 1000 steps.
- Only 3 of the 10 GUE histone tasks were tested.
- The fixed step budget under-trains larger tasks such as H3K4me3.
- I used newer library versions, fp16 and PyTorch attention (no Triton), so my baselines are my own reproduction. I have not compared them to the published GUE numbers.

## How to reproduce
1. Use Google Colab with a T4 GPU and get the GUE dataset from the DNABERT-2 authors (not included here).
2. Clone https://github.com/MAGICS-LAB/DNABERT_2, install `transformers==4.38.2`, `peft==0.10.0`, `einops`, `scikit-learn`, `accelerate==0.27.2`, and uninstall `triton`.
3. Patch `train.py` to build the model with `BertConfig.from_pretrained(...)` (see the notebook).
4. For the RC arm, reverse-complement each training sequence and append it to the training file.
5. Train with `--max_steps 4491 --learning_rate 3e-5 --per_device_train_batch_size 8 --model_max_length 128 --fp16 --seed N`.

## Next steps
- More seeds and more GUE tasks, with a fixed number of epochs per task instead of a fixed step count.
- Test-time averaging over forward and reverse-complement predictions.

## Reference
Zhou et al., DNABERT-2 (ICLR 2024). Code: https://github.com/MAGICS-LAB/DNABERT_2
