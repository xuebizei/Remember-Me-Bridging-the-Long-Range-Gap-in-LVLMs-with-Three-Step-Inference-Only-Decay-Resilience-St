# LLaVA POPE with semantic kernels

Evaluation code for running LLaVA-1.5 on [POPE](https://github.com/AoiDragon/POPE) with a modified attention forward on layer 31.

Before softmax, the attention scores are reweighted by a Gaussian kernel and a rational-quadratic (heavy-tailed) kernel. Query-key cosine similarity scales each kernel bandwidth between 0.5x and 1.5x. The direct semantic-similarity term is not added as a third weight.

This repository is derived from [LLaVA](https://github.com/haotian-liu/LLaVA) and keeps its Apache-2.0 license.

## Run

Download LLaVA-1.5 weights and the COCO val2014 images used by POPE, then put the images at `playground/data/eval/pope/val2014`.

```bash
bash scripts/v1_5/eval/pope.sh
```

The kernel modification is in `llava/eval/model_vqa_loader.py` (`custom_attn_forward`). It is attached to `model.model.layers[31].self_attn` before generation. Scores are computed with `eval_pope.py` against `playground/data/eval/pope/coco`.

Model weights and COCO images are not included.
