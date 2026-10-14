# The Code of "Remember-Me-Bridging-the-Long-Range-Gap-in-LVLMs-with-Three-Step-Inference-Only-Decay-Resilience-strategies"

Evaluation code for running LLaVA-1.5 on [POPE](https://github.com/AoiDragon/POPE). Before softmax, layer 31 reweights attention scores with a Gaussian kernel and a rational-quadratic kernel. Query-key cosine similarity scales each kernel bandwidth between 0.5x and 1.5x.

This repository is derived from [LLaVA](https://github.com/haotian-liu/LLaVA) and keeps its Apache-2.0 license.

## Run

Download LLaVA-1.5 weights and the COCO val2014 images used by POPE, then put the images at `playground/data/eval/pope/val2014`.

```bash
bash scripts/v1_5/eval/pope.sh
```

The kernel modification is in `llava/eval/model_vqa_loader.py` (`custom_attn_forward`). It is attached to `model.model.layers[31].self_attn` before generation. Scores are computed with `eval_pope.py` against `playground/data/eval/pope/coco`.

Model weights and COCO images are not included.
