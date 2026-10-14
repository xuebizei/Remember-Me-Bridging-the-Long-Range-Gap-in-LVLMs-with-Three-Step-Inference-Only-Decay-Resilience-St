# Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies

Official paper of "Remember Me: Bridging the Long-Range Gap in LVLMs with Three-Step Inference-Only Decay Resilience Strategies": [arXiv](https://arxiv.org/abs/2511.09868) · [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/42409)

## Run

Download LLaVA-1.5 weights and the COCO val2014 images used by POPE, then put the images at `playground/data/eval/pope/val2014`.

```bash
bash scripts/v1_5/eval/pope.sh
```

Core code: `llava/eval/model_vqa_loader.py`, function `custom_attn_forward`. It is attached to layer 31 before generation.
