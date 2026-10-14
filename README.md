## Run

Download LLaVA-1.5 weights and the COCO val2014 images used by POPE, then put the images at `playground/data/eval/pope/val2014`.

```bash
bash scripts/v1_5/eval/pope.sh
```

Core code: `llava/eval/model_vqa_loader.py`, function `custom_attn_forward`. It is attached to layer 31 before generation.
