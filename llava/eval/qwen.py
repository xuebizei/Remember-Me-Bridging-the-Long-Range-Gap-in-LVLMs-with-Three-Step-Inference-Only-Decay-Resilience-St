from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from analysis_utils import distance_wise_attention_analysis, plot_attention_by_distance
import torch, json

# Load model and processor
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "~/conference_new/LLaVA/qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
model.generation_config.output_attentions = True
model.generation_config.return_dict_in_generate = True
processor = AutoProcessor.from_pretrained("~/conference_new/LLaVA/qwen/Qwen2.5-VL-7B-Instruct")

# Load ScienceQA samples
with open('data/scienceqa_test_img.jsonl') as f:
    samples = [json.loads(l.strip()) for l in f][:10]  # sample few

attn_dist_list = []

for item in samples:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": item["image"]},
            {"type": "text", "text": item["question"]}
        ]
    }]
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True
    ).to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            output_attentions=True,
            return_dict_in_generate=True
        )

    # decoder_attentions[-1][0] 是最后一层第一头
    attn = outputs.decoder_attentions[-1][0]  # [B,H,Q,K]
    avg_by_dist = distance_wise_attention_analysis(attn)
    attn_dist_list.append(avg_by_dist)

# 聚合多个样本
import numpy as np
min_len = min(len(d) for d in attn_dist_list)
all_avg = np.mean([a[:min_len] for a in attn_dist_list], axis=0)

# 绘图
plot_attention_by_distance(
    [all_avg],
    ["Qwen2.5-VL Baseline"]
)
