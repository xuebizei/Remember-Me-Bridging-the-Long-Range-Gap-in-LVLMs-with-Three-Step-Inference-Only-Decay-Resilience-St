import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from transformers.cache_utils import Cache, DynamicCache
from PIL import Image
import math
import random
import matplotlib.pyplot as plt
import inspect
import types
from typing import List, Optional, Tuple, Union

# random.seed(42)

#print('tokenizer_image_token', inspect.getfile(tokenizer_image_token)) #tokenizer_image_token ~/conference_new/LLaVA/llava/mm_utils.py

import torch.nn as nn
import torch.nn.functional as F
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def compute_distance(q_len: int, k_len: int, device, dtype):
    """
    构造 [q_len, k_len] 的非负距离矩阵 distance[i,j] = max(0, j - (k_len-q_len+i))
    """
    idx_i = torch.arange(k_len - q_len, k_len, device=device).unsqueeze(1)  # [q_len,1]
    idx_j = torch.arange(k_len,           device=device).unsqueeze(0)      # [1,k_len]
    return (idx_j - idx_i).clamp(min=0).to(dtype)

def custom_attn_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
        )

    bsz, q_len, _ = hidden_states.size()

    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)

    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask



    B, H, q_len, k_len = attn_weights.shape

    # 1) 距离矩阵
    distance = compute_distance(q_len, k_len,
                                    device=attn_weights.device,
                                    dtype=attn_weights.dtype)  # [q_len, k_len]
    dist = distance.unsqueeze(0).unsqueeze(0)                # [1,1,q_len,k_len]
    #print('dist', dist)
    # 2) 语义相似度 sem_sim ∈[-1,1] → sem_pos ∈[0,1]
    q_norm  = F.normalize(query_states, dim=-1)              # [B,H,q_len,D]
    k_norm  = F.normalize(key_states,   dim=-1)              # [B,H,k_len,D]
    sem_sim = torch.matmul(q_norm, k_norm.transpose(-1,-2))  # [B,H,q_len,k_len]
    sem_pos = (sem_sim + 1.0) / 2.0                          # [B,H,q_len,k_len]

    # 3) 映射到伸缩因子 scale ∈[0.5,1.5]
    min_s, max_s = 0.5, 1.5
    scale = min_s + (max_s - min_s) * sem_pos                # [B,H,q_len,k_len]

    # 4) 反推 σ₀、σ₁ 保证最远处保留 min_w
    d_max = k_len - 1
    min_w = 0.3
    sigma0 = d_max / math.sqrt(-2.0 * math.log(min_w))       # Gaussian σ₀
    alpha  = 0.5                                            # RQ 形状
    sigma1 = d_max / math.sqrt(
        2 * alpha * (min_w**(-1.0/alpha) - 1.0)
    )                                                       # RQ σ₁

    # 5) 语义驱动核计算（σ_scaled = σ * scale）
    sigma0_s = sigma0 * scale                                # [B,H,q_len,k_len]
    sigma1_s = sigma1 * scale

    w_gauss = torch.exp(-0.5 * (dist / sigma0_s)**2)         # [B,H,q_len,k_len]
    w_rq    = (1 + (dist**2) / (2 * alpha * sigma1_s**2)).pow(-alpha)
    oral_attn_weights = attn_weights
    # 6) 融合三路权重：高斯、RQ、先验
    #    先验也用残差形式：越高的先验，额外拉高最终权重
    #print('w_gauss', w_gauss)
    attn_weights = (
        attn_weights * w_gauss
        # + attn_weights * w_rq
        # + attn_weights * sem_pos
        + attn_weights
    )
    # print('=========================================')
    # print(oral_attn_weights.mean(1).squeeze(0))
    # print()
    # print(attn_weights.mean(1).squeeze(0))
    # print('=========================================')
    
    if attn_weights.shape[-1] == attn_weights.shape[-2]:
        
        
        B, H, q_len, k_len = attn_weights.shape

        # 1) 距离矩阵
        distance = compute_distance(q_len, k_len,
                                        device=attn_weights.device,
                                        dtype=attn_weights.dtype)  # [q_len, k_len]
        dist = distance.unsqueeze(0).unsqueeze(0)                # [1,1,q_len,k_len]
        #print('dist', dist)
        # 2) 语义相似度 sem_sim ∈[-1,1] → sem_pos ∈[0,1]
        q_norm  = F.normalize(query_states, dim=-1)              # [B,H,q_len,D]
        k_norm  = F.normalize(key_states,   dim=-1)              # [B,H,k_len,D]
        sem_sim = torch.matmul(q_norm, k_norm.transpose(-1,-2))  # [B,H,q_len,k_len]
        sem_pos = (sem_sim + 1.0) / 2.0                          # [B,H,q_len,k_len]

        # 3) 映射到伸缩因子 scale ∈[0.5,1.5]
        min_s, max_s = 0.5, 1.5
        scale = min_s + (max_s - min_s) * sem_pos                # [B,H,q_len,k_len]

        # 4) 反推 σ₀、σ₁ 保证最远处保留 min_w
        d_max = k_len - 1
        min_w = 0.3
        sigma0 = d_max / math.sqrt(-2.0 * math.log(min_w))       # Gaussian σ₀
        alpha  = 0.5                                            # RQ 形状
        sigma1 = d_max / math.sqrt(
            2 * alpha * (min_w**(-1.0/alpha) - 1.0)
        )                                                       # RQ σ₁

        # 5) 语义驱动核计算（σ_scaled = σ * scale）
        sigma0_s = sigma0 * scale                                # [B,H,q_len,k_len]
        sigma1_s = sigma1 * scale

        w_gauss = torch.exp(-0.5 * (dist / sigma0_s)**2)         # [B,H,q_len,k_len]
        w_rq    = (1 + (dist**2) / (2 * alpha * sigma1_s**2)).pow(-alpha)
        oral_attn_weights = attn_weights
        # 6) 融合三路权重：高斯、RQ、先验
        #    先验也用残差形式：越高的先验，额外拉高最终权重
        #print('w_gauss', w_gauss)
        print('oral_attn_weights', oral_attn_weights.mean(1).squeeze(0))
        attn_weights = (
            #attn_weights * w_gauss
            attn_weights * w_rq
            # + attn_weights * sem_pos
            + attn_weights
        )
        
        
        
        
        
        
        
        #print('dist', dist)
        print('attn_weights', attn_weights.mean(1).squeeze(0))
        # img_token_len = 576
        # img_token_start = 35
        # txt_token_start = img_token_start + img_token_len
        # prompt_to_image_attn = attn_weights[:, :, txt_token_start:, img_token_start:img_token_start + img_token_len].mean(dim=1).mean(dim=-2).squeeze(0)
        
        # total = prompt_to_image_attn.sum()  # sum over prompt & image
        # normalized = prompt_to_image_attn / (total)
        # print('normalized', normalized.shape)
        
        # new_dir = '~/conference_new/LLaVA/llava/eval/gaussian_png'
        # plt.imshow(normalized.reshape(24,24).to(torch.float32).detach().cpu().numpy(), cmap='viridis')
        # plt.colorbar()
        # plt.savefig(new_dir, dpi=600, bbox_inches='tight')  
    
    # upcast attention to fp32
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2).contiguous()

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    print('model_path', model_path) #~/conference_new/LLaVA/liuhaotian
    model_name = get_model_name_from_path(model_path)
    print('model_name', model_name) #liuhaotian
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    for i, line in enumerate(tqdm(questions[20:])): #这里原来有tqdm
        idx = line["id"]
        question = line['conversations'][0] #这个0就是选了question
        qs = question['value'].replace('<image>', '').strip()
        cur_prompt = qs  #question

        if 'image' in line:
            
            image_file = line["image"]
            image = Image.open(os.path.join(args.image_folder, image_file))
            image_tensor = process_images([image], image_processor, model.config)[0]
            images = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            # print('image in line', image_sizes)
            #print('image_sizes', image_sizes)
            if getattr(model.config, 'mm_use_im_start_end', False):
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
                
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + qs
            cur_prompt = '<image>' + '\n' + cur_prompt
        # else:
        #     images = None
        #     image_sizes = None

            if args.single_pred_prompt:
                #print('use single_pred_prompt')
                qs = qs + '\n' + "Answer with the option's letter from the given choices directly."
                cur_prompt = cur_prompt + '\n' + "Answer with the option's letter from the given choices directly."

            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            #print('inspect', inspect.getsource(model.forward))
            #print('input_ids', input_ids, input_ids.shape)
            
            layer31 = model.model.layers[31]
            orig_attn31 = layer31.self_attn
            layer31.self_attn.forward = types.MethodType(custom_attn_forward, orig_attn31)
            
            
            
            # print('self,',inspect.getsource(layer31.forward))
            # print('self,',inspect.getsource(model.model.forward))
            #print('self,',inspect.getfile(model.model.forward))
            #print('image in line', image_sizes)
            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=images,
                    image_sizes=image_sizes,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    max_new_tokens=1024,
                    use_cache=True,
                    output_attentions=True,
                    return_dict = True
                
                )
                #print('input_ids', input_ids)
            #print('getfile', inspect.getfile(model.__init__)) #getfile ~/conference_new/LLaVA/llava/model/language_model/llava_llama.py
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            # print('outputs', outputs)
            # print('====')
            decoder_attn = model.text_image_if
            
            img_token_pos = (input_ids[0] == -200).nonzero(as_tuple=True)
            #print('img_token_pos', img_token_pos) #35
            IMG_TOKEN_LEN = 576
            #print('decoder_attn', decoder_attn.shape)
            #text_image_if = decoder_attn[0, img_token_pos + IMG_TOKEN_LEN: decoder_attn.shape[1] - 1, img_token_pos: img_token_pos + IMG_TOKEN_LEN].mean(0)
           # print('text_image_if', text_image_if.shape)
            
            
            ans_id = shortuuid.uuid()
            ans_file.write(json.dumps({"question_id": idx,
                                    "prompt": cur_prompt,
                                    "text": outputs,
                                    "answer_id": ans_id,
                                    "model_id": model_name,
                                    "metadata": {}}) + "\n")
            ans_file.flush()
            break
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.json")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v0")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--answer-prompter", action="store_true")
    parser.add_argument("--single-pred-prompt", action="store_true")
    args = parser.parse_args()

    eval_model(args)
