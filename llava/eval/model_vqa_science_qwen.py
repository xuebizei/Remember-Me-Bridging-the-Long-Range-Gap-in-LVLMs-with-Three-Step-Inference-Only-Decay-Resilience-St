import argparse
import json
import os
import torch
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import shortuuid

from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")


QUESTION_TEMPLATE = "{Question} Output the thinking process in <think> </think> and final answer (number) in <answer> </answer> tags."




def visualize_attention(
    multihead_attention,
    output_path="atten_map_1.png",
    title="Layer",
    total_input=108,
    total_gen=0,
    stride=1
):
    """
    attention map 可视化，横纵坐标显示SYS/IMG/USER/GEN的区间。
    """
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float() # (n_tokens, n_tokens)
    total_token = averaged_attention.shape[0]
    total_gen = total_token - total_input
    full_labels = make_full_labels(total_input=total_input, total_gen=total_gen)

    # 池化
    if stride > 1:
        averaged_attention = torch.nn.functional.avg_pool2d(
            averaged_attention.unsqueeze(0).unsqueeze(0),
            stride, stride=stride).squeeze(0).squeeze(0)
        pooled_labels = pool_labels(full_labels, stride=stride)
    else:
        pooled_labels = full_labels

    # 绘图
    plt.figure(figsize=(8, 8), dpi=400)
    cmap = plt.colormaps['viridis']
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())
    ax = sns.heatmap(
        averaged_attention,
        cmap=cmap,
        norm=log_norm,
        square=True,
        # cbar_kws={'label': 'Attention score'},
    )
    ax.set_xticks(range(len(pooled_labels)))
    ax.set_yticks(range(len(pooled_labels)))
    ax.set_xticklabels(pooled_labels, fontsize=5, rotation=90)
    ax.set_yticklabels(pooled_labels, fontsize=5, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return None, averaged_attention

def print_all_attention_shapes(attentions):
    """
    attentions: list of length T (生成步数)
                each element is a list/tuple of length L (层数),
                each inner tensor shape = (B, H, S, S)
    """
    for step_idx, step_attns in enumerate(attentions):
        for layer_idx, attn_tensor in enumerate(step_attns):
            print(f"Step {step_idx:3d} | Layer {layer_idx:3d} | Shape {tuple(attn_tensor.shape)}")


def split_list(lst, n):
    chunk_size = (len(lst) + n - 1) // n
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def get_chunk(lst, n, idx):
    return split_list(lst, n)[idx]

# def inference(args):
#     pass

def make_full_labels(total_input=108, total_gen=0):
    """
    只在每个token区块的首尾添加标签，其余为空，便于横纵坐标简洁显示。
    区间：0-13(SYS) 14-30(IMG) 31-(total_input-1)(QUESTION) total_input+(GEN)
    """
    labels = [""] * (total_input + total_gen)

    # system区
    labels[0] = "SYS_START"
    labels[13] = "SYS_END"

    # image区
    labels[14] = "IMG_START"
    labels[30] = "IMG_END"

    # question区
    labels[31] = "QUESTION_START"
    labels[total_input-1] = "QUESTION_END"

    # gen区（如有生成token）
    if total_gen > 0:
        labels[total_input] = "GEN_START"
        labels[total_input + total_gen - 1] = "GEN_END"

    return labels


def visualize_attention(multihead_attention,output_path="atten_map_1.png",title="Layer 5"):
    # Assuming the input is a numpy array of shape (1, num_heads, n_tokens, n_tokens)
    # First, we average the attention scores over the multiple heads
    total_input=108
    averaged_attention = torch.mean(multihead_attention, axis=1)[0].float()# Shape: (n_tokens, n_tokens)
    total_token = averaged_attention.shape[0]
    total_gen = total_token - total_input
    full_labels = make_full_labels(total_input=total_input, total_gen=total_gen)
    # pooling the attention scores  with stride 20
    averaged_attention = torch.nn.functional.avg_pool2d(averaged_attention.unsqueeze(0).unsqueeze(0), 1, stride=1).squeeze(0).squeeze(0)
    pooled_labels = full_labels
    plt.figure(figsize=(8, 8), dpi=400)
    cmap = plt.colormaps['viridis']
    log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())
    ax = sns.heatmap(
        averaged_attention,
        cmap=cmap,
        norm=log_norm,
        square=True,
        # cbar_kws={'label': 'Attention score'},
    )
    ax.set_xticks(range(len(pooled_labels)))
    ax.set_yticks(range(len(pooled_labels)))
    ax.set_xticklabels(pooled_labels, fontsize=5, rotation=90)
    ax.set_yticklabels(pooled_labels, fontsize=5, rotation=0)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    return None, averaged_attention
    
    # cmap = plt.cm.get_cmap("viridis")
    # plt.figure(figsize=(5, 5),dpi=400)

    # # Log normalization
    # log_norm = LogNorm(vmin=0.0007, vmax=averaged_attention.max())

    # # set the x and y ticks to 20x of the original


    # ax = sns.heatmap(averaged_attention,
    #             cmap=cmap,  # custom color map
    #             norm=log_norm,  # 
    #             # cbar_kws={'label': 'Attention score'},
    #             )
    
    # # remove the x and y ticks
    
    # # replace the x and y ticks with string

    # x_ticks = [str(i*1) for i in range(0,averaged_attention.shape[0])]
    # y_ticks = [str(i*1) for i in range(0,averaged_attention.shape[0])]
    # ax.set_xticks([i for i in range(0,averaged_attention.shape[0])])
    # ax.set_yticks([i for i in range(0,averaged_attention.shape[0])])
    # ax.set_xticklabels(x_ticks)
    # ax.set_yticklabels(y_ticks)

    # # change the x tinks font size
    # plt.xticks(fontsize=1)
    # plt.yticks(fontsize=1)
    
    # # make y label vertical
    # plt.yticks(rotation=0)
    # plt.xticks(rotation=90)     
    
    # plt.title(title)
    # # tight layout
    # plt.savefig(output_path, bbox_inches='tight')
    # # plt.show()

    # top_five_attentions = []
    # for row in averaged_attention:
    #     # Use torch.topk to get the top 5 values and their indices
    #     top_values, top_indices = torch.topk(row, 10)
    #     # Convert to lists and append to the overall list
    #     top_five_line = list(zip(top_indices.tolist(), top_values.tolist()))
    #     top_five_attentions.append(top_five_line)
        
    # return None ,averaged_attention    

def inference(message, processor, model, append_output=""):
    all_outputs = []
    outputs_attention = []
    prompt = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True) + append_output
    image_inputs, video_inputs = process_vision_info(message)
        
    inputs = processor(
        text=prompt,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        )
    inputs = inputs.to("cuda")

    #assert model.config.output_attentions is True or hasattr(model.config, "output_attentions")
        #print('assert true')
    generated_ids = model.generate(**inputs, 
                                   use_cache=True, 
                                   max_new_tokens=512, 
                                   do_sample=False,
                                   output_attentions=True,
                                   return_dict_in_generate=True,
        )
            
       
        #print('generated_ids', generated_ids, generated_ids.shape)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids['sequences'])
        ]
        
    out_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        # print('generated_ids', generated_ids.keys())
        # print('out_text', out_text)
        
    all_outputs.extend(out_text)
    outputs_attention.append(generated_ids['attentions'])
    return out_text, all_outputs, outputs_attention
    
def eval_model(args):
    
    # all_outputs = []
    # outputs_attention = []
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
    args.model_path, torch_dtype="auto", device_map="auto"
    ).eval()
    total_layers = model.config.num_hidden_layers
    #
    processor = AutoProcessor.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    # 读取问题并分块
    with open(args.question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    os.makedirs(os.path.dirname(args.answers_file), exist_ok=True)
    out_f = open(args.answers_file, "w", encoding="utf-8")

    for item in tqdm(questions):
        qid = item['id']
        raw_q = item['conversations'][0]['value'].replace('<image>', '').strip()
        formatted_question = QUESTION_TEMPLATE.format(Question=raw_q)

    
        message = [{
            "role": "user",
            "content": []
        }]

        # 图像（以 PIL.Image 加载）加入 content
        if 'image' in item:
            img_path = os.path.join(args.image_folder, item['image'])
            image = Image.open(img_path).convert("RGB")
            message[0]['content'].append({
                "type": "image",
                "image": image
            })
        else:
            image = None  # 以防后续用到

        # 加入文本
        message[0]['content'].append({
            "type": "text",
            "text": formatted_question
        })

        out_text, all_outputs, outputs_attention = inference(message, processor, model)
        #_, model_output, outputs_attention = inference(message, processor, model, append_output=out_text[0])
        result = {
            "question_id": qid,
            "prompt": raw_q,
            "text": out_text[0],
            "answer_id": shortuuid.uuid(),
            "model_id": os.path.basename(args.model_path)
        }

        out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
        out_f.flush()
        # return all_outputs, outputs_attention
        
        
    out_f.close()
    print(f" All results saved to {args.answers_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path',    type=str, required=True)
    parser.add_argument('--image-folder',  type=str, default='')
    parser.add_argument('--question-file', type=str, required=True)
    parser.add_argument('--answers-file',  type=str, required=True)
    parser.add_argument('--num-chunks',    type=int, default=1)
    parser.add_argument('--chunk-idx',     type=int, default=0)
    parser.add_argument('--temperature',   type=float, default=0.0)
    args = parser.parse_args()
    final_path="./attention"
    eval_model(args)
    #model_output_ori,outputs_attention = eval_model(args)
    # for i in outputs_attention: #这里的i[0]就是里面的第零个向量
    #     for j in range(0,28):
    #         top5_attention,average_attentions = visualize_attention(i[0][j].cpu(),output_path=final_path+"/atten_map_"+str(j)+".png",title="Layer "+str(j+1))

