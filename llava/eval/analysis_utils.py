import torch
import numpy as np
import matplotlib.pyplot as plt

def compute_distance_matrix(seq_len):
    idx = torch.arange(seq_len)
    return (idx[None, :] - idx[:, None]).abs()  # [Q, K]

def distance_wise_attention_analysis(attn_weights):
    """
    attn_weights: [B, H, Q, K]
    Returns: List[float], mean attention for each relative distance
    """
    B, H, Q, K = attn_weights.shape
    distance = compute_distance_matrix(K).to(attn_weights.device)  # [K, K]
    distance = distance.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)  # [B, H, Q, K]

    attn_weights = attn_weights.masked_fill(torch.isnan(attn_weights), 0.0)

    max_dist = Q - 1
    avg_attn_by_dist = []
    for d in range(max_dist + 1):
        mask = (distance == d)
        selected_vals = attn_weights[mask]
        if selected_vals.numel() > 0:
            avg_attn_by_dist.append(selected_vals.mean().item())
        else:
            avg_attn_by_dist.append(0.0)
    return avg_attn_by_dist

def plot_attention_by_distance(attn_list, label_list, max_dist=None):
    plt.figure(figsize=(6, 4))
    for attn_vals, label in zip(attn_list, label_list):
        x = np.arange(len(attn_vals))
        if max_dist:
            x = x[:max_dist]
            attn_vals = attn_vals[:max_dist]
        plt.plot(x, attn_vals, label=label)
    plt.xlabel("Relative Token Distance")
    plt.ylabel("Average Attention")
    plt.title("Distance-wise Attention Retention")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("distance_wise_analysis.pdf", dpi=300)
    plt.show()
