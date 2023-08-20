import torch
from torch.utils.checkpoint import checkpoint
import torch.nn as nn 
from transformers.activations import ACT2FN

class GPTNeoXMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense_h_to_4h = nn.Linear(512, 2048)
        self.dense_4h_to_h = nn.Linear(2048, 512)
        self.act = ACT2FN["gelu"]

    def forward(self, hidden_states):
        hidden_states = self.dense_h_to_4h(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.dense_4h_to_h(hidden_states)
        return hidden_states

import torch
from functools import partial
from torch import nn, einsum
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from .bpt_triton import matmul, add

from torch.nn.functional import scaled_dot_product_attention
from einops import rearrange
from datetime import datetime
# helper functions

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# regular attention

def attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    **kwargs
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    sim = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias):
        sim = sim + attn_bias

    mask_value = -torch.finfo(sim.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        sim = sim.masked_fill(~mask, mask_value)

    if causal:
        i, j = sim.shape[-2:]
        mask = torch.ones(i, j, device = q.device, dtype = torch.bool).triu(j - i + 1)
        sim = sim.masked_fill(mask, mask_value)

    sim = sim - sim.amax(dim = -1, keepdim = True).detach()
    attn = sim.softmax(dim = -1)

    out = einsum('b h i j, b h j d -> b h i d', attn, v)
    return out

# memory efficient attention

def summarize_qkv_chunk(q, k, v, mask, attn_bias_chunk, causal, qk_start_indices, dropout):
    q_start_index, k_start_index, q_chunk_size, k_chunk_size, device = *qk_start_indices, q.shape[-2], k.shape[-2], q.device

    weight = einsum('b h i d, b h j d -> b h i j', q, k)

    if exists(attn_bias_chunk):
        weight = weight + attn_bias_chunk

    mask_value = -torch.finfo(weight.dtype).max

    if exists(mask):
        mask = rearrange(mask, 'b j -> b 1 1 j')
        weight = weight.masked_fill(~mask, mask_value)

    if causal and q_start_index < (k_start_index + k_chunk_size - 1):
        causal_mask = torch.ones((q_chunk_size, k_chunk_size), dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
        weight = weight.masked_fill(causal_mask, mask_value)

    weight_max = weight.amax(dim = -1, keepdim = True).detach()
    weight = weight - weight_max

    exp_weight = weight.exp()

    exp_weight = F.dropout(exp_weight, p = dropout)

    weighted_value = einsum('b h i j, b h j d -> b h i d', exp_weight, v)

    return exp_weight.sum(dim = -1), weighted_value, rearrange(weight_max, '... 1 -> ...')

checkpointed_summarize_qkv_chunk = partial(checkpoint, summarize_qkv_chunk)

def memory_efficient_attention(
    q, k, v,
    mask = None,
    causal = False,
    attn_bias = None,
    q_bucket_size = 512,
    k_bucket_size = 1024,
    eps = 1e-8,
    dropout = 0.,
    training = False
):
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # function

    needs_backwards = q.requires_grad or k.requires_grad or v.requires_grad
    summarize_qkv_fn = checkpointed_summarize_qkv_chunk if needs_backwards else summarize_qkv_chunk

    # chunk all the inputs

    q_chunks = q.split(q_bucket_size, dim = -2)
    k_chunks = k.split(k_bucket_size, dim = -2)
    v_chunks = v.split(k_bucket_size, dim = -2)
    mask_chunks = mask.split(k_bucket_size, dim = -1) if exists(mask) else ((None,) * len(k_chunks))

    if exists(attn_bias):
        i, j = attn_bias.shape[-2:]
        attn_bias_chunks = attn_bias.split(q_bucket_size, dim = -2)
        attn_bias_chunks = list(map(lambda t: t.split(k_bucket_size, dim = -1), attn_bias_chunks))

    # loop through all chunks and accumulate

    out = []
    for q_index, q_chunk in enumerate(q_chunks):
        exp_weights = []
        weighted_values = []
        weight_maxes = []

        for k_index, (k_chunk, v_chunk, mask_chunk) in enumerate(zip(k_chunks, v_chunks, mask_chunks)):
            q_start_index = q_index * q_bucket_size
            k_start_index = k_index * k_bucket_size

            if causal and k_start_index > (q_start_index + q_chunk.shape[-2] - 1):
                # if chunk is to be all masked out causally, skip
                continue

            attn_bias_chunk = attn_bias_chunks[q_index][k_index] if exists(attn_bias) else None

            exp_weight_chunk, weighted_value_chunk, weight_max_chunk = summarize_qkv_fn(
                q_chunk,
                k_chunk,
                v_chunk,
                mask_chunk,
                attn_bias_chunk,
                causal,
                (q_start_index, k_start_index),
                dropout if training else 0.
            )

            exp_weights.append(exp_weight_chunk)
            weighted_values.append(weighted_value_chunk)
            weight_maxes.append(weight_max_chunk)

        weight_maxes = torch.stack(weight_maxes, dim = -1)

        weighted_values = torch.stack(weighted_values, dim = -1)
        exp_weights = torch.stack(exp_weights, dim = -1)

        global_max = weight_maxes.amax(dim = -1, keepdim = True)
        renorm_factor = (weight_maxes - global_max).exp().detach()

        exp_weights = exp_weights * renorm_factor
        weighted_values = weighted_values * rearrange(renorm_factor, '... c -> ... 1 c')

        all_values = weighted_values.sum(dim = -1)
        all_weights = exp_weights.sum(dim = -1)

        normalized_values = all_values / (rearrange(all_weights, '... -> ... 1') + eps)
        out.append(normalized_values)

    return torch.cat(out, dim = -2)

def blockwise_compute_ffn(cell, inputs, chunk_size):
    inputs = torch.split(inputs, chunk_size, dim=1)
    num_q = len(inputs)

    def ffn(cell, _, hidden_states):
        outputs = cell(hidden_states)
        return outputs
    
    outputs = []
    for i in range(num_q):
        outputs.append(ffn(cell, None, inputs[i]))
    
    res = torch.concat(outputs, dim=1)
    # res = rearrange(res, 'n b c d -> b (n c) d')
    return res

if __name__ == "__main__":
    # Blocked mem stuff
    q = torch.rand(2, 512, 16, 128)
    k = torch.rand(2, 512, 16, 128)
    v = torch.rand(2, 512, 16, 128)
    bias = torch.rand(2, 1, 512, 2048)

    # Blocked FFN Stuff
    x = torch.rand(2, 256, 512)
    cell = GPTNeoXMLP()
    startTime = datetime.now()
    y_pt_mem = memory_efficient_attention(q, k, v, q_bucket_size=512, k_bucket_size=512)
    print('pythonic mem eff attn', datetime.now() - startTime)

    torch.backends.cuda.sdp_kernel(True)
    torch.backends.cuda.enable_flash_sdp(True)
    startTime = datetime.now()
    y_pt_mem = scaled_dot_product_attention(q, k, v)
    print('pythonic mem eff attn', datetime.now() - startTime)

    startTime = datetime.now()
    y_pt_ffn = blockwise_compute_ffn(cell, x, 256)
    print('pythonic blocked ffn', datetime.now() - startTime)

    startTime = datetime.now()
    y_pt_ffn = blockwise_compute_ffn_triton(cell, x)
    print('pythonic blocked ffn', datetime.now() - startTime)
    print(y_pt_ffn.shape)
