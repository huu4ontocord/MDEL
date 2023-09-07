import itertools
import json
import linecache
import math
import os
import pickle
import socket
from logging import getLogger
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple, Union

import git
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import Dataset, Sampler

def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=-100):
    """From fairseq"""
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    nll_loss = nll_loss.sum()  # mean()? Scared to break other math.
    smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False
        
        
def calc_ce_loss(attention_mask, lm_labels, s_logits, t_logits, temperature, restrict_ce_to_mask):
    """Copy pasted from distillbert (transformers/examples/distillation/)"""
    # mask has False at padding_idx
    ce_loss_fct = nn.KLDivLoss(reduction="batchmean") 
    if restrict_ce_to_mask:
        mask = (lm_labels > -1)  # (bs, seq_length, voc_size)
    else:
        mask = attention_mask  # (bs, seq_length, voc_size)

    mask = mask.unsqueeze(-1).expand_as(s_logits)
    mask = torch.gt(mask, 0)
    s_logits_slct = torch.masked_select(s_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
    s_logits_slct = s_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
    t_logits_slct = torch.masked_select(t_logits, mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
    t_logits_slct = t_logits_slct.view(-1, s_logits.size(-1))  # (bs * seq_length, voc_size) modulo the 1s in mask
    assert t_logits_slct.size() == s_logits_slct.size()

    loss_ce = (
        ce_loss_fct(
            nn.functional.log_softmax(s_logits_slct / temperature, dim=-1),
            nn.functional.softmax(t_logits_slct / temperature, dim=-1),
        )
        * (temperature) ** 2
    )
    return loss_ce


def calc_hidden_loss(attention_mask, lm_labels,hidden_states, hidden_states_T, matches, normalize_hidden, restrict_ce_to_mask):
    """MSE(student_hid, teacher_hid[matches]). Called "Intermediate supervision" in paper. Inspired by TinyBERT."""
    msg = "expected list or tuple for hidden_states, got tensor of shape: "
    assert not isinstance(hidden_states, torch.Tensor), f"{msg}{hidden_states.shape}"
    assert not isinstance(hidden_states_T, torch.Tensor), f"{msg}{hidden_states_T.shape}"
    if restrict_ce_to_mask:
        mask = lm_labels > -1  # (bs, seq_length, voc_size)
    else:
        mask = attention_mask  # (bs, seq_length, voc_size)

    mask = mask.to(hidden_states[0])
    valid_count = mask.sum() * hidden_states[0].size(-1)
    s_states = torch.stack([hidden_states[i] for i in range(len(matches))])
    t_states = torch.stack([hidden_states_T[j] for j in matches])
    assert s_states.shape == t_states.shape, f"{s_states.shape} != {t_states.shape}"
    if normalize_hidden:
        s_states = nn.functional.layer_norm(s_states, s_states.shape[1:])
        t_states = nn.functional.layer_norm(t_states, t_states.shape[1:])
    mse = nn.functional.mse_loss(s_states, t_states, reduction="none")
    masked_mse = (mse * mask.unsqueeze(0).unsqueeze(-1)).sum() / valid_count
    return masked_mse


def eval(s_model, ds, with_tracking=False):
    s_model.eval()
    lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
    losses = []
    for step, batch in enumerate(ds):
        with torch.no_grad():
            input_ids, attn_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
            lm_labels = batch["input_ids"]

            s_outputs = s_model(
                input_ids,
                attention_mask=None,
            )
            
        s_logits, s_hidden_states = s_outputs["logits"], s_outputs["hidden_states"]
        shift_logits = s_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()
        loss = lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))    
        
        losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))

    losses = torch.cat(losses)
    try:
        eval_loss = torch.mean(losses)
        perplexity = math.exp(eval_loss)
    except OverflowError:
        perplexity = float("inf")

    logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
    if with_tracking:
        accelerator.log(
            {
                "perplexity": perplexity,
                "eval_loss": eval_loss,
                "step": completed_steps,
                "train_loss": total_loss.item() / len(train_dataloader),
            },
            step=completed_steps,
        )
    accelerator.print(f"eval_loss: {eval_loss}")
    accelerator.print(f"perplexity: {perplexity}")
    accelerator.print(f"train_loss: {total_loss.item() / len(train_dataloader)}")