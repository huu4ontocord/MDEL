import argparse
import glob
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import Trainer
from make_student import *
from utils import label_smoothed_nll_loss, freeze_params
from make_student import *

def get_falcon(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    tokenizer =  AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model.seqlen = 2048
    return model, tokenizer



class DistillationTrainer(Trainer):
    # length_field_name should possibly be part of TrainingArguments instead
    def __init__(self, length_field_name=None, swap_lang_prob=3, alpha_ce=0.5, alpha_hid=0.5, alpha_clm=0.5, temperature=2.0, all_teachers={}, normalize_hidden=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.length_field_name = length_field_name
        
        self.t_model, self.tokenizer = get_falcon("tiiuae/falcon-7b")
        
        # self.s_model, layer_ids = create_student_by_copying_alternating_layers(t_model, d=16, save_path='student/')
        
        freeze_params(self.t_model)
        
        self.d_matches = get_layers_to_supervise(
            n_student=16, n_teacher=32
        )
        
        self.restrict_ce_to_mask = False        
        self.alpha_ce = alpha_ce
        self.alpha_hid = alpha_hid
        self.alpha_clm = alpha_clm
        self.temperature = temperature
        self.normalize_hidden = normalize_hidden
        
        self.ce_loss_fct = nn.KLDivLoss(reduction="batchmean") 
        self.lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        
        
    def calc_hidden_loss(self, attention_mask, lm_labels,hidden_states, hidden_states_T, matches, normalize_hidden):
        """MSE(student_hid, teacher_hid[matches]). Called "Intermediate supervision" in paper. Inspired by TinyBERT."""
        msg = "expected list or tuple for hidden_states, got tensor of shape: "
        assert not isinstance(hidden_states, torch.Tensor), f"{msg}{hidden_states.shape}"
        assert not isinstance(hidden_states_T, torch.Tensor), f"{msg}{hidden_states_T.shape}"
        if self.restrict_ce_to_mask:
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
    
    def calc_ce_loss(self, attention_mask, lm_labels, s_logits, t_logits):
        """Copy pasted from distillbert (transformers/examples/distillation/)"""
        # mask has False at padding_idx
        if self.restrict_ce_to_mask:
            mask = (lm_labels > -1)  # (bs, seq_length, voc_size)
        else:
            mask = attention_mask  # (bs, seq_length, voc_size)
            
        sel_mask = mask.unsqueeze(-1).expand_as(s_logits)
        vocab_size = s_logits.size(-1)
        s_logits_slct = torch.masked_select(s_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        t_logits_slct = torch.masked_select(t_logits, sel_mask)  # (bs * seq_length * voc_size) modulo the 1s in mask
        s_logits_slct = s_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        t_logits_slct = t_logits_slct.view(-1, vocab_size)  # (bs * seq_length, voc_size) modulo the 1s in mask
        assert t_logits_slct.size() == s_logits_slct.size()
        loss_ce = (
            self.ce_loss_fct(
                nn.functional.log_softmax(s_logits_slct / self.temperature, dim=-1),
                nn.functional.softmax(t_logits_slct / self.temperature, dim=-1),
            )
            * (self.temperature) ** 2
        )
        return loss_ce
        
    def compute_loss(self, model, inputs, return_outputs=False):
        pad_token_id = self.tokenizer.pad_token_id
        input_ids, attn_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        lm_labels = input_ids.new(input_ids.size()).copy_(input_ids)
        lm_labels[~attn_mask] = -100  # previously `lm_labels[1-attn_mask] = -1`, cf pytorch 1.2.0 compatibility
        # sanity checks
        assert 0 <= input_ids.min() <= input_ids.max() < self.tokenizer.vocab_size
        # noinspection PyCallingNonCallable
        s_outputs = model(
            input_ids,
            attention_mask=None,
        )

        t_outputs = self.t_model(
            input_ids,
            attention_mask=None,
        )
        print(5)
        s_logits, s_hidden_states = s_outputs["logits"], s_outputs["hidden_states"]
        t_logits, t_hidden_states = t_outputs["logits"], t_outputs["hidden_states"]
        
        
        if self.alpha_ce > 0.0:
            loss_ce = self.calc_ce_loss(attn_mask, lm_labels, s_logits, t_logits)
                
        if self.alpha_hid > 0.0: 
            hid_loss = self.calc_hidden_loss(
                    attn_mask,
                    lm_labels,
                    s_hidden_states,
                    t_hidden_states,
                    self.d_matches,
                    normalize_hidden=self.normalize_hidden,
            )
        
        if self.alpha_clm > 0.0:
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            loss_clm = self.lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))        
        
        loss = self.alpha_ce * loss_ce + self.alpha_clm * loss_clm + self.alpha_hid * hid_loss
        
        return (loss, loss_ce, loss_clm, hid_loss, s_outputs) if return_outputs else (loss, loss_ce, loss_clm, hid_loss)

    
    