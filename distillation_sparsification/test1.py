import math
import time

import torch
import torch.nn as nn
import transformers

from sparsegpt import *
from modelutils import *
import sys
#from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

from falcon7b import modelling_RW
from datasets import load_dataset
from make_student import *
import multiprocessing
from transformers import DataCollatorForLanguageModeling
from distill import DistillationTrainer
from datasets import load_from_disk
from torch.utils.data import DataLoader

from accelerate import Accelerator, DistributedType, notebook_launcher
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from transformers import Trainer, TrainingArguments
import deepspeed

device = torch.device('cuda:3')
ds = load_from_disk("processed_ds/processed")
print(ds)
train_dataset = ds["test"]
eval_dataset = ds["validation"]

tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
print(tokenizer)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
t_model = AutoModelForCausalLM.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
print(t_model.config)
t_model.config.output_hidden_states=True

# model = AutoModelForCausalLM.from_pretrained('student/', trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
model, layer_ids = create_student_by_copying_alternating_layers(t_model, d=8, save_path='student1/')
print(model)
print(layer_ids)
# pytorch_total_params = sum(p.numel() for p in model.parameters())

# print(pytorch_total_params)
# train_dataloader = DataLoader(
#     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=1
# )

# batch = next(iter(train_dataloader))
# inputs_id = batch['input_ids'].to(device)
# print(inputs_id.shape)

# outputs = t_model(input_ids=inputs_id, attention_mask=None)
# print(list(outputs.keys()))