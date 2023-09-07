import math
import time

import torch
import torch.nn as nn
import transformers

from sparsegpt import *
from modelutils import *
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
from falcon7b import modelling_RW
from datasets import load_dataset
from make_student import *
import multiprocessing
from datasets import load_from_disk
from itertools import chain



tokenizer =  AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
ds = load_dataset('JeanKaddour/minipile')
# print(ds.column_names)

def preprocess_function(examples):
    return tokenizer(examples["text"], )

block_size = 2048

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
    # customize this part to your needs.
    if total_length >= block_size:
        total_length = (total_length // block_size) * block_size
    else:
        total_length = 0
    # Split by chunks of block_size.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    del result['token_type_ids']
    return result


tokenized_minipile = ds.map(
    preprocess_function,
    batched=True,
    num_proc=multiprocessing.cpu_count(),
    remove_columns=ds['validation'].column_names,
)

tokenized_minipile.save_to_disk("ds/raw")

# tokenized_minipile = load_from_disk("ds/raw")

ds = tokenized_minipile.map(group_texts, batched=True, num_proc=multiprocessing.cpu_count())


ds.save_to_disk("ds/processed")

# ds = load_from_disk("ds/processed")
# print(ds)
# train_dataset = ds["train"]
# eval_dataset = ds["validation"]
# test_dataset = ds["test"]
# print(test_dataset)
# eval_dataset = eval_dataset.select(
#     (
#         i for i in range(len(eval_dataset)) 
#         if len(eval_dataset[i]['input_ids']) == 2048
#     )
# )

# test_dataset = test_dataset.select(
#     (
#         i for i in range(len(test_dataset)) 
#         if len(test_dataset[i]['input_ids']) == 2048
#     )
# )

# train_dataset = train_dataset.select(
#     (
#         i for i in range(len(train_dataset)) 
#         if len(train_dataset[i]['input_ids']) == 2048
#     )
# )

# ds["train"] = train_dataset
# ds["validation"] = eval_dataset
# ds["test"] = test_dataset

print(ds)
# print(eval_dataset)
# print(test_dataset)
# print(train_dataset)