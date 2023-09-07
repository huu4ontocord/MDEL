import math
import time
import sys

import torch
import torch.nn as nn
import transformers

from sparsegpt import *
from modelutils import *
from utils import * 
from tracker import *
from lion import Lion

import sys
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

import gc
torch.cuda.empty_cache()
gc.collect()


ds = load_from_disk("ds/processed")
train_dataset = ds["train"]
eval_dataset = ds["validation"]
test_dataset = ds["test"]

eval_dataset = eval_dataset.select(
    (
        i for i in range(len(eval_dataset)) 
        if len(eval_dataset[i]['input_ids']) == 2048
    )
)


def get_falcon(model):
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    tokenizer =  AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model, trust_remote_code=True, output_hidden_states=True, torch_dtype=torch.bfloat16)
    model.seqlen = 2048
    return model, tokenizer

t_model, tokenizer = get_falcon("tiiuae/falcon-7b")
t_model.config.output_hidden_states=True
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# s_model = AutoModelForCausalLM.from_pretrained('student1/', trust_remote_code=True, torch_dtype=torch.bfloat16)
s_model = AutoModelForCausalLM.from_pretrained('falcon1b/adamwlr3e-4', trust_remote_code=True, torch_dtype=torch.bfloat16)

s_model.config.output_hidden_states=True

per_device_train_batch_size = 2
per_device_eval_batch_size = 2
learning_rate = 3e-4
lr_scheduler_type = "cosine"
num_warmup_steps = 300
max_train_steps = 2997 
num_train_epochs = 1
weight_decay = 0.0
gradient_accumulation_steps = 16
output_dir = 'log/'
with_tracking = True
report_to = "wandb"
alpha_ce = 0.8
alpha_clm = 1.0
alpha_hid = 1.0
temperature = 2.0
max_grad_norm = 1.0
restrict_ce_to_mask = False
normalize_hidden = False
train = False
model_dir = "falcon1b/no_hid"
is_save = True
d_matches = get_layers_to_supervise(
            n_student=8, n_teacher=32
        )
lm_loss_fct = nn.CrossEntropyLoss(ignore_index=-100)

accelerator_log_kwargs = {}


if with_tracking:
    accelerator_log_kwargs["log_with"] = report_to
    accelerator_log_kwargs["project_dir"] = output_dir
    
accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, **accelerator_log_kwargs)

train_dataloader = DataLoader(
    train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size, drop_last=True
)
eval_dataloader = DataLoader(
    eval_dataset, shuffle=False, collate_fn=data_collator, batch_size=per_device_eval_batch_size, drop_last=True
)
test_dataloader = DataLoader(
    test_dataset, shuffle=False, collate_fn=data_collator, batch_size=per_device_eval_batch_size, drop_last=True
)

no_decay = ["bias", "layer_norm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in s_model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in s_model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, betas=(0.9, 0.95), lr=learning_rate)
# optimizer = Lion(optimizer_grouped_parameters, betas=(0.95, 0.98), lr=learning_rate)


overrode_max_train_steps = False
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)

if max_train_steps is None:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
    num_training_steps=max_train_steps * gradient_accumulation_steps,
)

t_model, s_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    t_model, s_model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

experiment_config = {
    "num_iterations": 1,
    "learning_rate": 3e-4,
}

accelerator.print(f"train_dataset: {train_dataset}")
accelerator.print(f"eval_dataset: {eval_dataset}")
accelerator.print(f"test_dataset: {test_dataset}")


accelerator.init_trackers("clm_no_trainer", experiment_config, init_kwargs={"wandb": {"entity": "ontocord"}})

# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if overrode_max_train_steps:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs

num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
accelerator.print(f"num_train_epochs: {num_train_epochs}")
if num_train_epochs > 1:
    sys.exit()

# Train!
total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0

# update the progress_bar if load from checkpoint
progress_bar.update(completed_steps)
accelerator.print("training started")
if train:
    for epoch in range(starting_epoch, num_train_epochs):
        accelerator.print("Train!")
        s_model.train()
        if with_tracking:
            total_loss = 0

        active_dataloader = train_dataloader
        for step, batch in enumerate(active_dataloader):
            with accelerator.accumulate(s_model):
                input_ids, attn_mask, labels = batch["input_ids"], batch["attention_mask"], batch["labels"]
                lm_labels = batch["input_ids"]
                assert 0 <= input_ids.min() <= input_ids.max() < tokenizer.vocab_size

                with torch.no_grad():
                    t_outputs = t_model(
                        input_ids,
                        attention_mask=None,
                    )

                s_outputs = s_model(
                    input_ids,
                    attention_mask=None,
                )

                t_logits, t_hidden_states = t_outputs["logits"], t_outputs["hidden_states"]
                s_logits, s_hidden_states = s_outputs["logits"], s_outputs["hidden_states"]

                if alpha_ce > 0.0:
                    loss_ce = calc_ce_loss(attn_mask, lm_labels, s_logits, t_logits, temperature, restrict_ce_to_mask)

                if alpha_hid > 0.0: 
                    hid_loss = calc_hidden_loss(
                            attn_mask,
                            lm_labels,
                            s_hidden_states,
                            t_hidden_states,
                            d_matches,
                            normalize_hidden=normalize_hidden,
                            restrict_ce_to_mask = restrict_ce_to_mask
                    )
                else:
                    hid_loss = 0

                if alpha_clm > 0.0:
                    shift_logits = s_logits[..., :-1, :].contiguous()
                    shift_labels = lm_labels[..., 1:].contiguous()
                    loss_clm = lm_loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))        

                loss = (alpha_ce * loss_ce) + (alpha_clm * loss_clm) + (alpha_hid * hid_loss)            

                # loss = outputs.loss
                # We keep track of the loss at each epoch
                if with_tracking:
                    total_loss += loss.detach().float()
                accelerator.backward(loss)
                accelerator.clip_grad_norm_(s_model.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                accelerator.log({"training_loss": loss, "clm_loss": loss_clm, "hid_loss": hid_loss, "ce_loss": loss_ce}, step=step)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= max_train_steps:
                break

        if is_save:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(s_model)
            unwrapped_model.save_pretrained(
                model_dir, is_main_process=accelerator.is_main_process, save_function=accelerator.save
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(model_dir)

        s_model.eval()
        losses = []
        for step, batch in enumerate(eval_dataloader):
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
else:
    for step, batch in enumerate(test_dataloader):
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
    if False:
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
accelerator.end_training()






"""
eval_loss: 2.84375
perplexity: 17.180070153709277
train_loss: 2.6160560232485404
"""


"""
eval_loss: 2.390625
perplexity: 10.920317008742302
train_loss: 12.452868275646372
"""