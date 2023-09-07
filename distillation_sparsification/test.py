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

ds = load_from_disk("processed_ds/processed")
print(ds)
train_dataset = ds["train"]
eval_dataset = ds["validation"]


eval_dataset = eval_dataset.select((i for i in range(len(eval_dataset)-1)))

for k,v in eval_dataset[-1].items():
    print(k, torch.tensor(v).shape)
# for k,v in eval_dataset[-1].items():
#     print(k, v.shape)


# tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b", trust_remote_code=True)
# data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
# if tokenizer.pad_token is None:
#     tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# model = AutoModelForCausalLM.from_pretrained('student/', trust_remote_code=True, torch_dtype=torch.bfloat16)

# # ds_config = {
# #     "tensor_parallel": {"tp_size": 1},
# #     "dtype": "fp16",
# #     "replace_with_kernel_inject": True,
# #     "replace_method": "auto",
# # }

# # ds_model = deepspeed.init_inference(model=model, config=ds_config)

# per_device_train_batch_size = 2
# per_device_eval_batch_size = 2
# learning_rate = 3e-4
# lr_scheduler_type = "cosine"
# num_warmup_steps = 100
# max_train_steps = 1_000
# num_train_epochs = 1
# weight_decay = 0.01
# gradient_accumulation_steps = 1
# output_dir = 'log/'
# with_tracking = True
# report_to = "tensorboard"

# accelerator_log_kwargs = {}

# if with_tracking:
#     accelerator_log_kwargs["log_with"] = report_to
#     accelerator_log_kwargs["project_dir"] = output_dir

# accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, log_with="tensorboard", project_dir="log")

# train_dataloader = DataLoader(
#     train_dataset, shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
# )
# eval_dataloader = DataLoader(
#     eval_dataset, collate_fn=data_collator, batch_size=per_device_eval_batch_size
# )

# no_decay = ["bias", "layer_norm.weight"]
# optimizer_grouped_parameters = [
#     {
#         "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
#         "weight_decay": weight_decay,
#     },
#     {
#         "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
#         "weight_decay": 0.0,
#     },
# ]
# optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

# overrode_max_train_steps = False
# num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
# if max_train_steps is None:
#     max_train_steps = num_train_epochs * num_update_steps_per_epoch
#     overrode_max_train_steps = True

# lr_scheduler = get_scheduler(
#     name=lr_scheduler_type,
#     optimizer=optimizer,
#     num_warmup_steps=num_warmup_steps * gradient_accumulation_steps,
#     num_training_steps=max_train_steps * gradient_accumulation_steps,
# )

# model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
#     model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
# )

# experiment_config = {
#     "num_iterations": 1,
#     "learning_rate": 3e-4,
# }


# accelerator.init_trackers("clm_no_trainer", experiment_config)

# # We need to recalculate our total training steps as the size of the training dataloader may have changed.
# num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
# if overrode_max_train_steps:
#     max_train_steps = num_train_epochs * num_update_steps_per_epoch
# # Afterwards we recalculate our number of training epochs
# num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

# # Train!
# total_batch_size = per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps

# progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
# completed_steps = 0
# starting_epoch = 0

# # update the progress_bar if load from checkpoint
# progress_bar.update(completed_steps)
# print("training started")
# for epoch in range(starting_epoch, num_train_epochs):
#     print("Train!")
#     model.train()
#     if with_tracking:
#         total_loss = 0

#     active_dataloader = train_dataloader
#     for step, batch in enumerate(active_dataloader):
#         with accelerator.accumulate(model):
#             batch.pop("token_type_ids")
#             outputs = model(**batch)
#             loss = outputs.loss
#             # We keep track of the loss at each epoch
#             if with_tracking:
#                 total_loss += loss.detach().float()
#             accelerator.backward(loss)
#             optimizer.step()
#             lr_scheduler.step()
#             optimizer.zero_grad()
#             accelerator.log({"training_loss": loss}, step=step)

#         # Checks if the accelerator has performed an optimization step behind the scenes
#         if accelerator.sync_gradients:
#             progress_bar.update(1)
#             completed_steps += 1

#         if completed_steps >= max_train_steps:
#             break

#     model.eval()
#     losses = []
#     for step, batch in enumerate(eval_dataloader):
#         with torch.no_grad():
#             batch.pop("token_type_ids")
#             outputs = model(**batch)

#         loss = outputs.loss
#         losses.append(accelerator.gather_for_metrics(loss.repeat(per_device_eval_batch_size)))

#     losses = torch.cat(losses)
#     try:
#         eval_loss = torch.mean(losses)
#         perplexity = math.exp(eval_loss)
#     except OverflowError:
#         perplexity = float("inf")

#     logger.info(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")
#     if with_tracking:
#         accelerator.log(
#             {
#                 "perplexity": perplexity,
#                 "eval_loss": eval_loss,
#                 "train_loss": total_loss.item() / len(train_dataloader),
#                 "epoch": epoch,
#                 "step": completed_steps,
#             },
#             step=completed_steps,
#         )
        # print(f"epoch: {epoch}")
        # print(f"eval_loss: {eval_loss}")
        # print(f"train_loss: {total_loss.item() / len(train_dataloader)}")
        # print(f"perplexity: {perplexity}")
# accelerator.end_training()



