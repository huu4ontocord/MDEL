# pylint: disable=missing-docstring,C0301,W1203,R1720,C0116,R0912,R0914,R0916,C0103,W0613,R0915,W0632,E0401
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.
Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling
# task. Pointers for this are left as comments.

import argparse
import logging
import math
import os
import sys
from distutils.util import strtobool
from itertools import chain
from pathlib import Path

import datasets
import evaluate
import transformers
import wandb
import yaml
from datasets import load_dataset
from huggingface_hub import ModelCard, Repository
from transformers import (CONFIG_MAPPING, MODEL_FOR_CAUSAL_LM_MAPPING,
                          AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          Trainer, TrainingArguments, default_data_collator,
                          is_torch_tpu_available, set_seed)
from transformers.modelcard import TrainingSummary
from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import OptimizerNames
from transformers.utils.versions import require_version
from wandb.sdk.lib.runid import generate_id

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.28.0.dev0")

require_version(
    "datasets>=1.8.0",
    "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt",
)

logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def _strtobool(x):
    return bool(strtobool(x))


def read_yamls(dir):
    conf = {}
    no_conf = True

    for config_file in Path(dir).glob("**/*.yaml"):
        no_conf = False
        with config_file.open("r") as f:
            conf.update(yaml.safe_load(f))

    if no_conf:
        print(f"WARNING: No yaml files found in {dir}")

    return conf


def argument_parsing(notebook=False, notebook_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="""
        Multiple configs can be passed to set different options.
        For example, run as:

           ./trainer_sft.py --configs default juweles

    """,
    )
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--deepspeed", action="store_true")
    parser.add_argument("--no_deepspeed", action="store_true")
    parser.add_argument("--wandb-entity", type=str, default="open-assistant")
    parser.add_argument(
        "--resume_from_checkpoint",
        action="store_true",
        help="Resume from last saved checkpoint",
    )
    parser.add_argument("--rng_seed", type=int, help="rng seed")
    parser.add_argument(
        "--show_dataset_stats",
        action="store_true",
        help="Show dataset stats",
        default=False,
    )
    parser.set_defaults(deepspeed=False)

    if notebook:
        args, remaining = parser.parse_known_args(notebook_args)
    else:
        args, remaining = parser.parse_known_args()

    # Config from YAML
    conf = {}
    configs = read_yamls("configs/")
    conf.update(configs["defaults"])
    try:
        for name in args.configs:
            if "," in name:
                for n in name.split(","):
                    conf.update(configs[n])
            else:
                conf.update(configs[name])
    except KeyError as e:
        print(f'Error: Could not find the config "{e.args[0]}" in config.yaml')
        exit(1)

    conf["wandb_entity"] = args.wandb_entity
    conf["local_rank"] = args.local_rank
    conf["deepspeed"] = args.deepspeed
    if args.no_deepspeed:
        conf['deepspeed'] = None
    conf["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.rng_seed is not None:
        conf["rng_seed"] = args.rng_seed
    conf["show_dataset_stats"] = args.show_dataset_stats

    # get the world size in deeepspeed
    if conf["deepspeed"]:
        conf["world_size"] = int(os.getenv("WORLD_SIZE", default="1"))
    else:
        conf["world_size"] = 1

    # Override config from command-line
    parser = argparse.ArgumentParser()
    for key, value in conf.items():
        type_ = type(value) if value is not None else str
        if type_ == bool:
            type_ = _strtobool
        parser.add_argument(f"--{key}", type=type_, default=value)
        # Allow --no-{key}  to remove it completely
        parser.add_argument(f"--no-{key}", dest=key, action="store_const", const=None)

    return parser.parse_args(remaining)


def main():

    training_conf = argument_parsing()
    optimizer = (
        OptimizerNames.ADAMW_BNB
        if training_conf.quantization
        else OptimizerNames.ADAMW_HF
    )

    args = TrainingArguments(
        output_dir=training_conf.output_dir,
        num_train_epochs=training_conf.num_train_epochs,
        warmup_steps=training_conf.warmup_steps,
        learning_rate=float(training_conf.learning_rate),
        deepspeed=training_conf.deepspeed if training_conf.deepspeed else None,
        optim=optimizer,
        fp16=training_conf.dtype in ["fp16", "float16"],
        bf16=training_conf.dtype in ["bf16", "bfloat16"],
        local_rank=training_conf.local_rank,
        gradient_checkpointing=training_conf.gradient_checkpointing,
        gradient_accumulation_steps=training_conf.gradient_accumulation_steps,
        per_device_train_batch_size=training_conf.per_device_train_batch_size,
        per_device_eval_batch_size=training_conf.per_device_eval_batch_size,
        adam_beta1=training_conf.adam_beta1,
        adam_beta2=training_conf.adam_beta2,
        adam_epsilon=float(training_conf.adam_epsilon),
        weight_decay=training_conf.weight_decay,
        max_grad_norm=training_conf.max_grad_norm,
        logging_steps=training_conf.logging_steps,
        save_total_limit=training_conf.save_total_limit,
        evaluation_strategy=training_conf.evaluation_strategy,
        eval_steps=training_conf.eval_steps,
        save_strategy=training_conf.save_strategy,
        save_steps=training_conf.save_steps,
        resume_from_checkpoint=training_conf.resume_from_checkpoint,
        report_to="wandb" if training_conf.log_wandb else None,
        do_train=training_conf.do_train,
        do_eval=training_conf.do_eval,
        push_to_hub=training_conf.push_to_hub,
        push_to_hub_model_id=training_conf.push_to_hub_model_id,
        push_to_hub_organization=training_conf.push_to_hub_organization,
    )

    if len(args.report_to) >= 1 and args.local_rank == 0:
        for report_to in args.report_to:
            if report_to == "wandb" and not args.deepspeed:
                wandb_track = True
                wandb_api = wandb.Api()
                wandb_project = training_conf.wandb_project
                wandb_entity = training_conf.wandb_entity
                wandb_run_name = training_conf.wandb_run_name
                if wandb_project is not None:
                    os.environ["WANDB_PROJECT"] = wandb_project
                else:
                    raise ValueError(
                        "wandb_project must be specified if report_to is wandb"
                    )
                if wandb_run_name is not None:
                    os.environ["WANDB_NAME"] = wandb_run_name
                else:
                    os.environ["WANDB_NAME"] = (
                        f"{training_conf.model_name_or_path.split('/')[-1]}"
                        f"-{training_conf.dataset_name.split('/')[-1]}"
                    )
                if wandb_entity is not None:
                    os.environ["WANDB_ENTITY"] = wandb_entity
                else:
                    os.environ["WANDB_ENTITY"] = wandb_api.default_entity
                run_id = generate_id()
                os.environ["WANDB_RUN_ID"] = run_id
                wandb_run_url = f"https://wandb.ai/{training_conf.wandb_entity}/" \
                                f"{training_conf.wandb_project}/runs/{run_id}"

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if args.should_log:
        # The default of args.log_level is passive, so we set log
        # level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {args.local_rank}, device: {args.device}, n_gpu: {args.n_gpu}"
        f"distributed training: {bool(args.local_rank != -1)}, 16-bits training: {args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(args.output_dir) and args.do_train and not args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is None and len(os.listdir(args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(args.seed)
    if training_conf.validation_splits is not None:
        training_conf.validation_splits = training_conf.validation_splits.split(",")
    if training_conf.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            training_conf.dataset_name,
            cache_dir=training_conf.cache_dir,
            use_auth_token=True if training_conf.use_auth_token else None,
            streaming=training_conf.streaming,
        )
        if "validation" not in raw_datasets.keys() and (
            training_conf.validation_splits is None or all(split not in raw_datasets.keys()
                                                           for split in training_conf.validation_splits)
        ):
            raw_datasets["validation"] = load_dataset(
                training_conf.dataset_name,
                training_conf.dataset_config_name,
                split=f"train[:{training_conf.validation_split_percentage}%]",
                cache_dir=training_conf.cache_dir,
                use_auth_token=True if training_conf.use_auth_token else None,
                streaming=training_conf.streaming,
            )
            raw_datasets["train"] = load_dataset(
                training_conf.dataset_name,
                training_conf.dataset_config_name,
                split=f"train[{training_conf.validation_split_percentage}%:]",
                cache_dir=training_conf.cache_dir,
                use_auth_token=True if training_conf.use_auth_token else None,
                streaming=training_conf.streaming,
            )
    else:
        data_files = {}
        dataset_args = {}
        if training_conf.train_file is not None:
            data_files["train"] = training_conf.train_file
        if training_conf.validation_file is not None:
            data_files["validation"] = training_conf.validation_file
        extension = (
            training_conf.train_file.split(".")[-1]
            if training_conf.train_file is not None
            else training_conf.validation_file.split(".")[-1]
        )
        if extension == "txt":
            extension = "text"
            dataset_args["keep_linebreaks"] = training_conf.keep_linebreaks
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=training_conf.cache_dir,
            use_auth_token=True if training_conf.use_auth_token else None,
            **dataset_args,
        )
        # If no validation data is there, validation_split_percentage will be
        # used to divide the dataset.
        if "validation" not in raw_datasets.keys() and (
            training_conf.validation_splits is None or all(split not in raw_datasets.keys()
                                                           for split in training_conf.validation_splits)
        ):
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{training_conf.validation_split_percentage}%]",
                cache_dir=training_conf.cache_dir,
                use_auth_token=True if training_conf.use_auth_token else None,
                **dataset_args,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{training_conf.validation_split_percentage}%:]",
                cache_dir=training_conf.cache_dir,
                use_auth_token=True if training_conf.use_auth_token else None,
                **dataset_args,
            )

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": training_conf.cache_dir,
        "revision": training_conf.model_revision,
        "use_auth_token": True if training_conf.use_auth_token else None,
    }
    if training_conf.config_name:
        config = AutoConfig.from_pretrained(training_conf.config_name, **config_kwargs)
    elif training_conf.model_name_or_path:
        config = AutoConfig.from_pretrained(
            training_conf.model_name_or_path, **config_kwargs
        )
    else:
        config = CONFIG_MAPPING[training_conf.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if training_conf.config_overrides is not None:
            logger.info(f"Overriding config: {training_conf.config_overrides}")
            config.update_from_string(training_conf.config_overrides)
            logger.info(f"New config: {config}")

    tokenizer_kwargs = {
        "cache_dir": training_conf.cache_dir,
        "use_fast": training_conf.use_fast_tokenizer,
        "revision": training_conf.model_revision,
        "use_auth_token": True if training_conf.use_auth_token else None,
    }
    if training_conf.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            training_conf.tokenizer_name, **tokenizer_kwargs
        )
    elif training_conf.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            training_conf.model_name_or_path, **tokenizer_kwargs
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if training_conf.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            training_conf.model_name_or_path,
            from_tf=bool(".ckpt" in training_conf.model_name_or_path),
            config=config,
            cache_dir=training_conf.cache_dir,
            revision=training_conf.model_revision,
            use_auth_token=True if training_conf.use_auth_token else None,
            low_cpu_mem_usage=training_conf.low_cpu_mem_usage,
        )
    else:
        model = AutoModelForCausalLM.from_config(config)
        n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
        logger.info(
            f"Training new model from scratch - Total size={n_params / 2 ** 20:.2f}M params"
        )

    training_layers = [int(layer) for layer in training_conf.training_layers.split(",")]
    # freeze model
    model.requires_grad_(False)
    # unfreeze the layer to be trained
    for layer in training_layers:
        model.gpt_neox.layers[layer].requires_grad_(True)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if args.do_train:
        column_names = list(raw_datasets["train"].features)
    else:
        column_names = list(raw_datasets["validation"].features)
    text_column_name = "text" if "text" in column_names else column_names[0]

    # since this will be pickled to avoid _LazyModule error in Hasher force
    # logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger(
        "transformers.tokenization_utils_base"
    )

    def tokenize_function(examples):
        with CaptureLogger(tok_logger) as cl:
            output = tokenizer(examples[text_column_name])
        # clm input could be much much longer than block_size
        if "Token indices sequence length is longer than the" in cl.out:
            tok_logger.warning(
                "^^^^^^^^^^^^^^^^ Please ignore the warning above - this long input will be chunked into smaller bits"
                " before being passed to the model."
            )
        return output

    with args.main_process_first(desc="dataset map tokenization"):
        if not training_conf.streaming:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=training_conf.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not training_conf.overwrite_cache,
                desc="Running tokenizer on dataset",
            )
        else:
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                remove_columns=column_names,
            )

    if training_conf.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if training_conf.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({training_conf.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(training_conf.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our
    # dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i: i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    with args.main_process_first(desc="grouping texts together"):
        if not training_conf.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=training_conf.preprocessing_num_workers,
                load_from_cache_file=not training_conf.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if training_conf.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), training_conf.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if args.do_eval:
        if training_conf.validation_splits is not None:
            eval_datasets = {
                split: lm_datasets[split] for split in training_conf.validation_splits
            }
        else:
            eval_datasets = {"validation": lm_datasets["validation"]}
        if training_conf.max_eval_samples is not None:
            for key, eval_dataset in eval_datasets.items():
                max_eval_samples = min(
                    len(eval_dataset), training_conf.max_eval_samples
                )
                eval_datasets[key] = eval_dataset.select(range(max_eval_samples))

        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_datasets if args.do_eval else None,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change
        # it.
        data_collator=default_data_collator,
        compute_metrics=compute_metrics
        if args.do_eval and not is_torch_tpu_available()
        else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics
        if args.do_eval and not is_torch_tpu_available()
        else None,
    )

    # Training
    if args.do_train:
        checkpoint = None
        if args.resume_from_checkpoint is not None:
            checkpoint = args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            training_conf.max_train_samples
            if training_conf.max_train_samples is not None
            else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        eval_dataset = datasets.concatenate_datasets(
            [eval_dataset for eval_dataset in eval_datasets.values()]
        )

        metrics = trainer.evaluate(eval_dataset=eval_dataset)

        max_eval_samples = (
            training_conf.max_eval_samples
            if training_conf.max_eval_samples is not None
            else len(eval_dataset)
        )
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    kwargs = {
        "finetuned_from": training_conf.model_name_or_path,
        "tasks": "text-generation",
    }
    if training_conf.dataset_name is not None:
        kwargs["dataset_tags"] = training_conf.dataset_name
        if training_conf.dataset_config_name is not None:
            kwargs["dataset_args"] = training_conf.dataset_config_name
            kwargs[
                "dataset"
            ] = f"{training_conf.dataset_name} {training_conf.dataset_config_name}"
        else:
            kwargs["dataset"] = training_conf.dataset_name

    if args.push_to_hub:
        training_summary = TrainingSummary.from_trainer(trainer, **kwargs)
        if args.local_rank == 0:
            training_summary.dataset_metadata[0].pop(
                "config", None
            )  # Will get error if default which is null
            summary_card = training_summary.to_model_card()
            if wandb_track:
                summary_card = f"{summary_card}\n\n## Wandb Report\n {wandb_run_url}"
            model_card = ModelCard(summary_card)
            repo = Repository(local_dir=args.output_dir)
            repo.git_pull()
            with open(f"{args.output_dir}/README.md", "w") as f:
                f.write(str(model_card))
            repo.push_to_hub()
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()
