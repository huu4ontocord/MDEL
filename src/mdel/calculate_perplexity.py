from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import AutoTokenizer
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import math
import argparse


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer if args.tokenizer else args.model
    )
    model = AutoModelForCausalLM.from_pretrained(args.model)

    tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def prep_dataset(args, tokenizer):
    ds = load_dataset(args.dataset, split=args.split)
    ds = ds.flatten()
    print("Loaded Dataset", ds[0])

    def preprocess_function(examples):
        return tokenizer(
            [" ".join(x) for x in examples[args.dataset_key]],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized_ds = ds.map(
        preprocess_function,
        batched=True,
        num_proc=4,
        remove_columns=ds.column_names,
    )

    return tokenized_ds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate perplexity of a model on a dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the HF model e.g. MDEL/merged-arxiv-github",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=False,
        help="Optional tokenizer name e.g. MDEL/merged-arxiv-github. If not provided, will use the model name",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Name of the HF dataset e.g. MDEL/pubmed_abstracts",
    )
    parser.add_argument(
        "--split",
        type=str,
        required=True,
        help="Name of the split to evaluate on e.g. validation",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        required=False,
        default=1024,
        help="Max length of the input sequence",
    )
    parser.add_argument(
        "--dataset-key",
        type=str,
        required=False,
        default='text',
        help="Key to use to access the dataset e.g. text or answers.text",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer, model = load_model(args)
    dataset = prep_dataset(args, tokenizer)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir="perplexity-results",
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset,
        data_collator=data_collator,
    )

    eval_results = trainer.evaluate()
    perplexity = math.exp(eval_results['eval_loss'])
    message = f"Perplexity for {args.model} on {args.dataset}[{args.split}]: {perplexity}"

    print(message)

    with open(f"perplexity-{args.model}-{args.dataset}-{args.split}.txt", "w") as f:
        f.write(message)

