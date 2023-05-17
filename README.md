# MDEL

Multi-Domain Expert Layers

# Environment Setup

To set up the development environment, run `make setup_dev`. This will setup the
pre-commit hooks.

## Creating Expert Datasets

First, make sure you followed the Environment Setup guidelines.

To create an expert dataset using the Pile data, follow these steps:

1. Download the Pile shard 1 data: `./scripts/get_pile_shard1_data.sh`
2. To set the domain, edit the variable `SUBSET_NAME` in
   `scripts/create_domain_pile_mix.sh`. This should be set to a valid value of
   the Pile's variable `pile_set_name`. A list of valid values can be found
   below.
3. Run the above script to process the dataset
4. Authenticate into Hugginface:
   `export HF_ACCESS_TOKEN={YOUR HUGGINGFACE TOKEN}`
5. Set the dataset name in `scripts/upload_to_hf.sh`
6. Run the above script to upload the processed dataset to HuggingFace

### Pile Subsets

- Pile-CC
- PubMed Central
- Books3†
- OpenWebText2
- ArXiv
- Github
- FreeLaw
- Stack Exchange
- USPTO Backgrounds
- PubMed Abstracts
- Gutenberg (PG-19)†
- OpenSubtitles†
- Wikipedia (en)†
- DM Mathematics†
- Ubuntu IRC
- BookCorpus2
- EuroParl†
- HackerNews
- YoutubeSubtitles
- PhilPapers
- NIH ExPorter
- Enron Emails†

# Training Expert Models

1. Clone this repo and follow the Environment Setup instructions
2. Set up HF authentication: `export HUGGING_FACE_HUB_TOKEN=[FILL ME]`
3. Set up W&B authentication: `export WANDB_API_KEY=[FILL ME]`
4. Edit the variable `DATASET` in script `src/mdel/train.sh` to match a valid
   dataset name on the
   [MDEL HF](https://huggingface.co/Multi-Domain-Expert-Layers).
5. Run the above script in background mode to start the training: `./train.sh &`
6. The trained model should be uploaded to the MDEL HF

# Merging Expert Models

1. Clone this repo and follow the Environment Setup instructions
2. Set up HF authentication: `export HUGGING_FACE_HUB_TOKEN=[FILL ME]`
3. Run the merge script

```bash
python src/mdel/merge_experts.py \
   --hf-repo your_hf_username/desired_name_of_merged_model \
   -e mdel/expert_1 \
   -e mdel/expert_2 \
   -e mdel/expert_n
```

# Evaluating Perplexity of Models

1. Clone this repo and follow the Environment Setup instructions
2. Set up HF authentication: `export HUGGING_FACE_HUB_TOKEN=[FILL ME]`
3. Run the perplexity script

```bash
python3 src/mdel/calculate_perplexity.py \
   --model Multi-Domain-Expert-Layers/expert-arxiv \
   --dataset Multi-Domain-Expert-Layers/arxiv \
   --split validation_domain
```

# References

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., ... &
Leahy, C. (2020).The pile: An 800gb dataset of diverse text for language
modeling. _arXiv preprint arXiv:2101.00027_.
