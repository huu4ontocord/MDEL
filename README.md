# MDEL

Multi-Domain Expert Layers

# Environment Setup

To set up the development environment, run `make setup_dev`. This will setup the
pre-commit hooks.

## Creating Expert Datasets

First, make sure you followed the Environment Setup guidelines.

To mix the USPTO data with Pile data, follow these steps:

1. Download the Pile shard 1 data: `scripts/get_pile_shard1_data.sh`
2. Process the dataset: `scripts/create_uspto_pile_mix.sh`
3. Authenticate into Hugginface:
   `export HF_HF_ACCESS_TOKEN={YOUR HUGGINGFACE TOKEN}`
4. Upload the processed dataset to HuggingFace: `scripts/upload_to_hf.sh`
