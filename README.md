# MDEL

Multi-Domain Expert Layers

# Environment Setup

To set up the development environment, run `make venv setup_dev`. This will
setup the pre-commit hooks.

## Creating Expert Datasets

First, make sure you followed the Environment Setup guidelines.

To mix the USPTO data with Pile data, run the following scripts:

1. `scripts/get_uspto_data.sh`
2. `scripts/get_pile_shard1_data.sh`
3. `scripts/create_uspto_pile_mix.sh`

The resulting dataset will reside in `data/mix_uspto.json.zst`

# Training Expert Models

1. Clone this repo and follow the Environment Setup instructions
2. Set up HF authentication: `export HUGGING_FACE_HUB_TOKEN=[FILL ME]`
3. Set up W&B authentication: `export WANDB_API_KEY=[FILL ME]`
4. Edit the variable `DATASET` in script `src/mdel/train.sh` to match a valid
   dataset name on the
   [MDEL HF](https://huggingface.co/Multi-Domain-Expert-Layers).
5. Run the above script in background mode to start the training: `./train.sh &`
6. The trained model should be uploaded to the MDEL HF
