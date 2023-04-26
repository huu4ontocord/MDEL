# MDEL

Multi-Domain Expert Layers

# Environment Setup

To set up the development environment, run `make setup_dev`. This will setup the
pre-commit hooks.

## Creating Expert Datasets

First, make sure you followed the Environment Setup guidelines.

To mix the USPTO data with Pile data, run the following scripts:

1. `scripts/get_uspto_data.sh`
2. `scripts/get_pile_shard1_data.sh`
3. `scripts/create_uspto_pile_mix.sh`

The resulting dataset will reside in `data/mix_uspto.json.zst`
