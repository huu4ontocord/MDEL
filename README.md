# MDEL

Multi-Domain Expert Layers

# Environment Setup

To set up the development environment:
1. Create a virtual environment using `venv`
2. run `make setup_dev`. 

## Creating Expert Datasets

First, make sure you followed the Environment Setup guidelines.

To mix the USPTO data with Pile data, run the following scripts:

1. `scripts/get_uspto_data.sh`
2. `scripts/get_pile_shard1_data.sh`
3. `scripts/create_uspto_pile_mix.sh`

The resulting dataset will reside in `data/mix_uspto_all`
