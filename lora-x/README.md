# Lora-X - A library to support long context multimodal, multilingual, and multi-domain training

Lora-X is intended to create a base model for MDEL. We are aiming at llama2 architecture.

- TODO: confirm a100 40gb training at 70b 8K context on JUWELS
- TODO: work on ways to increase length given memory constraint
   +  test different lora configs and methods for improving perforamnce 
- TODO: use the ntk scaling in the HF 
- TODO: Add LLavaR code - in particular the projection layer
- TODO: integrate bpt and test speed difference
- TODO: add Hummingbird's dynamic token and media loading during training
- TODO: add proxy embeddings (m-clip)

# Credits
This library is based on the code of participants in MDEL
- https://github.com/jordiclive/scaled-rope (which is in turn based on jquesnelle/scaled-rope and Open Assistant code)
- https://github.com/arnavdantuluri/long-context-transformers
- reference for Hummingbird
  

# Multinode LoRA + Flash + DS 
This project adds LoRA, Flash attn patch and DS (Deepspeed) to [Scaled Rope](https://github.com/jquesnelle/scaled-rope) that can be run multinode.

Flash Attention 2 and LLaMA 2 ready 🚀

## Setup and Installation

1. To install the necessary dependencies, use pip to install the required Python packages:

    ```
    pip install -r requirements.txt
    ```

2. Update the `config.yaml` file as per your requirements.

## Usage

To run the application, use the following command:

```
python --configs defaults <your_override_config>
```

Replace `<your_override_config>` with your specific configuration specified in `configs/config.yaml`. Command line arguments can also be overridden.

**Please Note:** This uses the HF recent PR, so models are HF compatible. Linear scaling argument: 'interpolation_factor', i.e. how much you want to scale the model. If set to None will scale `config.max_position_embeddings / 4096`. As this is the default for LLaMA 2.
        

## Data
- Specify packed untokenized datasets on the hub under dataset_names e.g. (`Multi-Domain-Expert-Layers/the_pile_books3_packed_128k`)
- If pretokenized=True, specify a single pre-tokenized dataset on the hub under dataset_names (`conceptofmind/rp-packed-32k-no-filter` for OpenLLaMA)

### Running on a Single Node

Use the following commands for running on a single node:

1. Export the necessary paths:

    ```
    export PYTHONPATH="/mnt/data/jordiclive/scaled-rope:$PYTHONPATH"
    export TRANSFORMERS_CACHE="/mnt/data/jordiclive/transformers_cache"
    export HF_DATASETS_CACHE="/mnt/data/jordiclive/transformers_cache"
    export HF_HOME="/mnt/data/jordiclive/transformers_cache"
    export WANDB_API_KEY=""
    ```

2. Run the script:

    ```
    deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 61500 finetune.py --output_dir saved_ckpts_32k --configs defaults lora-7b-llama2 --deepspeed
    ```

### Running on Multiple Nodes

Example script using the slurm launcher with deepspeed: `scripts/juwels_booster.sh`
