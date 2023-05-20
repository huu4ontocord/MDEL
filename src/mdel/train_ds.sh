#!/bin/bash

DATASET=uspto
TRAINING_LAYERS=9,10,11,12,13

export WANDB_PROJECT=pythia-1b-deduped-layer-test-$DATASET
export WANDB_NAME="layer_$TRAINING_LAYERS"
export WANDB_ENTITY=ontocord

# check if venv or conda is activated
if [ -n "$CONDA_DEFAULT_ENV" ] || [ -n "$VIRTUAL_ENV" ]; then
    echo "Virtual environment is activated"
else
    echo "Error: virtual environment is not activated"
    exit 1
fi

deepspeed trainer.py \
        --configs defaults   \
        --dataset_name Multi-Domain-Expert-Layers/$DATASET \
        --model_name_or_path EleutherAI/pythia-1b-deduped \
        --output_dir "ckpts/pythia-1b-deduped/$DATASET/layer_$TRAINING_LAYERS" \
        --training_layers $TRAINING_LAYERS \
        --push_to_hub true \
        --push_to_hub_model_id expert-$DATASET \
        --push_to_hub_organization Multi-Domain-Expert-Layers \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name $WANDB_NAME \
        --validation_splits "validation_pile,validation_domain" \

