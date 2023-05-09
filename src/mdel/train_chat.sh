#!/bin/bash

DATASET=mini-pile-instruct
TRAINING_LAYERS=4,5,6,7,8

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

accelerate launch trainer.py \
        --dataset_name Multi-Domain-Expert-Layers/$DATASET \
        --model_name_or_path EleutherAI/pythia-1b-deduped \
        --output_dir "ckpts/pythia-1b-deduped/$DATASET/layer_$TRAINING_LAYERS" \
        --training_layers $TRAINING_LAYERS \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --preprocessing_num_workers 32 \
        --learning_rate 1e-4 \
        --block_size 512 \
        --num_train_epochs 1 \
        --gradient_accumulation_steps 8 \
        --do_train \
        --do_eval \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --overwrite_output_dir \
        --logging_steps 20 \
        --max_steps 1000 \
        --push_to_hub true \
        --push_to_hub_model_id expert-$DATASET \
        --push_to_hub_organization Multi-Domain-Expert-Layers
