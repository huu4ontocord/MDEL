#!/bin/bash

DATASET=uspto
TRAINING_LAYER=9,10,11,12,13

export WANDB_PROJECT=pythia-1b-deduped-layer-test-$DATASET
export WANDB_NAME="layer_$TRAINING_LAYER"
accelerate launch trainer.py \
        --dataset_name Multi-Domain-Expert-Layers/$DATASET \
        --model_name_or_path EleutherAI/pythia-1b-deduped \
        --output_dir "ckpts/pythia-1b-deduped/$DATASET/layer_$TRAINING_LAYER" \
        --training_layers $TRAINING_LAYER \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 1 \
        --preprocessing_num_workers 32 \
        --learning_rate 1e-4 \
        --block_size 512 \
        --num_train_epochs 1 \
        --gradient_accumulation_steps 8 \
        --do_train \
        --do_eval \
        --evluation_strategy steps \
        --eval_steps 200 \
        --overwrite_output_dir \
        --logging_steps 20 \
        --max_steps 1000
