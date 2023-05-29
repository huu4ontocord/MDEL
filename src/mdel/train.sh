#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=small-g
#SBATCH --account=project_462000259
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --partition=small-g

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

accelerate launch trainer.py \
        --configs defaults   \
        --dataset_name Multi-Domain-Expert-Layers/$DATASET \
        --model_name_or_path EleutherAI/pythia-1b-deduped \
        --output_dir "$SCRATCH/jstillerman/ckpts/pythia-1b-deduped/$DATASET/layer_$TRAINING_LAYERS" \
        --training_layers $TRAINING_LAYERS \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 8 \
        --preprocessing_num_workers 32 \
        --learning_rate 1e-4 \
        --block_size 512 \
        --num_train_epochs 1 \
        --max_train_samples 50000 \
        --gradient_accumulation_steps 8 \
        --evaluation_strategy steps \
        --eval_steps 200 \
        --logging_steps 20 \
        --max_steps 1000 \
        --push_to_hub true \
        --push_to_hub_model_id expert-$DATASET-perplexity-investigation \
        --wandb_entity $WANDB_ENTITY \
        --wandb_project $WANDB_PROJECT \
        --wandb_run_name $WANDB_NAME \
        --validation_splits "validation_pile,validation_domain" \
        --dtype "float32" \
        --no_deepspeed
