#!/bin/bash
export WANDB_PROJECT=pythia-6.9b-layer-test


for i in 6 18 30 9 20 28 12 19 26 15 22 24 ;
do
        export WANDB_NAME="layer_${i}"
        accelerate launch trainer.py \
                --train_file data/train_data.txt \
                --validation_file data/book_val.txt \
                --model_name_or_path EleutherAI/pythia-6.9b-deduped \
                --output_dir "ckpts/pythia-6.9b/books/layer_${i}" \
                --training_layer ${i} \
                --per_device_train_batch_size 1 \
                --per_device_eval_batch_size 1 \
                --preprocessing_num_workers 32 \
                --learning_rate 1e-4 \
                --block_size 512 \
                --num_train_epochs 1 \
                --gradient_accumulation_steps 8 \
                --do_train \
                --do_eval \
                --overwrite_output_dir \
                --logging_steps 20 \
                --max_steps 1000
done
