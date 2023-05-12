@echo off
accelerate launch trainer_chat.py ^
        --dataset_name Dahoas/full-hh-rlhf ^
        --model_name_or_path EleutherAI/pythia-160m ^
        --output_dir "output_chat" ^
        --training_layers "5,6,7" ^
        --separator "Assistant:" ^
        --prompt_column "prompt" ^
        --answer_column "response" ^
        --per_device_train_batch_size 1 ^
        --per_device_eval_batch_size 8 ^
        --preprocessing_num_workers 32 ^
        --learning_rate 1e-4 ^
        --block_size 512 ^
        --num_train_epochs 1 ^
        --gradient_accumulation_steps 8 ^
        --do_train ^
        --do_eval ^
        --evaluation_strategy steps ^
        --eval_steps 200 ^
        --overwrite_output_dir ^
        --logging_steps 20 ^
        --max_steps 1000

pause
