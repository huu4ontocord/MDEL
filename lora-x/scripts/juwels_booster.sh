#!/usr/bin/env bash
# SLURM Configuration
#SBATCH --account=cstdl
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition develbooster

# JUWELS Configuration
conda deactivate
module purge
ml use $OTHERSTAGES
module load Stages/2023 GCC/11.3.0  OpenMPI/4.1.4
module load CUDA/11.7

# Network Configuration
export NCCL_IB_TIMEOUT=50
export UCX_RC_TIMEOUT=4s
export NCCL_IB_RETRY_CNT=10
export NCCL_ASYNC_ERROR_HANDLING=1

# Environment Configuration
source /p/home/jusers/clive1/juwels/clive1/miniconda3/bin/activate jordan_lora
export WANDB_API_KEY="d8216641d549f9bb3d0c5074baa39e15dfd55030"
export HUGGING_FACE_HUB_TOKEN="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"
export PYTHONPATH="/p/home/jusers/clive1/juwels/clive1/scaled-rope:$PYTHONPATH"
export TRANSFORMERS_CACHE="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export HF_DATASETS_CACHE="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export HF_HOME="/p/home/jusers/clive1/juwels/clive1/transformers_cache"
export PATH="/p/software/juwelsbooster/stages/2023/software/OpenMPI/4.1.4-GCC-11.3.0/bin:$PATH"

# Juwls specific env
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export WANDB_MODE="offline"
export TRANSFORMERS_OFFLINE=1

# SLURM Host Configuration
hostfile='/p/home/jusers/clive1/juwels/hostfiles/hostfile.txt'
rm $hostfile

for i in `scontrol show hostnames $SLURM_NODELIST`
do
    echo $i slots=4 >>$hostfile
done

export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=12802
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`
export DLTS_HOSTFILE=$hostfile


# Print System Information
echo "GPUs available to job: $SLURM_JOB_GPUS"
echo "Total tasks: $SLURM_NTASKS"

deepspeed --master_port 12802 \
          --launcher slurm \
          --hostfile '/p/home/jusers/clive1/juwels/hostfiles/hostfile.txt' \
          --master_addr $MASTER_ADDR \
          --no_ssh_check \
          /p/home/jusers/clive1/juwels/clive1/scaled-rope/finetune.py \
          --output_dir saved_ckpts_32k \
          --configs defaults lora-7b-llama2 \
          --deepspeed
